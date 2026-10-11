"""`InputPort` backed by the mGBA libretro core, driven over ctypes.

Why this and not an in-process Python emulator: there isn't one for the GBA. PyBoy is
DMG/CGB only and cannot load a GBA ROM at all. The PyPI `mgba` bindings ship cp310/cp311
wheels with no sdist, and even on a supported interpreter they fail at import because the
wheel does not bundle the `libmgba.so.0.10` they link against. Building mGBA's bindings
from source needs cmake and pins them to one interpreter.

libretro sidesteps all of it. The ABI is frozen, `ctypes` needs no build step and no
particular Python version, and the input model is a *better* fit than PyBoy's: libretro
**pulls** the held set once per frame through a callback, so `pressed()` is literally what
the game reads. There is no push-an-edge-then-tick asymmetry to get wrong.

Everything non-obvious below was verified against mGBA 0.10.5's core, not inferred:

* ``retro_run()`` advances **exactly one frame** -- 10 calls produced 10 `video_refresh`
  and 10 `input_poll` callbacks.
* The core polls **exactly the ten GBA buttons** and never asks for X or Y.
* A synthesized GBA ROM boots: the HLE BIOS skips the Nintendo logo check that real
  hardware enforces, so no copyrighted data is needed to run tests.
* ``retro_serialize`` / ``retro_unserialize`` work (528,448 bytes), which is what backs
  `Snapshotting`.
* ``RETRO_MEMORY_SYSTEM_RAM`` is **not exposed** (size 0, NULL pointer), so there is no
  reading KEYINPUT back as an independent oracle. With libretro we *are* the input
  hardware; the thing worth asserting is that the core polled us and got the right bits.
* Framebuffer is 240x160 -- the GBA's resolution, not the Game Boy's 160x144.
* **The core cannot be trusted with a malformed ROM.** `retro_load_game` rejects a short
  buffer cleanly, but calling `retro_deinit` afterwards corrupts the process heap
  ("corrupted size vs. prev_size"), and the crash lands later in an unrelated call. So
  this module validates the header itself before the core ever sees the bytes, and never
  deinitializes after a failed load.

Rendering and audio callbacks are no-ops. The screen belongs to an observation layer that
does not exist yet; ignoring it here is also a large speed win.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Self

from gbharness.buttons import Button
from gbharness.port import EmulationEnded

# --- libretro ABI constants (libretro.h) ------------------------------------------
RETRO_API_VERSION: Final = 1

ENV_GET_CAN_DUPE: Final = 3
ENV_GET_SYSTEM_DIRECTORY: Final = 9
ENV_SET_PIXEL_FORMAT: Final = 10
ENV_SET_INPUT_DESCRIPTORS: Final = 11
ENV_GET_VARIABLE: Final = 15
ENV_SET_VARIABLES: Final = 16
ENV_GET_VARIABLE_UPDATE: Final = 17
ENV_GET_SAVE_DIRECTORY: Final = 31
ENV_SET_SYSTEM_AV_INFO: Final = 32
ENV_SET_SUBSYSTEM_INFO: Final = 34
ENV_SET_GEOMETRY: Final = 37
ENV_SET_CORE_OPTIONS_DISPLAY: Final = 55

RETRO_DEVICE_JOYPAD: Final = 1
RETRO_MEMORY_SAVE_RAM: Final = 0

#: GBA button -> libretro joypad id. Note A and B are 8 and 0, not 0 and 1: libretro's
#: numbering follows the SNES pad, so taking them in order would swap them.
JOYPAD_ID: Final[dict[Button, int]] = {
    Button.B: 0,
    Button.SELECT: 2,
    Button.START: 3,
    Button.UP: 4,
    Button.DOWN: 5,
    Button.LEFT: 6,
    Button.RIGHT: 7,
    Button.A: 8,
    Button.L: 10,
    Button.R: 11,
}
_BUTTON_BY_ID: Final[dict[int, Button]] = {v: k for k, v in JOYPAD_ID.items()}

SCREEN_WIDTH: Final = 240
SCREEN_HEIGHT: Final = 160

_CORE_NAMES: Final = (
    "mgba_libretro.so",
    "mgba_libretro.dylib",
    "mgba_libretro.dll",
)

_MULTIPLAYER_CORE_NAMES: Final = (
    "mgba_splitscreen_libretro.so",
    "mgba_splitscreen_libretro.dylib",
    "mgba_splitscreen_libretro.dll",
)

#: Where `scripts/fetch-core.sh` puts the core, relative to the project root.
VENDOR_SUBDIR: Final = Path("vendor") / "libretro"

_SYSTEM_CORE_DIRS: Final = (
    "/usr/lib/x86_64-linux-gnu/libretro",
    "/usr/lib/libretro",
    "/usr/local/lib/libretro",
    os.path.expanduser("~/.config/retroarch/cores"),
    "/Applications/RetroArch.app/Contents/Resources/cores",
    os.path.expanduser("~/Library/Application Support/RetroArch/cores"),
)


def _vendor_dirs() -> list[Path]:
    """Project-local core directories.

    Searched from the working directory upwards, so running from a subdirectory still
    finds the repo's own core, plus one derived from this file's location to cover an
    editable install invoked from elsewhere. A contributor who ran `scripts/fetch-core.sh`
    then needs no environment variable and no root.
    """
    roots: list[Path] = [Path.cwd(), *Path.cwd().parents]
    package_root = Path(__file__).resolve().parents[3]  # src/gbharness/backends -> root
    if package_root not in roots:
        roots.append(package_root)
    return [root / VENDOR_SUBDIR for root in roots]


# ctypes callback signatures. Kept at module scope so the types are created once.
_ENV_T = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)
_VIDEO_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t)
_AUDIO_T = ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int16)
_AUDIO_BATCH_T = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t)
_POLL_T = ctypes.CFUNCTYPE(None)
_STATE_T = ctypes.CFUNCTYPE(
    ctypes.c_int16, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint
)
_MLOG_T = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)


#: Whether `retro_init` has been called on the loaded core. Tracked at module scope
#: because libretro has no context handle and `retro_init` must not run twice.
_initialized = False

GBA_HEADER_SIZE: Final = 0xC0
_FIXED_VALUE_OFFSET: Final = 0xB2
_FIXED_VALUE: Final = 0x96


#: Receives the raw framebuffer pointer once per rendered frame: (ptr, width, height,
#: pitch_in_bytes). Deliberately untyped pixels -- this module does no pixel work, it only
#: offers the attachment point. Pitch is bytes per row and is wider than `width` (the core
#: pads rows), so a consumer must stride by pitch, not by width.
type FrameSink = Callable[[int, int, int, int], None]


class CoreNotFound(RuntimeError):
    """The mGBA libretro core could not be located."""


def validate_rom(data: bytes) -> None:
    """Reject anything that is not plausibly a GBA cartridge image.

    This exists because the core corrupts the heap on the failed-load path (see the module
    docstring), so a bad ROM must never reach it. Two checks are enough to catch everything
    that is not a GBA ROM at all:

    * at least a full 192-byte cartridge header, and
    * the fixed value ``0x96`` at offset 0xB2, which real hardware's BIOS verifies and
      every cartridge that boots therefore carries.

    Raises:
        ValueError: with the reason.
    """
    if len(data) < GBA_HEADER_SIZE:
        raise ValueError(
            f"not a Game Boy Advance ROM: {len(data)} bytes is shorter than the "
            f"{GBA_HEADER_SIZE}-byte cartridge header"
        )
    actual = data[_FIXED_VALUE_OFFSET]
    if actual != _FIXED_VALUE:
        raise ValueError(
            f"not a Game Boy Advance ROM: byte 0x{_FIXED_VALUE_OFFSET:02X} is "
            f"0x{actual:02X}, expected the fixed value 0x{_FIXED_VALUE:02X}"
        )


class _GameInfo(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("meta", ctypes.c_char_p),
    ]


class _Variable(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_char_p)]


class _MLogger(ctypes.Structure):
    _fields_ = [("log", _MLOG_T), ("filter", ctypes.c_void_p)]


def find_core(explicit: str | Path | None = None) -> Path:
    """Locate the platform's mGBA libretro core.

    Search order: the `explicit` argument, then ``$GBHARNESS_LIBRETRO_CORE``, then the
    project's own ``vendor/libretro/`` (see `_vendor_dirs`), then the system core
    directories. Project-local wins over system so a core pinned in the repo is the one
    actually used.

    This is the only core-discovery path in the package. It used to be duplicated in the
    test fixtures, which is precisely how ``vendor/`` came to work under pytest and not
    for ``examples/demo.py``.

    Raises:
        CoreNotFound: naming the three ways to get a core.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("GBHARNESS_LIBRETRO_CORE")
    if env:
        candidates.append(Path(env))
    search_dirs = [*_vendor_dirs(), *(Path(d) for d in _SYSTEM_CORE_DIRS)]
    candidates.extend(d / name for d in search_dirs for name in _CORE_NAMES)
    for path in candidates:
        if path.is_file():
            return path
    raise CoreNotFound(
        "could not find an mGBA libretro core. Get one with any of:\n"
        "  ./scripts/fetch-core.sh            (macOS/Linux, downloads into vendor/)\n"
        "  sudo apt install libretro-mgba     (Debian/Ubuntu, system-wide)\n"
        "  export GBHARNESS_LIBRETRO_CORE=... (point at an existing core)"
    )


def find_multiplayer_core(explicit: str | Path | None = None) -> Path:
    """Locate the linked 2-4 player mGBA Splitscreen libretro core."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("GBHARNESS_MULTIPLAYER_CORE")
    if env:
        candidates.append(Path(env))
    search_dirs = [*_vendor_dirs(), *(Path(d) for d in _SYSTEM_CORE_DIRS)]
    candidates.extend(d / name for d in search_dirs for name in _MULTIPLAYER_CORE_NAMES)
    for path in candidates:
        if path.is_file():
            return path
    raise CoreNotFound(
        "could not find the mGBA Splitscreen libretro core. Build it with "
        "scripts/fetch-multiplayer-core.sh, or set GBHARNESS_MULTIPLAYER_CORE"
    )


class LibretroInputPort:
    """Drive a Game Boy Advance through the mGBA libretro core.

    **One instance per process.** libretro's C API has no context handle -- a core keeps
    its state in globals -- so two ports would silently share one emulator. Constructing
    a second while one is open raises rather than corrupting both.
    """

    _instance: LibretroInputPort | None = None

    def __init__(
        self,
        rom: str | Path | bytes,
        *,
        core: str | Path | None = None,
        system_dir: str | Path | None = None,
        frame_sink: FrameSink | None = None,
        players: int = 1,
        core_options: Mapping[str, str] | None = None,
    ) -> None:
        if LibretroInputPort._instance is not None:
            raise RuntimeError(
                "a LibretroInputPort is already open in this process; libretro cores keep "
                "state in globals, so only one may exist at a time. Close the other first."
            )
        if not 1 <= players <= 4:
            raise ValueError(f"players must be in [1, 4], got {players}")
        self._core_path = find_core(core)
        self._frame_sink = frame_sink
        self._lib = ctypes.CDLL(str(self._core_path))
        self._players = players
        self._held_by_player: list[set[Button]] = [set() for _ in range(players)]
        self._option_values = {key: value.encode() for key, value in (core_options or {}).items()}
        self._closed = False
        self._loaded = False
        self.frames_run = 0
        self.polls = 0
        self.queried_ids: set[int] = set()
        self.queried_ports: set[int] = set()

        self._system_dir = ctypes.c_char_p(str(system_dir or Path.cwd()).encode())
        self._declare()
        self._install_callbacks()

        version = self._lib.retro_api_version()
        if version != RETRO_API_VERSION:
            raise RuntimeError(
                f"{self._core_path.name} reports libretro API version {version}, "
                f"expected {RETRO_API_VERSION}"
            )
        global _initialized
        if not _initialized:
            self._lib.retro_init()
            _initialized = True
        # No retro_deinit() if the load fails: doing so corrupts the heap. The core stays
        # initialized and ready for another load attempt instead.
        self._load(rom)
        LibretroInputPort._instance = self

    # --- construction helpers -------------------------------------------------------

    def _declare(self) -> None:
        lib = self._lib
        lib.retro_api_version.restype = ctypes.c_uint
        lib.retro_load_game.restype = ctypes.c_bool
        lib.retro_load_game.argtypes = [ctypes.POINTER(_GameInfo)]
        lib.retro_serialize_size.restype = ctypes.c_size_t
        lib.retro_serialize.restype = ctypes.c_bool
        lib.retro_serialize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.retro_unserialize.restype = ctypes.c_bool
        lib.retro_unserialize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.retro_get_memory_data.restype = ctypes.c_void_p
        lib.retro_get_memory_data.argtypes = [ctypes.c_uint]
        lib.retro_get_memory_size.restype = ctypes.c_size_t
        lib.retro_get_memory_size.argtypes = [ctypes.c_uint]

    def _install_callbacks(self) -> None:
        """Wire the six libretro callbacks.

        The closures are stored on the instance deliberately: ctypes does not keep a
        reference to a callback it has handed to C, so letting them be collected would
        hand the core dangling function pointers.
        """

        @_ENV_T
        def environment(cmd: int, data: int | None) -> bool:
            if cmd == ENV_SET_PIXEL_FORMAT:
                return True  # we never look at the framebuffer
            if data is None:
                # Every command below writes through `data`. A core that passes NULL for
                # one of them is misbehaving; decline rather than dereference it.
                return False
            if cmd == ENV_GET_CAN_DUPE:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = True
                return True
            if cmd in (ENV_GET_SYSTEM_DIRECTORY, ENV_GET_SAVE_DIRECTORY):
                ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))[0] = self._system_dir
                return True
            if cmd == ENV_GET_VARIABLE:
                variable = ctypes.cast(data, ctypes.POINTER(_Variable))[0]
                if variable.key is None:
                    return False
                value = self._option_values.get(variable.key.decode())
                if value is None:
                    return False
                variable.value = value
                return True
            if cmd == ENV_GET_VARIABLE_UPDATE:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = False
                return True
            return cmd in (
                ENV_SET_INPUT_DESCRIPTORS,
                ENV_SET_VARIABLES,
                ENV_SET_SYSTEM_AV_INFO,
                ENV_SET_SUBSYSTEM_INFO,
                ENV_SET_GEOMETRY,
                ENV_SET_CORE_OPTIONS_DISPLAY,
            )  # unhandled commands fall back to core defaults

        @_VIDEO_T
        def video(data: int | None, width: int, height: int, pitch: int) -> None:
            # No pixel handling here -- the framebuffer belongs to an observation layer
            # that does not exist yet. This is only the seam it will attach to, so that
            # layer never has to reach into this one's callbacks.
            if self._frame_sink is not None and data is not None:
                self._frame_sink(data, width, height, pitch)

        @_AUDIO_T
        def audio(left: int, right: int) -> None:
            pass

        @_AUDIO_BATCH_T
        def audio_batch(data: object, frames: int) -> int:
            return frames  # claim we consumed them all

        @_POLL_T
        def input_poll() -> None:
            self.polls += 1

        @_STATE_T
        def input_state(port: int, device: int, index: int, id_: int) -> int:
            if device != RETRO_DEVICE_JOYPAD or port >= self._players:
                return 0
            self.queried_ids.add(id_)
            self.queried_ports.add(port)
            button = _BUTTON_BY_ID.get(id_)
            return 1 if button is not None and button in self._held_by_player[port] else 0

        self._callbacks = (environment, video, audio, audio_batch, input_poll, input_state)
        self._lib.retro_set_environment(environment)
        self._lib.retro_set_video_refresh(video)
        self._lib.retro_set_audio_sample(audio)
        self._lib.retro_set_audio_sample_batch(audio_batch)
        self._lib.retro_set_input_poll(input_poll)
        self._lib.retro_set_input_state(input_state)

    def _load(self, rom: str | Path | bytes) -> None:
        if isinstance(rom, bytes):
            data, path = rom, b""
        else:
            path_obj = Path(rom)
            data, path = path_obj.read_bytes(), str(path_obj).encode()
        validate_rom(data)
        self._rom_buffer = ctypes.create_string_buffer(data, len(data))
        info = _GameInfo(
            path=path or None,
            data=ctypes.cast(self._rom_buffer, ctypes.c_void_p),
            size=len(data),
            meta=None,
        )
        if not self._lib.retro_load_game(ctypes.byref(info)):
            raise RuntimeError("the core refused to load this ROM")
        self._loaded = True

    # --- InputPort ------------------------------------------------------------------

    def press(self, button: Button) -> None:
        self.press_for(0, button)

    def release(self, button: Button) -> None:
        self.release_for(0, button)

    def advance(self, frames: int) -> bool:
        """Run `frames` frames. Each `retro_run()` is exactly one frame.

        Always returns True while the port is open: unlike a Game Boy cartridge, a
        libretro core has no "emulation ended" signal -- it runs until unloaded.
        """
        if frames < 0:
            raise ValueError(f"frames must be non-negative, got {frames}")
        self._check_open()
        for _ in range(frames):
            self._lib.retro_run()
        self.frames_run += frames
        return True

    def pressed(self) -> frozenset[Button]:
        return self.pressed_for(0)

    @property
    def players(self) -> int:
        """Number of independently addressable controller ports."""
        return self._players

    def press_for(self, player: int, button: Button) -> None:
        """Hold ``button`` for one zero-based player without advancing time."""
        self._check_open()
        self._check_player(player)
        self._held_by_player[player].add(button)

    def release_for(self, player: int, button: Button) -> None:
        """Release ``button`` for one zero-based player without advancing time."""
        self._check_open()
        self._check_player(player)
        self._held_by_player[player].discard(button)

    def pressed_for(self, player: int) -> frozenset[Button]:
        """Return the held buttons for one zero-based player."""
        self._check_player(player)
        return frozenset(self._held_by_player[player])

    # --- Snapshotting ---------------------------------------------------------------

    def snapshot(self) -> bytes:
        self._check_open()
        size = self._lib.retro_serialize_size()
        buffer = ctypes.create_string_buffer(size)
        if not self._lib.retro_serialize(ctypes.cast(buffer, ctypes.c_void_p), size):
            raise RuntimeError("retro_serialize failed")
        return buffer.raw

    def restore(self, state: bytes) -> None:
        self._check_open()
        buffer = ctypes.create_string_buffer(state, len(state))
        if not self._lib.retro_unserialize(ctypes.cast(buffer, ctypes.c_void_p), len(state)):
            raise RuntimeError("retro_unserialize failed (wrong size or core version?)")

    # --- lifecycle ------------------------------------------------------------------

    @property
    def core_path(self) -> Path:
        return self._core_path

    def release_all(self) -> None:
        """Let every held button up. Safe on a closed port."""
        if not self._closed:
            for held in self._held_by_player:
                held.clear()

    def close(self) -> None:
        """Release buttons, unload the ROM and deinit the core. Idempotent."""
        if self._closed:
            return
        global _initialized
        for held in self._held_by_player:
            held.clear()
        self._closed = True
        try:
            if self._loaded:
                self._lib.retro_unload_game()
            if _initialized:
                self._lib.retro_deinit()
                _initialized = False
        finally:
            if LibretroInputPort._instance is self:
                LibretroInputPort._instance = None

    def _check_open(self) -> None:
        if self._closed:
            raise EmulationEnded("port is closed; the core has been deinitialized")

    def _check_player(self, player: int) -> None:
        if not isinstance(player, int) or isinstance(player, bool):
            raise TypeError(f"player must be an int, got {type(player).__name__}")
        if not 0 <= player < self._players:
            raise IndexError(f"player index {player} is outside [0, {self._players - 1}]")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LibretroMultiplayerPort(LibretroInputPort):
    """A linked 2-4 player mGBA session exposed through independent controllers.

    The mGBA Splitscreen core owns all emulator instances and their virtual link cable.
    This class only routes each controller to libretro ports 0-3 and exposes the composite
    framebuffer through the normal frame sink.
    """

    def __init__(
        self,
        rom: str | Path | bytes,
        *,
        players: int = 2,
        core: str | Path | None = None,
        system_dir: str | Path | None = None,
        frame_sink: FrameSink | None = None,
        layout: str = "auto",
        four_swords_assist: bool = False,
    ) -> None:
        if not 2 <= players <= 4:
            raise ValueError(f"multiplayer sessions need 2-4 players, got {players}")
        if layout not in {"auto", "2x1", "1x2", "2x2", "speaker", "focus", "overlay"}:
            raise ValueError(f"unsupported multiplayer layout: {layout!r}")
        super().__init__(
            rom,
            core=find_multiplayer_core(core),
            system_dir=system_dir,
            frame_sink=frame_sink,
            players=players,
            core_options={
                "splitscreen_players": str(players),
                "splitscreen_layout": layout,
                "splitscreen_audio": "mixed",
                "splitscreen_fs_assist": "on" if four_swords_assist else "off",
                "splitscreen_overlays": "on",
            },
        )
        self._silence_verbose_core_logs()

    def _silence_verbose_core_logs(self) -> None:
        """Install a no-op mGBA logger; the development core is otherwise extremely loud."""

        @_MLOG_T
        def discard(
            logger: object,
            category: int,
            level: int,
            message: bytes,
            args: object,
        ) -> None:
            pass

        self._quiet_log_callback = discard
        self._quiet_logger = _MLogger(discard, None)
        self._lib.mLogSetDefaultLogger.argtypes = [ctypes.POINTER(_MLogger)]
        self._lib.mLogSetDefaultLogger(ctypes.byref(self._quiet_logger))

    def start_linked_session(self) -> None:
        """Confirm the core's player-count menu and boot all linked consoles."""
        self.advance(2)
        self.press_for(0, Button.A)
        self.advance(2)
        self.release_for(0, Button.A)
        self.advance(2)

    def close(self) -> None:
        super().close()
        if hasattr(self, "_quiet_logger"):
            self._lib.mLogSetDefaultLogger(None)
