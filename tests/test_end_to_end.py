"""The strongest test available: what the *game* saw, not what we think we sent.

libretro does not expose KEYINPUT for read-back, so unlike a Game Boy harness there is no
register to assert against (PLAN-input-layer.md §8). `examples/keyinput_rom.py` closes that
gap from the other side: a cartridge that writes one indicator pixel per button every frame.
Reading those pixels back out of the framebuffer recovers an exact, independent oracle.

Everything here therefore validates the whole chain at once -- `InputSession` → `safety` →
`scheduler` → `LibretroInputPort` → libretro's input callback → the ARM7 reading KEYINPUT →
VRAM → the frontend's framebuffer.
"""

from __future__ import annotations

import ctypes

import pytest

from gbharness.actions import hold, press, release, repeat, tap, wait
from gbharness.backends import LibretroInputPort
from gbharness.buttons import ALL, CANON_ORDER, Button
from gbharness.session import InputSession

WHITE = 0x4000  # indicators are 0xFFFF or 0x0000; any midpoint discriminates
WARMUP_FRAMES = 6  # the framebuffer is undefined until the cartridge first paints it


class Observed:
    """Records, per frame, the button set the cartridge reported seeing."""

    def __init__(self, bit_order: tuple[str, ...]) -> None:
        self._bits = bit_order
        self.frames: list[frozenset[Button]] = []

    def sink(self, ptr: int, width: int, height: int, pitch: int) -> None:
        row = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16))[0 : len(self._bits)]
        self.frames.append(
            frozenset(Button(self._bits[i]) for i, v in enumerate(row) if v > WHITE)
        )

    def reset(self) -> None:
        self.frames.clear()

    @property
    def peak(self) -> frozenset[Button]:
        return max(self.frames, key=len, default=frozenset())

    def held_frame_count(self, button: Button) -> int:
        return sum(1 for f in self.frames if button in f)


@pytest.fixture
def harness(test_rom_path, libretro_core, keyinput_rom):
    """A session driving the KEYINPUT reporter cartridge, warmed past the blank frames."""
    observed = Observed(keyinput_rom.KEYINPUT_BITS)
    with (
        LibretroInputPort(
            keyinput_rom.build_rom(), core=libretro_core, frame_sink=observed.sink
        ) as port,
        InputSession(port) as session,
    ):
        session.submit(wait(WARMUP_FRAMES))
        observed.reset()
        yield session, observed


def test_idle_means_the_game_sees_nothing(harness):
    session, observed = harness
    session.submit(wait(4))
    assert observed.peak == frozenset()


@pytest.mark.parametrize("button", CANON_ORDER, ids=lambda b: b.value)
def test_every_button_reaches_the_game_and_only_that_button(harness, button):
    """All ten inputs, verified inside the emulated hardware.

    This is what catches a wrong `JOYPAD_ID` mapping end-to-end: a swapped id would show
    up here as the game seeing a different button than the one we asked for.
    """
    session, observed = harness
    session.submit(hold([button.value], 4))
    assert observed.peak == {button}


def test_a_tap_is_held_for_exactly_the_requested_frames(harness):
    session, observed = harness
    session.submit(tap("a", hold_frames=5, gap_frames=3))
    assert observed.held_frame_count(Button.A) == 5
    assert len(observed.frames) == 8


def test_the_minimum_hold_is_two_frames_as_the_game_sees_it(harness):
    """The clamp is not cosmetic: the game must actually observe two frames."""
    session, observed = harness
    session.submit(tap("a", hold_frames=0))
    assert observed.held_frame_count(Button.A) == 2


def test_a_combo_is_simultaneous_inside_the_emulator(harness):
    session, observed = harness
    session.submit(hold(["l", "r"], 3))
    # Every frame of the hold shows *both*, which is what "simultaneous" has to mean.
    assert [f for f in observed.frames if f] == [frozenset({Button.L, Button.R})] * 3


def test_all_ten_buttons_can_be_held_at_once_in_hardware(harness):
    session, observed = harness
    session.submit(hold([b.value for b in CANON_ORDER], 3))
    assert observed.peak == ALL


def test_repeats_are_separated_by_the_gap_from_the_games_point_of_view(harness):
    session, observed = harness
    session.submit(repeat(tap("a", hold_frames=2, gap_frames=4), 3))
    # Collapse the timeline into runs, so three distinct presses must appear as three
    # separate held runs rather than one long one.
    runs = []
    for frame in observed.frames:
        held = Button.A in frame
        if not runs or runs[-1][0] != held:
            runs.append((held, 1))
        else:
            runs[-1] = (held, runs[-1][1] + 1)
    assert [count for held, count in runs if held] == [2, 2, 2]


def test_zero_gap_taps_look_like_one_hold_to_the_game(harness):
    """Why `gap_frames` defaults to 6 rather than 0 -- demonstrated, not asserted by fiat."""
    session, observed = harness
    session.submit(repeat(tap("a", hold_frames=2, gap_frames=0), 3))
    assert observed.held_frame_count(Button.A) == 6
    assert all(f for f in observed.frames), "no idle frame separates them"


def test_run_and_jump_the_game_never_loses_b(harness):
    """The invariant from the scheduler, confirmed inside the emulated hardware."""
    session, observed = harness
    session.submit(press("b"))
    observed.reset()
    session.submit(tap("a", hold_frames=3, gap_frames=2))
    assert all(Button.B in frame for frame in observed.frames), (
        "B was dropped during the tap"
    )
    assert observed.held_frame_count(Button.A) == 3
    session.submit(release())


def test_release_all_is_visible_to_the_game(harness):
    session, observed = harness
    session.submit(press(["up", "a"]))
    session.submit(wait(2))
    assert observed.peak == {Button.UP, Button.A}
    session.submit(release())
    observed.reset()
    session.submit(wait(2))
    assert observed.peak == frozenset()


def test_a_crash_mid_hold_leaves_nothing_held(harness):
    """A crash must not strand a button.

    Asserted at the session boundary rather than in the framebuffer, and deliberately so:
    `InputSession.close()` releases every button and then deinitializes the core in the
    same breath, so no frame is ever run in which the game could observe the release.
    That is the right behaviour -- there is no game left to observe it -- but it means the
    cartridge oracle cannot see this one, and a test that claimed otherwise would be
    asserting a fiction.
    """
    session, observed = harness
    session.submit(press(["left", "a"]))
    assert session.pressed() == {Button.LEFT, Button.A}
    session.close()
    assert session.pressed() == frozenset()


def test_what_the_game_sees_matches_what_the_session_reports(harness):
    """The two views must not drift: our bookkeeping vs the emulated hardware's."""
    session, observed = harness
    for action in (hold(["r"], 3), hold(["up", "b"], 3), tap("start", 4, 2)):
        observed.reset()
        result = session.submit(action)
        # `buttons_held` is post-release, so compare the peak against the action's set.
        assert observed.peak == action.buttons, (
            f"{action}: session sent {set(action.buttons)}, game saw {set(observed.peak)}"
        )
        assert len(observed.frames) == result.frames_advanced
