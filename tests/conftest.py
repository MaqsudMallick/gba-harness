"""Shared fixtures.

The integration tests need a bootable cartridge but must not depend on a copyrighted ROM,
so one is synthesized: a valid GBA header plus a two-instruction program that branches past
the header and spins. Verified to load in mGBA's libretro core.

That works because the **HLE BIOS skips the Nintendo logo check**. Real hardware verifies
the 156-byte logo at 0x04 and the header complement at 0xBD before handing control to the
cartridge; an emulator running without a BIOS image does not, so zeroed logo bytes boot
fine and no copyrighted data has to be committed or downloaded.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import pytest

from gbharness.backends.libretro_port import CoreNotFound, find_core

ROM_SIZE = 32 * 1024

# ARM instructions, little-endian.
_BRANCH_PAST_HEADER = 0xEA00002E  # b 0xC0
_SPIN = 0xEAFFFFFE  # b . (branch to self)


def build_test_rom(size: int = ROM_SIZE) -> bytes:
    """Build a minimal loadable Game Boy Advance cartridge image.

    The cartridge contains no game logic -- it branches past the header and spins. That is
    enough for input-layer tests, which assert on what the core polls rather than on
    anything a game does with the result.
    """
    rom = bytearray(size)
    rom[0x00:0x04] = _BRANCH_PAST_HEADER.to_bytes(4, "little")
    # 0x04..0x9F is the Nintendo logo on real cartridges; left zeroed (see module docstring)
    rom[0xA0:0xAC] = b"GBHARNESS   "[:12]  # title, 12 bytes
    rom[0xAC:0xB0] = b"GBHA"  # game code
    rom[0xB0:0xB2] = b"00"  # maker code
    rom[0xB2] = 0x96  # fixed value, checked by real BIOS
    rom[0xC0:0xC4] = _SPIN.to_bytes(4, "little")

    # Header complement check over 0xA0..0xBC.
    checksum = 0
    for addr in range(0xA0, 0xBD):
        checksum = (checksum - rom[addr]) & 0xFF
    rom[0xBD] = (checksum - 0x19) & 0xFF
    return bytes(rom)


@pytest.fixture(scope="session")
def test_rom_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Path to a synthesized, loadable test ROM, built once per session."""
    path = tmp_path_factory.mktemp("roms") / "gbharness-test.gba"
    path.write_bytes(build_test_rom())
    return path


@pytest.fixture(scope="session")
def test_rom_bytes() -> bytes:
    """A synthesized, loadable test ROM as raw bytes."""
    return build_test_rom()


@pytest.fixture(scope="session")
def libretro_core() -> pathlib.Path:
    """Path to mgba_libretro.so, or skip the test if it is not available.

    Deliberately just `find_core()`: the fixture used to carry its own copy of the search
    order, which drifted from the library's and made `vendor/` work under pytest but not
    for `examples/demo.py`. One discovery path only.
    """
    try:
        return find_core()
    except CoreNotFound as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def keyinput_rom() -> Any:
    """The `examples/keyinput_rom` module.

    Loaded by path rather than imported, because `examples/` is not a package. It stays in
    `examples/` because it is also the demo's cartridge and documents itself there; there
    is no reason to keep two copies of hand-assembled machine code.
    """
    path = pathlib.Path(__file__).parent.parent / "examples" / "keyinput_rom.py"
    spec = importlib.util.spec_from_file_location("keyinput_rom", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["keyinput_rom"] = module
    spec.loader.exec_module(module)
    return module
