"""Input backends. `LibretroInputPort` is the real one; `NullPort` is the test double."""

from gbharness.backends.libretro_port import (
    CoreNotFound,
    LibretroInputPort,
    LibretroMultiplayerPort,
    find_core,
    find_multiplayer_core,
    validate_rom,
)
from gbharness.backends.null_port import NullPort, PortEvent

__all__ = [
    "CoreNotFound",
    "LibretroInputPort",
    "LibretroMultiplayerPort",
    "NullPort",
    "PortEvent",
    "find_core",
    "find_multiplayer_core",
    "validate_rom",
]
