"""Input backends. `LibretroInputPort` is the real one; `NullPort` is the test double."""

from gbharness.backends.libretro_port import (
    CoreNotFound,
    LibretroInputPort,
    find_core,
    validate_rom,
)
from gbharness.backends.null_port import NullPort, PortEvent

__all__ = [
    "CoreNotFound",
    "LibretroInputPort",
    "NullPort",
    "PortEvent",
    "find_core",
    "validate_rom",
]
