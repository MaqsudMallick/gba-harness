"""Frame timing for the Game Boy Advance.

The unit of time in this package is the **frame**, never the millisecond. The GBA's LCD
controller completes one frame every 280,896 CPU cycles at 16,777,216 Hz, and games sample
the KEYINPUT register once per frame (typically in the VBlank handler). A press shorter
than one frame can be missed entirely, which is why `MIN_SAFE_PRESS_FRAMES` is 2, not 1.

Worth noting for anyone porting reasoning across from the original Game Boy: the refresh
rate is *identical* at 59.7275 Hz. Only the clock speed and the cycles-per-frame differ
(16.78 MHz / 280,896 rather than 4.19 MHz / 70,224), so every frame-timing conclusion
carries over unchanged.

`ms_to_frames` exists for humans reading a guide that quotes milliseconds. Nothing in the
agent-facing API takes milliseconds, so behaviour never depends on host speed.
"""

from __future__ import annotations

import math
from typing import Final, Literal

GBA_CPU_HZ: Final = 16_777_216
"""ARM7TDMI clock, in Hz (2^24)."""

CYCLES_PER_FRAME: Final = 280_896
"""CPU cycles the LCD controller takes to complete one frame, including VBlank."""

FRAME_RATE: Final = GBA_CPU_HZ / CYCLES_PER_FRAME
"""Frames per second: 59.7275..., not 60. Same as the original Game Boy."""

FRAME_MS: Final = 1_000.0 / FRAME_RATE
"""Milliseconds per frame: 16.742..."""

MIN_SAFE_PRESS_FRAMES: Final = 2
"""Shortest press a game can be relied upon to observe.

One frame is a coin flip: whether the game sees it depends on where its KEYINPUT read
falls relative to the press. Two frames guarantees at least one read inside the window.
"""

RoundMode = Literal["nearest", "up", "down"]


def ms_to_frames(ms: float, *, round_mode: RoundMode = "nearest") -> int:
    """Convert milliseconds to whole frames.

    Defaults to nearest. Use ``"up"`` when the duration is a minimum that must not be
    undershot (a hold that has to outlast an animation), and ``"down"`` when it is a
    ceiling.

    Raises:
        ValueError: if ``ms`` is negative or not finite.
    """
    if ms != ms or ms in (float("inf"), float("-inf")):  # NaN or infinity
        raise ValueError(f"ms must be a finite number, got {ms!r}")
    if ms < 0:
        raise ValueError(f"ms must be non-negative, got {ms!r}")
    exact = ms / FRAME_MS
    match round_mode:
        case "nearest":
            # round-half-up, not Python's banker's rounding: 1.5 frames -> 2
            return int(math.floor(exact + 0.5))
        case "up":
            return int(math.ceil(exact))
        case "down":
            return int(exact)
    raise ValueError(f"unknown round_mode {round_mode!r}; expected nearest, up or down")


def frames_to_ms(frames: int) -> float:
    """Convert whole frames to milliseconds of wall-clock game time."""
    if frames < 0:
        raise ValueError(f"frames must be non-negative, got {frames!r}")
    return frames * FRAME_MS
