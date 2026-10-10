"""Input layer for an agentic Game Boy Advance harness.

Scope: everything from "an agent asks for a button" down to "the right bits reach the
core's KEYINPUT poll on a known frame". The screen, memory introspection and the agent
loop belong to other layers and are deliberately absent.
"""

from gbharness.buttons import (
    ALL,
    DPAD,
    FACE,
    OPPOSED_PAIRS,
    SHOULDER,
    SYSTEM,
    Button,
    UnknownButtonError,
    canon_sort,
    parse,
    parse_set,
)
from gbharness.timing import (
    CYCLES_PER_FRAME,
    FRAME_MS,
    FRAME_RATE,
    GBA_CPU_HZ,
    MIN_SAFE_PRESS_FRAMES,
    frames_to_ms,
    ms_to_frames,
)

__all__ = [
    "ALL",
    "CYCLES_PER_FRAME",
    "DPAD",
    "FACE",
    "FRAME_MS",
    "FRAME_RATE",
    "GBA_CPU_HZ",
    "MIN_SAFE_PRESS_FRAMES",
    "OPPOSED_PAIRS",
    "SHOULDER",
    "SYSTEM",
    "Button",
    "UnknownButtonError",
    "canon_sort",
    "frames_to_ms",
    "ms_to_frames",
    "parse",
    "parse_set",
]
