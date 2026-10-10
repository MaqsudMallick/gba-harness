"""Visual ingestion: emulator frames in, observations out.

See PLAN-vision-layer.md. This package is an optional extra (`gbharness[vision]`); the
input layer deliberately does not depend on it.
"""

from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import SCREEN_HEIGHT, SCREEN_WIDTH, to_data_url, to_png, to_rgb

__all__ = [
    "SCREEN_HEIGHT",
    "SCREEN_WIDTH",
    "FrameLatch",
    "to_data_url",
    "to_png",
    "to_rgb",
]
