"""Visual ingestion: emulator frames in, observations out.

See PLAN-vision-layer.md. This package is an optional extra (`gbharness[vision]`); the
input layer deliberately does not depend on it.
"""

from gbharness.vision.capture import FrameLatch, RawFrame
from gbharness.vision.encode import SCREEN_HEIGHT, SCREEN_WIDTH, to_data_url, to_png, to_rgb
from gbharness.vision.multiplayer import crop_frame, split_player_frames

__all__ = [
    "SCREEN_HEIGHT",
    "SCREEN_WIDTH",
    "FrameLatch",
    "RawFrame",
    "crop_frame",
    "split_player_frames",
    "to_data_url",
    "to_png",
    "to_rgb",
]
