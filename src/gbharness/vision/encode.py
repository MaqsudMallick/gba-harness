"""Pixel conversion and PNG encoding.

The core hands us RGB565 with padded rows. Two facts drive everything here:

* **Pitch is not width.** Rows are padded (512 bytes for 240 visible pixels), so the
  conversion must stride by pitch and then slice to width. Getting this wrong produces a
  diagonally skewed image that still looks like a picture, which is the worst kind of bug.
* **A native frame is tiny and nearly free.** 240x160 is 40 patches, ~48 image tokens.
  Upscaling to 3x costs 8.6x the tokens, so scaling is opt-in and nearest-neighbour --
  bilinear smoothing destroys the hard pixel edges that make 8px GBA text legible.
"""

from __future__ import annotations

import base64
import io
from typing import Final

import numpy as np
from PIL import Image

from gbharness.vision.capture import RawFrame

SCREEN_WIDTH: Final = 240
SCREEN_HEIGHT: Final = 160


def to_rgb(frame: RawFrame) -> np.ndarray:
    """Convert a captured frame to an `(h, w, 3)` uint8 RGB array.

    Handles row padding. Channels are expanded from 5/6/5 bits by replicating the high
    bits into the low ones, which is the standard widening and keeps white at 255.
    """
    flat = np.frombuffer(frame.data, dtype=np.uint16)
    stride = frame.pitch // 2  # pitch is bytes; these are 16-bit pixels
    rows = flat[: frame.height * stride].reshape(frame.height, stride)
    pixels = rows[:, : frame.width]  # drop the row padding

    r = ((pixels >> 11) & 0x1F).astype(np.uint8)
    g = ((pixels >> 5) & 0x3F).astype(np.uint8)
    b = (pixels & 0x1F).astype(np.uint8)
    return np.dstack(
        [
            (r << 3) | (r >> 2),
            (g << 2) | (g >> 4),
            (b << 3) | (b >> 2),
        ]
    )


def to_png(frame: RawFrame, *, scale: int = 1) -> bytes:
    """Encode a frame as PNG.

    Args:
        scale: integer upscale factor, nearest-neighbour. Leave at 1 unless an eval shows
            the model needs it — see PLAN-vision-layer.md §10 for the cost.
    """
    if scale < 1:
        raise ValueError(f"scale must be at least 1, got {scale}")
    image = Image.fromarray(to_rgb(frame), mode="RGB")
    if scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale), Image.Resampling.NEAREST
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def to_data_url(frame: RawFrame, *, scale: int = 1) -> str:
    """Encode as a base64 PNG data URL.

    This is the only image form the Decisions endpoint accepts -- not hosted URLs, not
    Files-API ids -- so it is what the agent layer sends every turn.
    """
    return "data:image/png;base64," + base64.b64encode(to_png(frame, scale=scale)).decode()
