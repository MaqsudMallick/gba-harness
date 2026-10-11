"""Extract native per-player observations from a split-screen framebuffer."""

from __future__ import annotations

from gbharness.vision.capture import RawFrame

GBA_WIDTH = 240
GBA_HEIGHT = 160


def crop_frame(frame: RawFrame, x: int, y: int, width: int, height: int) -> RawFrame:
    """Copy a rectangular RGB565 region into a tightly packed ``RawFrame``."""
    if min(x, y, width, height) < 0 or width == 0 or height == 0:
        raise ValueError("crop coordinates must be non-negative and dimensions positive")
    if x + width > frame.width or y + height > frame.height:
        raise ValueError(
            f"crop {(x, y, width, height)} exceeds frame {frame.width}x{frame.height}"
        )
    row_bytes = width * 2
    rows = []
    for row in range(y, y + height):
        start = row * frame.pitch + x * 2
        rows.append(frame.data[start : start + row_bytes])
    return RawFrame(
        data=b"".join(rows),
        width=width,
        height=height,
        pitch=row_bytes,
        index=frame.index,
    )


def split_player_frames(frame: RawFrame, players: int) -> tuple[RawFrame, ...]:
    """Split an mGBA Splitscreen ``auto`` layout into native 240x160 views.

    Two players are side-by-side. Three and four players use a 2x2 grid. The helper is
    intentionally strict so an agent never silently observes the wrong player after a
    layout change.
    """
    if not 2 <= players <= 4:
        raise ValueError(f"players must be in [2, 4], got {players}")
    expected = (480, 160) if players == 2 else (480, 320)
    if (frame.width, frame.height) != expected:
        raise ValueError(
            f"auto layout for {players} players must be {expected[0]}x{expected[1]}, "
            f"got {frame.width}x{frame.height}"
        )
    regions = (
        (0, 0),
        (GBA_WIDTH, 0),
        (0, GBA_HEIGHT),
        (GBA_WIDTH, GBA_HEIGHT),
    )
    return tuple(
        crop_frame(frame, x, y, GBA_WIDTH, GBA_HEIGHT)
        for x, y in regions[:players]
    )
