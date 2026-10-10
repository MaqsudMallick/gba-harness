"""The frame latch.

A *latch*, not a pipeline: at 59.7 fps a ten-minute session is ~36,000 frames, and an agent
will look at a few hundred of them. So the sink does the minimum possible per frame and all
real work happens on demand.

The one thing it must do eagerly is **copy**. The framebuffer pointer libretro hands the
callback is only valid for the duration of that call; holding it and reading later is a
use-after-free that will mostly appear to work, and will produce torn frames under load.
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RawFrame:
    """One captured framebuffer, copied out of the core's memory.

    Attributes:
        data: raw bytes, `height * pitch` long, still in the core's pixel format.
        width: visible pixels per row (240 on GBA).
        height: rows (160).
        pitch: **bytes** per row, which is larger than `width * 2` because the core pads
            rows. Stride by this, never by width, or every row after the first is skewed.
        index: how many frames the latch has seen, for ordering.
    """

    data: bytes
    width: int
    height: int
    pitch: int
    index: int


class FrameLatch:
    """Holds the most recent frame. Attach `sink` to `LibretroInputPort(frame_sink=...)`.

    Thread-safe because the emulator may be ticked from the session's worker thread while
    an observation is read elsewhere; the lock is uncontended in the common case.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: RawFrame | None = None
        self.frames_seen = 0

    def sink(self, ptr: int, width: int, height: int, pitch: int) -> None:
        """libretro video callback. Must stay cheap: this runs every emulated frame."""
        size = height * pitch
        buffer = (ctypes.c_char * size).from_address(ptr)
        frame = RawFrame(
            data=buffer.raw,  # copy now, while the pointer is valid
            width=width,
            height=height,
            pitch=pitch,
            index=self.frames_seen,
        )
        with self._lock:
            self._latest = frame
            self.frames_seen += 1

    def latest(self) -> RawFrame | None:
        """The most recent frame, or None if the core has not rendered one yet."""
        with self._lock:
            return self._latest

    def require(self) -> RawFrame:
        """The most recent frame.

        Raises:
            RuntimeError: if no frame has been rendered. Usually means `advance()` has not
                been called, or the port was built without `frame_sink=latch.sink`.
        """
        frame = self.latest()
        if frame is None:
            raise RuntimeError(
                "no frame captured yet — advance the emulator at least one frame, and "
                "check the port was constructed with frame_sink=<latch>.sink"
            )
        return frame
