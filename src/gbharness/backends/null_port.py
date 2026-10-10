"""A recording `InputPort` with no emulator behind it.

This is where most of the suite runs. Because it records `(frame, op, detail)` triples, a
test can assert that the scheduler emitted exactly the right edges on exactly the right
frames -- which is the property that matters and the one that is invisible if you only
check the final game state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gbharness.buttons import Button
from gbharness.port import EmulationEnded


@dataclass(frozen=True, slots=True)
class PortEvent:
    """One recorded operation.

    Attributes:
        frame: frames elapsed when the operation was issued.
        op: ``"press"``, ``"release"`` or ``"advance"``.
        detail: the button, for edges; the frame count, for an advance.
    """

    frame: int
    op: str
    detail: Button | int

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"({self.frame}, {self.op!r}, {self.detail!r})"


@dataclass(eq=False)
class NullPort:
    """An `InputPort` that records instead of emulating.

    Args:
        ends_after: if set, `advance` reports emulation ended once this many frames have
            elapsed. Used to test that callers stop cleanly mid-sequence.
    """

    ends_after: int | None = None
    frame: int = 0
    events: list[PortEvent] = field(default_factory=list)
    _held: set[Button] = field(default_factory=set, repr=False)
    _closed: bool = field(default=False, repr=False)

    def press(self, button: Button) -> None:
        if self._closed:
            raise EmulationEnded("port is closed")
        if button in self._held:
            return  # idempotent: no duplicate edge
        self._held.add(button)
        self.events.append(PortEvent(self.frame, "press", button))

    def release(self, button: Button) -> None:
        if self._closed:
            raise EmulationEnded("port is closed")
        if button not in self._held:
            return  # idempotent
        self._held.discard(button)
        self.events.append(PortEvent(self.frame, "release", button))

    def advance(self, frames: int) -> bool:
        if frames < 0:
            raise ValueError(f"frames must be non-negative, got {frames}")
        if self._closed:
            raise EmulationEnded("port is closed")
        if frames == 0:
            return self._live()
        self.events.append(PortEvent(self.frame, "advance", frames))
        self.frame += frames
        return self._live()

    def pressed(self) -> frozenset[Button]:
        return frozenset(self._held)

    def close(self) -> None:
        self._closed = True

    def _live(self) -> bool:
        return self.ends_after is None or self.frame < self.ends_after

    # --- Snapshotting -------------------------------------------------------------
    def snapshot(self) -> bytes:
        return str(self.frame).encode()

    def restore(self, state: bytes) -> None:
        self.frame = int(state.decode())

    # --- test affordances ---------------------------------------------------------
    @property
    def edges(self) -> list[tuple[int, str, Button]]:
        """Just the press/release edges, as plain tuples, for readable assertions."""
        return [
            (e.frame, e.op, e.detail)
            for e in self.events
            if e.op in ("press", "release") and isinstance(e.detail, Button)
        ]
