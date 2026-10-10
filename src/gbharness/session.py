"""The single writer.

An emulator is not thread-safe, and nothing good happens if two callers tick it. One
`InputSession` owns one port, holds a lock, and is the only thing that mutates either. A
long sequence holds the lock for its whole duration, so an action is atomic -- two callers'
button presses can never interleave into a combo neither asked for.

The session is **synchronous**, guarded by a reentrant lock, which is a deliberate
simplification of the plan's async sketch: the emulator call is blocking CPU work, so an
async lock would buy nothing and cost testability. An async transport (the MCP server)
wraps calls in `asyncio.to_thread`, which preserves the single-writer property because the
lock is what enforces it, not the event loop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from types import TracebackType

from gbharness.actions import Action, Release
from gbharness.buttons import Button, canon_sort
from gbharness.port import InputPort, Snapshotting
from gbharness.safety import (
    EpisodeBudget,
    GuardReport,
    Limits,
    check_call_size,
    check_opposed,
    normalize,
)
from gbharness.scheduler import Scheduler


@dataclass(frozen=True, slots=True)
class ActionResult:
    """What one `submit` call did -- the payload a tool returns to its caller.

    Deliberately contains no game state: no screen, no memory. Conflating "what my input
    did" with "what the game looks like now" is what makes these harnesses hard to evolve,
    and the observation layer is a separate concern that does not exist yet.
    """

    frames_advanced: int
    total_frames: int
    buttons_held: tuple[Button, ...]
    warnings: tuple[str, ...]
    clamped: dict[str, dict[str, int]]
    budget_remaining: int | None
    emulation_ended: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "frames_advanced": self.frames_advanced,
            "total_frames": self.total_frames,
            "buttons_held": [b.value for b in self.buttons_held],
            "warnings": list(self.warnings),
            "clamped": self.clamped,
            "budget_remaining": self.budget_remaining,
            "emulation_ended": self.emulation_ended,
        }


@dataclass
class InputSession:
    """Serializes all input and timekeeping for one emulator.

    Args:
        port: the backend to drive.
        limits: policy. Defaults are the unattended-agent defaults.
    """

    port: InputPort
    limits: Limits = field(default_factory=Limits)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _closed: bool = field(default=False, repr=False)
    scheduler: Scheduler = field(init=False, repr=False)
    budget: EpisodeBudget = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.budget = EpisodeBudget(self.limits.episode_frame_budget)
        self.scheduler = Scheduler(self.port)

    # --- the one mutating entry point -----------------------------------------------

    def submit(self, action: Action) -> ActionResult:
        """Run one action under the lock, applying every guard.

        Guard order matters and is fixed: normalize durations first so the budget checks
        see the values that will actually run; refuse an oversized call before any input
        reaches the game; then let the opposed-pair policy see real held sets as they form.

        Raises:
            CallTooLarge: the call's total frames exceed the per-call ceiling.
            BudgetExhausted: the episode budget cannot cover this call.
            OpposedInput: under a ``reject`` opposed-pair policy.
            EmulationEnded: the session is closed.
        """
        with self._lock:
            self._check_open()
            report = GuardReport()
            normalized = normalize(action, self.limits, report)
            total = check_call_size(normalized, self.limits)
            self.budget.check(total)

            def guard(held: frozenset[Button]) -> None:
                check_opposed(held, self.limits, report)

            self.scheduler.guard = guard
            try:
                run = self.scheduler.run(normalized)
            finally:
                self.scheduler.guard = None
            self.budget.charge(run.frames_advanced)

            return ActionResult(
                frames_advanced=run.frames_advanced,
                total_frames=self.scheduler.frame,
                buttons_held=canon_sort(run.buttons_held),
                warnings=tuple(report.warnings),
                clamped=report.clamped,
                budget_remaining=self.budget.remaining,
                emulation_ended=run.emulation_ended,
            )

    # --- state ----------------------------------------------------------------------

    @property
    def frame(self) -> int:
        """Frames advanced through this session. Monotonic; never rewound by a restore."""
        return self.scheduler.frame

    def pressed(self) -> frozenset[Button]:
        with self._lock:
            return self.port.pressed()

    def release_all(self) -> ActionResult | None:
        """Let every held button up. Returns None if nothing was held."""
        with self._lock:
            if not self.port.pressed():
                return None
            return self.submit(Release())

    # --- snapshots ------------------------------------------------------------------

    def snapshot(self) -> bytes:
        """Capture emulator state for later rollback.

        Raises:
            TypeError: if the backend does not support snapshots.
        """
        with self._lock:
            self._check_open()
            if not isinstance(self.port, Snapshotting):
                raise TypeError(f"{type(self.port).__name__} does not support snapshots")
            return self.port.snapshot()

    def restore(self, state: bytes) -> None:
        """Roll the emulator back. The session clock is *not* rewound.

        The clock is the transcript's ordering key, so rewinding it would make two
        different moments share a frame number and break replay. Game time went back;
        session time did not.
        """
        with self._lock:
            self._check_open()
            if not isinstance(self.port, Snapshotting):
                raise TypeError(f"{type(self.port).__name__} does not support snapshots")
            self.port.restore(state)

    # --- lifecycle ------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release every held button, then close the port. Idempotent."""
        with self._lock:
            if self._closed:
                return
            try:
                for button in canon_sort(self.port.pressed()):
                    self.port.release(button)
            finally:
                self._closed = True
                close = getattr(self.port, "close", None)
                if callable(close):
                    close()

    def _check_open(self) -> None:
        if self._closed:
            from gbharness.port import EmulationEnded

            raise EmulationEnded("session is closed")

    def __enter__(self) -> InputSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # The release is in close(), which runs whatever happened above. A crash
        # mid-hold must not leave the D-pad jammed.
        self.close()
