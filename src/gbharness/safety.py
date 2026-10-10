"""Guards against input an agent will confidently ask for and should not get.

Two rules shape this module:

**Every guard that fires is reported back to the caller**, in the result, not just logged.
Silent clamping produces an agent that never learns why its input had no effect.

**Clamp a value; reject a shape.** A single out-of-range number is clamped and reported --
the intent survives. But a *tree* whose total exceeds the per-call ceiling is rejected
outright rather than truncated, because a half-executed sequence leaves the game in an
unpredictable mid-state, which is strictly worse for the caller than a clear error naming
the limit. Truncation looks more forgiving and is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from gbharness.actions import (
    Action,
    Hold,
    Press,
    Release,
    Repeat,
    Sequence,
    Tap,
    Wait,
)
from gbharness.buttons import OPPOSED_PAIRS, Button, canon_sort
from gbharness.timing import FRAME_RATE, MIN_SAFE_PRESS_FRAMES

OpposedPolicy = Literal["allow", "warn", "reject"]


class SafetyError(ValueError):
    """Base for refusals raised by this module."""


class CallTooLarge(SafetyError):
    """One call asked for more frames than `Limits.max_frames_per_call` permits."""


class BudgetExhausted(SafetyError):
    """The episode's cumulative frame budget is spent."""


class OpposedInput(SafetyError):
    """Opposing D-pad directions were held together under a ``reject`` policy."""


@dataclass(frozen=True, slots=True)
class Limits:
    """Policy knobs. The defaults are what an unattended agent should run with.

    Attributes:
        min_hold_frames: floor for any press duration. A 1-frame press is a coin flip
            against the game's once-per-frame joypad poll, so 0- and 1-frame holds are
            raised to this rather than silently doing nothing.
        max_hold_frames: ceiling for a single press duration (~10 s at default).
        max_frames_per_call: ceiling on the total game time one call may consume.
        episode_frame_budget: cumulative ceiling across the session; None for unlimited.
        opposed_pairs: what to do when up+down or left+right end up held together. Real
            hardware allows it (four independent switches) and some games rely on it,
            while others glitch -- so the default surfaces it rather than deciding.
            Only the D-pad axes count: L+R is an ordinary chord several games use.
    """

    min_hold_frames: int = MIN_SAFE_PRESS_FRAMES
    max_hold_frames: int = 600
    max_frames_per_call: int = 600
    episode_frame_budget: int | None = None
    opposed_pairs: OpposedPolicy = "warn"


@dataclass(frozen=True, slots=True)
class Clamp:
    """A single value the guards changed."""

    field: str
    requested: int
    applied: int

    def describe(self) -> str:
        return f"{self.field}: {self.requested} -> {self.applied}"


@dataclass
class GuardReport:
    """Accumulates what the guards did, for the caller's result payload."""

    warnings: list[str] = field(default_factory=list)
    clamps: list[Clamp] = field(default_factory=list)

    def clamp(self, field_name: str, requested: int, lo: int, hi: int) -> int:
        """Clamp `requested` into [lo, hi], recording it if it moved."""
        applied = max(lo, min(hi, requested))
        if applied != requested:
            self.clamps.append(Clamp(field_name, requested, applied))
            self.warnings.append(f"{field_name} clamped from {requested} to {applied}")
        return applied

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def clamped(self) -> dict[str, dict[str, int]]:
        """The clamps as a plain JSON-friendly mapping."""
        return {c.field: {"requested": c.requested, "applied": c.applied} for c in self.clamps}


def total_frames(action: Action) -> int:
    """Game time one action will consume, in frames. Exact, not an estimate."""
    match action:
        case Tap():
            return action.hold_frames + action.gap_frames
        case Hold() | Wait():
            return action.frames
        case Press() | Release():
            return 0
        case Sequence():
            return sum(total_frames(i) for i in action.items)
        case Repeat():
            return total_frames(action.item) * action.times
    raise TypeError(f"not an action: {action!r}")


def normalize(action: Action, limits: Limits, report: GuardReport) -> Action:
    """Clamp an action's durations into policy, recording every change.

    Press durations are floored to `Limits.min_hold_frames`; waits and gaps are not, since
    a zero-frame wait is a legitimate no-op while a zero-frame press is not an input.
    """
    match action:
        case Tap():
            return Tap(
                action.buttons,
                report.clamp(
                    "hold_frames",
                    action.hold_frames,
                    limits.min_hold_frames,
                    limits.max_hold_frames,
                ),
                report.clamp("gap_frames", action.gap_frames, 0, limits.max_frames_per_call),
            )
        case Hold():
            return Hold(
                action.buttons,
                report.clamp(
                    "frames", action.frames, limits.min_hold_frames, limits.max_hold_frames
                ),
            )
        case Wait():
            return Wait(report.clamp("frames", action.frames, 0, limits.max_frames_per_call))
        case Press() | Release():
            return action
        case Sequence():
            return Sequence(tuple(normalize(i, limits, report) for i in action.items))
        case Repeat():
            return Repeat(normalize(action.item, limits, report), action.times)
    raise TypeError(f"not an action: {action!r}")


def check_call_size(action: Action, limits: Limits) -> int:
    """Return the action's total frames, or refuse if it exceeds the per-call ceiling.

    Raises:
        CallTooLarge: with the requested and permitted totals, and the remedy.
    """
    total = total_frames(action)
    if total > limits.max_frames_per_call:
        raise CallTooLarge(
            f"this call would advance {total} frames, over the per-call limit of "
            f"{limits.max_frames_per_call} "
            f"({limits.max_frames_per_call / FRAME_RATE:.1f}s of game time). "
            "Split it into several smaller calls."
        )
    return total


def opposed_conflicts(held: frozenset[Button]) -> tuple[tuple[Button, Button], ...]:
    """Which opposing D-pad pairs are both present in `held`."""
    return tuple(pair for pair in OPPOSED_PAIRS if held.issuperset(pair))


def check_opposed(held: frozenset[Button], limits: Limits, report: GuardReport) -> None:
    """Apply the opposed-pair policy to a set of simultaneously held buttons.

    Raises:
        OpposedInput: under the ``reject`` policy.
    """
    if limits.opposed_pairs == "allow":
        return
    conflicts = opposed_conflicts(held)
    if not conflicts:
        return
    described = ", ".join("+".join(b.value for b in canon_sort(set(p))) for p in conflicts)
    message = (
        f"opposing D-pad directions held together ({described}); real hardware permits "
        "this but many games glitch"
    )
    if limits.opposed_pairs == "reject":
        raise OpposedInput(message)
    report.warn(message)


@dataclass
class EpisodeBudget:
    """Cumulative frame budget for a session.

    Checked *before* a call runs, using the action's exact total, so a refusal happens
    before any input reaches the game rather than halfway through a sequence.
    """

    limit: int | None = None
    consumed: int = 0

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.consumed)

    def check(self, frames: int) -> None:
        """Raises:
        BudgetExhausted: if `frames` would exceed the budget.
        """
        if self.limit is None:
            return
        if self.consumed + frames > self.limit:
            raise BudgetExhausted(
                f"episode frame budget exhausted: {self.consumed}/{self.limit} frames used, "
                f"this call needs {frames}"
            )

    def charge(self, frames: int) -> None:
        self.consumed += frames
