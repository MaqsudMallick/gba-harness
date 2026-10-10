"""Declarative input actions.

Frozen, hashable, JSON-round-trippable value objects. They are what gets logged to the
transcript and replayed, so they must mean exactly one thing -- no implicit timing, no
ambient state.

`buttons` is always a *set*, never a single button, which makes combos first-class rather
than a special case: ``up+a``, or the ``start+select+a+b`` soft reset.

Validation here is structural only -- an action that could never make sense (a tap with no
buttons, a negative frame count) is rejected at construction. Policy limits (how long a
hold may be, how many frames one call may consume) belong to `safety`, because they are
configurable and they report back to the caller rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gbharness.buttons import Button, canon_sort, parse_set
from gbharness.timing import MIN_SAFE_PRESS_FRAMES

DEFAULT_HOLD_FRAMES = MIN_SAFE_PRESS_FRAMES
DEFAULT_GAP_FRAMES = 6
"""Frames of no input after a tap.

Without a gap, menus with input debouncing read consecutive taps as one continuous hold,
so "press A twice" silently becomes "press A once".
"""


def _frozen(buttons: object) -> frozenset[Button]:
    return parse_set(buttons)


def _check_frames(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative, got {value}")


@dataclass(frozen=True, slots=True)
class Tap:
    """Press buttons, hold briefly, release, then pause.

    The pause is part of the action rather than the caller's problem: a tap that leaves no
    gap behind it is the most common way to make a menu miss the next input.
    """

    buttons: frozenset[Button]
    hold_frames: int = DEFAULT_HOLD_FRAMES
    gap_frames: int = DEFAULT_GAP_FRAMES

    def __post_init__(self) -> None:
        if not self.buttons:
            raise ValueError("Tap needs at least one button")
        _check_frames(self.hold_frames, "hold_frames")
        _check_frames(self.gap_frames, "gap_frames")


@dataclass(frozen=True, slots=True)
class Hold:
    """Press buttons, hold for a span, release. No trailing gap."""

    buttons: frozenset[Button]
    frames: int

    def __post_init__(self) -> None:
        if not self.buttons:
            raise ValueError("Hold needs at least one button")
        _check_frames(self.frames, "frames")


@dataclass(frozen=True, slots=True)
class Press:
    """Press buttons and leave them held. Open-ended -- pair with `Release`."""

    buttons: frozenset[Button]

    def __post_init__(self) -> None:
        if not self.buttons:
            raise ValueError("Press needs at least one button")


@dataclass(frozen=True, slots=True)
class Release:
    """Release buttons. An empty set means *release everything currently held*."""

    buttons: frozenset[Button] = frozenset()


@dataclass(frozen=True, slots=True)
class Wait:
    """Advance the game with no input."""

    frames: int

    def __post_init__(self) -> None:
        _check_frames(self.frames, "frames")


@dataclass(frozen=True, slots=True)
class Sequence:
    """An ordered run of actions, executed as one unit."""

    items: tuple[Action, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("Sequence.items must be a tuple (it has to stay hashable)")


@dataclass(frozen=True, slots=True)
class Repeat:
    """Run one action `times` times."""

    item: Action
    times: int

    def __post_init__(self) -> None:
        if not isinstance(self.times, int) or isinstance(self.times, bool):
            raise TypeError(f"times must be an int, got {type(self.times).__name__}")
        if self.times < 1:
            raise ValueError(f"times must be at least 1, got {self.times}")


type Action = Tap | Hold | Press | Release | Wait | Sequence | Repeat

_BY_TAG: dict[str, type] = {
    "tap": Tap,
    "hold": Hold,
    "press": Press,
    "release": Release,
    "wait": Wait,
    "sequence": Sequence,
    "repeat": Repeat,
}
_TAG_OF: dict[type, str] = {cls: tag for tag, cls in _BY_TAG.items()}


# --- convenience constructors (accept loose button tokens) --------------------------


def tap(
    buttons: object, hold_frames: int = DEFAULT_HOLD_FRAMES, gap_frames: int = DEFAULT_GAP_FRAMES
) -> Tap:
    return Tap(_frozen(buttons), hold_frames, gap_frames)


def hold(buttons: object, frames: int) -> Hold:
    return Hold(_frozen(buttons), frames)


def press(buttons: object) -> Press:
    return Press(_frozen(buttons))


def release(buttons: object = ()) -> Release:
    return Release(_frozen(buttons) if buttons else frozenset())


def wait(frames: int) -> Wait:
    return Wait(frames)


def sequence(*items: Action) -> Sequence:
    return Sequence(tuple(items))


def repeat(item: Action, times: int) -> Repeat:
    return Repeat(item, times)


# --- serialization ------------------------------------------------------------------


def to_dict(action: Action) -> dict[str, Any]:
    """Convert an action to a JSON-serializable dict.

    Button sets are emitted in canonical order so the output is byte-stable -- a transcript
    has to diff cleanly across runs, and `frozenset` iteration order is not guaranteed.
    """
    tag = _TAG_OF[type(action)]
    match action:
        case Tap():
            return {
                "type": tag,
                "buttons": [b.value for b in canon_sort(action.buttons)],
                "hold_frames": action.hold_frames,
                "gap_frames": action.gap_frames,
            }
        case Hold():
            return {
                "type": tag,
                "buttons": [b.value for b in canon_sort(action.buttons)],
                "frames": action.frames,
            }
        case Press() | Release():
            return {"type": tag, "buttons": [b.value for b in canon_sort(action.buttons)]}
        case Wait():
            return {"type": tag, "frames": action.frames}
        case Sequence():
            return {"type": tag, "items": [to_dict(i) for i in action.items]}
        case Repeat():
            return {"type": tag, "item": to_dict(action.item), "times": action.times}
    raise TypeError(f"not an action: {action!r}")  # pragma: no cover


def from_dict(data: dict[str, Any]) -> Action:
    """Rebuild an action from `to_dict` output.

    Raises:
        ValueError: on an unknown or missing ``type``.
        KeyError: if a required field for that type is absent.
    """
    try:
        tag = data["type"]
    except KeyError:
        raise ValueError("action dict has no 'type' key") from None
    if tag not in _BY_TAG:
        known = ", ".join(sorted(_BY_TAG))
        raise ValueError(f"unknown action type {tag!r}; expected one of: {known}")
    match tag:
        case "tap":
            return Tap(_frozen(data["buttons"]), data["hold_frames"], data["gap_frames"])
        case "hold":
            return Hold(_frozen(data["buttons"]), data["frames"])
        case "press":
            return Press(_frozen(data["buttons"]))
        case "release":
            return Release(_frozen(data["buttons"]))
        case "wait":
            return Wait(data["frames"])
        case "sequence":
            return Sequence(tuple(from_dict(i) for i in data["items"]))
        case "repeat":
            return Repeat(from_dict(data["item"]), data["times"])
    raise AssertionError(f"unreachable: {tag}")  # pragma: no cover
