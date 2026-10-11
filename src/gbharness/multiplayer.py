"""Atomic, synchronized control for linked multi-agent emulator sessions."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from gbharness.actions import Action, Hold, Repeat, Sequence, Tap, Wait
from gbharness.buttons import Button, canon_sort, parse_set


class LinkedInputPort(Protocol):
    """The multi-controller surface implemented by ``LibretroMultiplayerPort``."""

    @property
    def players(self) -> int: ...

    def press_for(self, player: int, button: Button) -> None: ...

    def release_for(self, player: int, button: Button) -> None: ...

    def pressed_for(self, player: int) -> frozenset[Button]: ...

    def advance(self, frames: int) -> bool: ...


@dataclass(frozen=True, slots=True)
class LinkedStepResult:
    """State after one simultaneous controller step."""

    frames_advanced: int
    total_frames: int
    held_by_player: tuple[tuple[Button, ...], ...]
    emulation_ended: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "frames_advanced": self.frames_advanced,
            "total_frames": self.total_frames,
            "held_by_player": [
                [button.value for button in buttons] for buttons in self.held_by_player
            ],
            "emulation_ended": self.emulation_ended,
        }


@dataclass(frozen=True, slots=True)
class LinkedActionResult:
    """Result of executing one timed action per player in shared lockstep."""

    frames_advanced: int
    total_frames: int
    emulation_ended: bool


@dataclass(frozen=True, slots=True)
class _Phase:
    buttons: frozenset[Button]
    frames: int


@dataclass
class LinkedSession:
    """Apply every player's exact held set before advancing one shared clock.

    A normal ``InputSession`` serializes one controller's actions. Linked games instead
    need all agents' decisions to land on the same emulated frame. ``step`` updates every
    named controller first and calls the core clock only once, preserving lockstep.

    Players omitted from ``inputs`` keep their current held set. Pass an empty value to
    release a player explicitly.
    """

    port: LinkedInputPort
    max_step_frames: int = 600
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _frame: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_step_frames < 1:
            raise ValueError("max_step_frames must be positive")

    @property
    def frame(self) -> int:
        return self._frame

    def step(self, inputs: Mapping[int, object], frames: int) -> LinkedStepResult:
        """Set controller states atomically, then advance the linked emulators."""
        if not isinstance(frames, int) or isinstance(frames, bool):
            raise TypeError(f"frames must be an int, got {type(frames).__name__}")
        if not 0 <= frames <= self.max_step_frames:
            raise ValueError(f"frames must be in [0, {self.max_step_frames}], got {frames}")

        desired: dict[int, frozenset[Button]] = {}
        for player, raw_buttons in inputs.items():
            if not isinstance(player, int) or isinstance(player, bool):
                raise TypeError(f"player must be an int, got {type(player).__name__}")
            if not 0 <= player < self.port.players:
                raise IndexError(f"player index {player} is outside [0, {self.port.players - 1}]")
            buttons = parse_set(raw_buttons)
            self._reject_opposed(player, buttons)
            desired[player] = buttons

        with self._lock:
            for player, wanted in desired.items():
                current = self.port.pressed_for(player)
                for button in current - wanted:
                    self.port.release_for(player, button)
                for button in wanted - current:
                    self.port.press_for(player, button)
            live = self.port.advance(frames)
            self._frame += frames
            held = tuple(
                canon_sort(self.port.pressed_for(player)) for player in range(self.port.players)
            )
            return LinkedStepResult(
                frames_advanced=frames,
                total_frames=self._frame,
                held_by_player=held,
                emulation_ended=not live,
            )

    def release_all(self) -> LinkedStepResult:
        return self.step({player: () for player in range(self.port.players)}, 0)

    def submit(self, actions: Mapping[int, Action]) -> LinkedActionResult:
        """Execute independent finite macros without letting linked clocks drift.

        Every action is compiled to held-button phases. The shortest live phase chooses
        the next shared clock boundary, so presses and releases for all players are
        applied before the core advances again. Open-ended ``Press``/``Release`` actions
        are deliberately excluded: an autonomous turn must always end with its buttons
        released, including when another player's macro is longer.
        """
        timelines: dict[int, list[_Phase]] = {}
        for player, action in actions.items():
            if not isinstance(player, int) or isinstance(player, bool):
                raise TypeError(f"player must be an int, got {type(player).__name__}")
            if not 0 <= player < self.port.players:
                raise IndexError(f"player index {player} is outside [0, {self.port.players - 1}]")
            timelines[player] = _compile_phases(action)

        start = self.frame
        offsets = {player: 0 for player in timelines}
        remaining = {
            player: (phases[0].frames if phases else 0) for player, phases in timelines.items()
        }
        ended = False
        while any(offsets[player] < len(timelines[player]) for player in timelines):
            active = {
                player: timelines[player][offsets[player]].buttons
                if offsets[player] < len(timelines[player])
                else frozenset()
                for player in timelines
            }
            duration = min(
                remaining[player]
                for player in timelines
                if offsets[player] < len(timelines[player])
            )
            result = self.step(active, duration)
            ended = result.emulation_ended
            for player in timelines:
                if offsets[player] >= len(timelines[player]):
                    continue
                remaining[player] -= duration
                if remaining[player] == 0:
                    offsets[player] += 1
                    if offsets[player] < len(timelines[player]):
                        remaining[player] = timelines[player][offsets[player]].frames
            if ended:
                break

        # Finite actions never leak held input into the next model turn.
        self.step({player: () for player in timelines}, 0)
        return LinkedActionResult(
            frames_advanced=self.frame - start,
            total_frames=self.frame,
            emulation_ended=ended,
        )

    @staticmethod
    def _reject_opposed(player: int, buttons: frozenset[Button]) -> None:
        for first, second in ((Button.LEFT, Button.RIGHT), (Button.UP, Button.DOWN)):
            if first in buttons and second in buttons:
                raise ValueError(
                    f"player {player + 1} cannot hold opposed directions "
                    f"{first.value}+{second.value}"
                )


def _compile_phases(action: Action) -> list[_Phase]:
    """Flatten a finite declarative action into shared-clock controller phases."""
    match action:
        case Wait():
            return [_Phase(frozenset(), action.frames)] if action.frames else []
        case Hold():
            return [_Phase(action.buttons, action.frames)] if action.frames else []
        case Tap():
            phases = []
            if action.hold_frames:
                phases.append(_Phase(action.buttons, action.hold_frames))
            if action.gap_frames:
                phases.append(_Phase(frozenset(), action.gap_frames))
            return phases
        case Sequence():
            return [phase for item in action.items for phase in _compile_phases(item)]
        case Repeat():
            phases = _compile_phases(action.item)
            return phases * action.times
        case _:
            raise TypeError(
                "linked autonomous actions must be finite Hold, Tap, Wait, Sequence, "
                f"or Repeat values, got {type(action).__name__}"
            )
