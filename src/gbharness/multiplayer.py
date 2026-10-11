"""Atomic, synchronized control for linked multi-agent emulator sessions."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

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
                raise IndexError(
                    f"player index {player} is outside [0, {self.port.players - 1}]"
                )
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
                canon_sort(self.port.pressed_for(player))
                for player in range(self.port.players)
            )
            return LinkedStepResult(
                frames_advanced=frames,
                total_frames=self._frame,
                held_by_player=held,
                emulation_ended=not live,
            )

    def release_all(self) -> LinkedStepResult:
        return self.step({player: () for player in range(self.port.players)}, 0)

    @staticmethod
    def _reject_opposed(player: int, buttons: frozenset[Button]) -> None:
        for first, second in ((Button.LEFT, Button.RIGHT), (Button.UP, Button.DOWN)):
            if first in buttons and second in buttons:
                raise ValueError(
                    f"player {player + 1} cannot hold opposed directions "
                    f"{first.value}+{second.value}"
                )
