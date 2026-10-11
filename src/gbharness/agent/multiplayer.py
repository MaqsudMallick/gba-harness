"""Reusable visual-agent plumbing for one player in a linked session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from gbharness.buttons import Button, parse_set
from gbharness.vision.capture import RawFrame
from gbharness.vision.encode import to_rgb
from gbharness.vision.multiplayer import split_player_frames


@dataclass(frozen=True, slots=True)
class PlayerDecision:
    """One bounded controller decision made from a player's own screen."""

    buttons: frozenset[Button]
    scene: str
    progress: float | None = None
    note: str = ""

    @classmethod
    def holding(
        cls,
        buttons: object,
        *,
        scene: str,
        progress: float | None = None,
        note: str = "",
    ) -> PlayerDecision:
        return cls(parse_set(buttons), scene, progress, note)


class VisualPlayerAgent(Protocol):
    """A game adapter that converts one native player view into held buttons."""

    @property
    def name(self) -> str: ...

    @property
    def decision_interval(self) -> int: ...

    def reset(self) -> None: ...

    def decide(self, rgb: np.ndarray, frame: int) -> PlayerDecision: ...


@dataclass(slots=True)
class AutonomousPlayer:
    """Crop the assigned view and cadence an interchangeable game agent.

    Player numbers are zero-based internally, matching libretro controller ports. The
    frontend owns the shared clock; this object only observes and returns a held set, so
    it can never accidentally advance one linked console ahead of another.
    """

    player: int
    players: int
    agent: VisualPlayerAgent
    last_decision: PlayerDecision = field(
        default_factory=lambda: PlayerDecision(frozenset(), "boot")
    )
    decisions: int = 0
    _next_decision_frame: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 2 <= self.players <= 4:
            raise ValueError(f"players must be in [2, 4], got {self.players}")
        if not 0 <= self.player < self.players:
            raise ValueError(
                f"player must be in [0, {self.players - 1}], got {self.player}"
            )
        if self.agent.decision_interval < 1:
            raise ValueError("agent decision_interval must be positive")

    def reset(self) -> None:
        self.agent.reset()
        self.last_decision = PlayerDecision(frozenset(), "boot")
        self.decisions = 0
        self._next_decision_frame = 0

    def update(self, composite: RawFrame, frame: int) -> PlayerDecision:
        """Return a cached decision, refreshing it on the agent's chosen cadence."""
        if frame < self._next_decision_frame:
            return self.last_decision
        views = split_player_frames(composite, self.players)
        self.last_decision = self.agent.decide(to_rgb(views[self.player]), frame)
        self.decisions += 1
        self._next_decision_frame = frame + self.agent.decision_interval
        return self.last_decision
