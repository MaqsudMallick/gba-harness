"""Shared planning contracts for cooperative linked-game agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from gbharness.agent.contracts import ActionOption, Decision


@dataclass(frozen=True, slots=True)
class TeamDecisionRequest:
    goal: str
    state: str
    image_png: bytes
    actions: tuple[ActionOption, ...]
    ai_players: tuple[int, ...]
    turn: int
    player_images: tuple[bytes, ...] = ()
    player_roles: tuple[str, ...] = ()
    stage: str = "explore"


@dataclass(frozen=True, slots=True)
class PlayerProposal:
    """One player's private visual analysis before team coordination."""

    player: int
    provider: str
    scene: str
    position: str
    obstacle: str
    proposed_actions: tuple[str, ...]
    confidence: float
    note: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class TeamDecision:
    decisions: dict[int, Decision]
    human_instruction: str
    objective: str
    captain: str
    input_tokens: int = 0
    output_tokens: int = 0
    plans: dict[int, tuple[str, ...]] = field(default_factory=dict)
    human_plan: tuple[str, ...] = ()
    proposals: tuple[PlayerProposal, ...] = ()
    model_calls: int = 1
    stage: str = "explore"


class TeamDecisionProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def calls_per_plan(self) -> int: ...

    def decide_team(self, request: TeamDecisionRequest) -> TeamDecision: ...
