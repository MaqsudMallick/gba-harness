"""Shared planning contracts for cooperative linked-game agents."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class TeamDecision:
    decisions: dict[int, Decision]
    human_instruction: str
    objective: str
    captain: str
    input_tokens: int = 0
    output_tokens: int = 0


class TeamDecisionProvider(Protocol):
    @property
    def name(self) -> str: ...

    def decide_team(self, request: TeamDecisionRequest) -> TeamDecision: ...
