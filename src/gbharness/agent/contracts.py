"""Provider-neutral contracts for visual game agents.

The emulator, game policy, and model vendor intentionally meet only in this module. A
provider receives a screenshot plus a finite action vocabulary and returns one choice; it
never receives direct access to the emulator. That boundary lets the same runner use
Vertex AI, OpenAI Decisions, a local model, or a deterministic test double.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from gbharness.actions import Action


@dataclass(frozen=True, slots=True)
class ActionOption:
    """One action the current game adapter permits."""

    key: str
    description: str


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Everything a model needs for one bounded control decision."""

    goal: str
    state: str
    image_png: bytes
    actions: tuple[ActionOption, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    """Normalized result returned by every model provider."""

    action: str
    confidence: float
    scene: str
    terminal: bool = False
    progress: float | None = None
    note: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: object | None = None


class DecisionProvider(Protocol):
    """A visual classifier that selects exactly one allowed action."""

    @property
    def name(self) -> str: ...

    def decide(self, request: DecisionRequest) -> Decision: ...


@dataclass(frozen=True, slots=True)
class GamePolicy:
    """Game-specific goal and macros consumed by the provider-neutral runner."""

    name: str
    goal: str
    actions: Mapping[str, Action]
    descriptions: Mapping[str, str]

    def __post_init__(self) -> None:
        action_keys = set(self.actions)
        description_keys = set(self.descriptions)
        if not action_keys:
            raise ValueError("a game policy needs at least one action")
        if action_keys != description_keys:
            missing = sorted(action_keys - description_keys)
            extra = sorted(description_keys - action_keys)
            raise ValueError(f"action descriptions differ: missing={missing}, extra={extra}")

    @property
    def options(self) -> tuple[ActionOption, ...]:
        return tuple(ActionOption(key, self.descriptions[key]) for key in self.actions)

    def resolve(self, key: str) -> Action:
        try:
            return self.actions[key]
        except KeyError:
            raise KeyError(f"unknown action {key!r} for policy {self.name!r}") from None

