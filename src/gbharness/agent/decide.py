"""Thin wrapper over `POST /v1/decisions`.

Every call to OpenAI in this project goes through here, so a beta API changing shape is a
one-module problem (PLAN-agent-layer.md §10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from gbharness.agent.policy import QUESTIONS

#: Decisions bills input tokens only -- no output, no cache charges.
USD_PER_INPUT_TOKEN = 0.10 / 1_000_000


@dataclass(frozen=True, slots=True)
class Verdict:
    """One turn's answers, flattened into what the loop actually needs."""

    action: str | None
    confidence: float
    screen_kind: str | None
    waiting_for_input: float | None
    refused: tuple[str, ...]
    input_tokens: int
    usd: float
    raw: Any = field(repr=False, default=None)

    @property
    def ok(self) -> bool:
        return self.action is not None and not self.refused


class Decider:
    """Asks the Decisions endpoint what to do next."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.total_input_tokens = 0
        self.total_usd = 0.0
        self.calls = 0

    def decide(self, image_data_url: str, state: str) -> Verdict:
        """One decision from one screen.

        Args:
            image_data_url: a `data:image/png;base64,...` URL. The endpoint accepts no
                other image form -- no hosted URLs, no Files-API ids.
            state: compact run state. Keep it short: there is no prompt caching here, so
                every token is billed fresh each turn.
        """
        decision = self._client.decisions.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": state},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            questions=QUESTIONS,
        )

        action: str | None = None
        confidence = 0.0
        screen_kind: str | None = None
        waiting: float | None = None
        refused: list[str] = []

        for answer in decision.answers:
            if answer.type == "refusal":
                refused.append(answer.name or "?")
            elif answer.name == "action":
                action = str(answer.choice)
                confidence = float(answer.confidence)
            elif answer.name == "screen_kind":
                screen_kind = str(answer.choice)
            elif answer.name == "waiting_for_input":
                waiting = float(answer.probability)

        tokens = int(getattr(decision.usage, "input_tokens", 0) or 0)
        usd = tokens * USD_PER_INPUT_TOKEN
        self.total_input_tokens += tokens
        self.total_usd += usd
        self.calls += 1

        return Verdict(
            action=action,
            confidence=confidence,
            screen_kind=screen_kind,
            waiting_for_input=waiting,
            refused=tuple(refused),
            input_tokens=tokens,
            usd=usd,
            raw=decision,
        )
