"""Vertex adapter parsing without making network calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gbharness.agent.contracts import ActionOption, DecisionRequest
from gbharness.agent.vertex import VertexDecisionProvider


class FakeModels:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.kwargs: dict[str, Any] = {}

    def generate_content(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(
            parsed=self.payload,
            text=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=42,
                candidates_token_count=7,
            ),
        )


def request() -> DecisionRequest:
    return DecisionRequest(
        goal="win",
        state="turn 1",
        image_png=b"not-decoded-by-the-fake",
        actions=(
            ActionOption("go", "Advance."),
            ActionOption("wait", "Wait."),
        ),
    )


def test_vertex_returns_a_normalized_decision() -> None:
    models = FakeModels(
        {
            "action": "go",
            "confidence": 0.9,
            "scene": "gameplay",
            "terminal": False,
            "progress": 0.2,
            "note": "Path is clear.",
        }
    )
    provider = VertexDecisionProvider(
        project="test",
        location="global",
        model="gemini-test",
        client=SimpleNamespace(models=models),
    )
    decision = provider.decide(request())
    assert decision.action == "go"
    assert decision.confidence == 0.9
    assert decision.input_tokens == 42
    assert decision.output_tokens == 7
    assert models.kwargs["model"] == "gemini-test"
    assert models.kwargs["config"].max_output_tokens == 1024
    assert models.kwargs["config"].thinking_config.thinking_budget == 0


def test_vertex_bounds_thinking_for_reasoning_models() -> None:
    models = FakeModels(
        {
            "action": "wait",
            "confidence": 0.8,
            "scene": "menu",
            "terminal": False,
            "progress": None,
            "note": "Waiting for link.",
        }
    )
    provider = VertexDecisionProvider(
        project="test",
        location="global",
        model="gemini-2.5-pro",
        client=SimpleNamespace(models=models),
    )

    provider.decide(request())

    assert models.kwargs["config"].thinking_config.thinking_budget == 128


def test_vertex_rejects_an_action_outside_the_policy() -> None:
    models = FakeModels(
        {
            "action": "cheat",
            "confidence": 1,
            "scene": "gameplay",
            "terminal": False,
            "progress": None,
            "note": "",
        }
    )
    provider = VertexDecisionProvider(
        project="test",
        location="global",
        model="gemini-test",
        client=SimpleNamespace(models=models),
    )
    with pytest.raises(ValueError, match="allowed actions"):
        provider.decide(request())
