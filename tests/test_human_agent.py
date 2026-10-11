from __future__ import annotations

from gbharness.agent.contracts import ActionOption, DecisionRequest
from gbharness.agent.human import HumanActionSource, HumanDecisionProvider


def request() -> DecisionRequest:
    return DecisionRequest(
        goal="cooperate",
        state="P1 waiting",
        image_png=b"png",
        actions=(ActionOption("move_up", "Move north"), ActionOption("wait", "Wait")),
    )


def test_human_provider_consumes_queued_bounded_action() -> None:
    source = HumanActionSource()
    source.submit("move_up")

    decision = HumanDecisionProvider(source, timeout_seconds=0).decide(request())

    assert decision.action == "move_up"
    assert decision.scene == "human_input"
    assert decision.input_tokens == 0


def test_human_provider_rejects_action_outside_policy() -> None:
    source = HumanActionSource()
    source.submit("unsafe_action")

    decision = HumanDecisionProvider(source, timeout_seconds=0).decide(request())

    assert decision.action == "wait"
