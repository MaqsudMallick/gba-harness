"""Long-horizon runner behavior with a deterministic model double."""

from __future__ import annotations

from typing import cast

from gbharness.actions import wait
from gbharness.agent.contracts import Decision, DecisionRequest, GamePolicy
from gbharness.agent.runner import AgentRunner, RunnerConfig
from gbharness.backends.null_port import NullPort
from gbharness.session import InputSession
from gbharness.vision.capture import FrameLatch, RawFrame


class StaticFrames:
    def require(self) -> RawFrame:
        return RawFrame(data=b"\x00" * 8, width=2, height=2, pitch=4, index=0)


class FakeProvider:
    name = "fake"

    def __init__(self, decisions: list[Decision]) -> None:
        self.decisions = iter(decisions)

    def decide(self, request: DecisionRequest) -> Decision:
        assert request.goal == "finish"
        return next(self.decisions)


def policy() -> GamePolicy:
    return GamePolicy(
        name="test-game",
        goal="finish",
        actions={"go": wait(3), "wait": wait(2)},
        descriptions={"go": "Advance.", "wait": "Pause."},
    )


def test_runner_requires_two_terminal_observations() -> None:
    provider = FakeProvider(
        [
            Decision("go", 0.9, "gameplay"),
            Decision("wait", 0.9, "complete", terminal=True),
            Decision("wait", 0.9, "complete", terminal=True),
        ]
    )
    with InputSession(NullPort()) as session:
        runner = AgentRunner(
            session,
            cast(FrameLatch, StaticFrames()),
            provider,
            policy(),
            RunnerConfig(max_turns=5, terminal_confirmations=2),
        )
        result = runner.run()
    assert result.status == "completed"
    assert [turn.action for turn in result.turns] == ["go", "wait"]
    assert result.total_frames == 5


def test_low_confidence_falls_back_to_wait() -> None:
    provider = FakeProvider([Decision("go", 0.2, "gameplay")])
    with InputSession(NullPort()) as session:
        runner = AgentRunner(
            session,
            cast(FrameLatch, StaticFrames()),
            provider,
            policy(),
            RunnerConfig(max_turns=1, confidence_threshold=0.7),
        )
        result = runner.run()
    assert result.turns[0].action == "wait"
    assert result.total_frames == 2
