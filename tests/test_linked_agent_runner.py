from __future__ import annotations

from typing import cast

from gbharness.agent.contracts import Decision, DecisionRequest
from gbharness.agent.linked_runner import (
    LinkedAgentRunner,
    LinkedRunnerConfig,
    PlayerAssignment,
)
from gbharness.agent.zelda import FOUR_SWORDS_POLICY
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession
from gbharness.vision.capture import FrameLatch, RawFrame


class StaticComposite:
    def require(self) -> RawFrame:
        return RawFrame(
            data=b"\x00" * (480 * 320 * 2),
            width=480,
            height=320,
            pitch=480 * 2,
            index=1,
        )


class RecordingPort:
    players = 4

    def __init__(self) -> None:
        self.held = [set[Button]() for _ in range(self.players)]

    def press_for(self, player: int, button: Button) -> None:
        self.held[player].add(button)

    def release_for(self, player: int, button: Button) -> None:
        self.held[player].discard(button)

    def pressed_for(self, player: int) -> frozenset[Button]:
        return frozenset(self.held[player])

    def advance(self, frames: int) -> bool:
        del frames
        return True


class FakeProvider:
    def __init__(self, name: str, decisions: list[Decision]) -> None:
        self._name = name
        self.decisions = iter(decisions)
        self.requests: list[DecisionRequest] = []

    @property
    def name(self) -> str:
        return self._name

    def decide(self, request: DecisionRequest) -> Decision:
        self.requests.append(request)
        return next(self.decisions)


def assignments(decisions: list[list[Decision]]) -> tuple[PlayerAssignment, ...]:
    return tuple(
        PlayerAssignment(index, f"role {index + 1}", FakeProvider(f"model-{index}", values))
        for index, values in enumerate(decisions)
    )


def test_four_agents_observe_private_views_and_act_on_one_clock() -> None:
    assigned = assignments(
        [
            [Decision("move_up", 0.9, "room", input_tokens=10, output_tokens=2)],
            [Decision("move_down", 0.9, "room", input_tokens=11, output_tokens=2)],
            [Decision("move_left", 0.9, "room", input_tokens=12, output_tokens=2)],
            # Low confidence must become the safe wait macro.
            [Decision("move_right", 0.2, "room", input_tokens=13, output_tokens=2)],
        ]
    )
    session = LinkedSession(RecordingPort())
    runner = LinkedAgentRunner(
        session=session,
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=1, max_model_calls=4),
    )

    result = runner.run()

    assert result.status == "turn_limit"
    assert result.model_calls == 4
    assert result.input_tokens == 46
    assert result.output_tokens == 8
    assert result.total_frames == 18
    assert [player.action for player in result.turns[0].players] == [
        "move_up",
        "move_down",
        "move_left",
        "wait",
    ]
    for index, assignment in enumerate(assigned):
        provider = cast(FakeProvider, assignment.provider)
        assert f"Player {index + 1} of 4" in provider.requests[0].state
        assert provider.requests[0].image_png.startswith(b"\x89PNG")


def test_team_completion_requires_consensus_twice() -> None:
    terminal = Decision("wait", 0.9, "room_complete", terminal=True)
    assigned = assignments([[terminal, terminal] for _ in range(4)])
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(
            max_turns=3,
            max_model_calls=8,
            terminal_confirmations=2,
        ),
    )

    result = runner.run()

    assert result.status == "completed"
    assert result.model_calls == 8
    assert len(result.turns) == 1


def test_model_call_budget_stops_before_partial_team_turn() -> None:
    assigned = assignments([[Decision("wait", 1.0, "menu")] for _ in range(4)])
    result = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=2, max_model_calls=3),
    ).run()

    assert result.status == "call_budget"
    assert result.model_calls == 0
    assert result.turns == ()
