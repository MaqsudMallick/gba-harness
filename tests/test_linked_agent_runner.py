from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from gbharness.agent.contracts import Decision, DecisionRequest
from gbharness.agent.human import HumanActionSource, HumanDecisionProvider
from gbharness.agent.linked_runner import (
    LinkedAgentRunner,
    LinkedRunnerConfig,
    PlayerAssignment,
    _meaningful_visual_change,
)
from gbharness.agent.team import TeamDecision, TeamDecisionRequest
from gbharness.agent.zelda import FOUR_SWORDS_POLICY
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession
from gbharness.observability import io_attributes
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


class SequenceComposite:
    def __init__(self, values: list[int]) -> None:
        self._frames = [
            RawFrame(
                data=bytes([value, value]) * (480 * 320),
                width=480,
                height=320,
                pitch=480 * 2,
                index=index,
            )
            for index, value in enumerate(values, start=1)
        ]
        self._index = 0

    def require(self) -> RawFrame:
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return frame


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


class HumanAwarePort(RecordingPort):
    def __init__(self) -> None:
        super().__init__()
        self.human_moved = threading.Event()

    def press_for(self, player: int, button: Button) -> None:
        super().press_for(player, button)
        if player == 0 and button == Button.UP:
            self.human_moved.set()


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


class WaitForHumanProvider:
    def __init__(self, name: str, human_moved: threading.Event) -> None:
        self._name = name
        self.human_moved = human_moved

    @property
    def name(self) -> str:
        return self._name

    def decide(self, request: DecisionRequest) -> Decision:
        del request
        if not self.human_moved.wait(timeout=1):
            raise TimeoutError("human action was not executed during agent reasoning")
        return Decision("wait", 1.0, "room")


class FakeTeamProvider:
    name = "rotating-captain"

    def __init__(self, *, stage: str = "explore") -> None:
        self.requests: list[TeamDecisionRequest] = []
        self.stage = stage

    def decide_team(self, request: TeamDecisionRequest) -> TeamDecision:
        self.requests.append(request)
        actions = ("move_left", "interact", "move_right")
        return TeamDecision(
            decisions={
                player: Decision(action, 0.9, "shared puzzle", note="Follow the joint plan.")
                for player, action in zip(request.ai_players, actions, strict=True)
            },
            human_instruction="P1 hold the left switch.",
            objective="occupy both switches",
            captain="vertex:captain",
            input_tokens=100,
            output_tokens=20,
            human_plan=("interact",),
            stage=self.stage,
        )


@dataclass
class RecordingTelemetry:
    spans: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)
    annotations: dict[str, dict[str, object]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @contextmanager
    def span(
        self,
        name: str,
        kind: str,
        *,
        input_value: object | None = None,
        output_value: object | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        recorded = dict(attributes or {})
        recorded.update(io_attributes(kind, input_value=input_value, output_value=output_value))
        with self._lock:
            self.spans.append((name, kind, recorded))
        yield name

    def annotate(self, span: str, attributes: Mapping[str, object]) -> None:
        with self._lock:
            self.annotations.setdefault(span, {}).update(attributes)

    def record_output(self, span: str, kind: str, value: object) -> None:
        self.annotate(span, io_attributes(kind, output_value=value))

    def log(self, template: str, **values: object) -> None:
        del template, values


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


def test_one_linked_chain_names_every_player_and_shared_execution() -> None:
    telemetry = RecordingTelemetry()
    assigned = assignments([[Decision("wait", 0.9, "room")] for _ in range(4)])
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=1, max_model_calls=4),
        telemetry=telemetry,
    )

    runner.run()

    kinds = {name: kind for name, kind, _attributes in telemetry.spans}
    assert kinds["coordinate_four_player_decisions"] == "CHAIN"
    assert [kinds[f"player_{player}_decision"] for player in range(1, 5)] == [
        "AGENT",
        "AGENT",
        "AGENT",
        "AGENT",
    ]
    assert kinds["verify_team_completion"] == "GUARDRAIL"
    assert kinds["execute_synchronized_controller_actions"] == "TOOL"
    assert telemetry.annotations["coordinate_four_player_decisions"]["episode.model_calls"] == 4


def test_human_participant_is_a_tool_and_does_not_consume_model_budget() -> None:
    telemetry = RecordingTelemetry()
    assigned = list(assignments([[Decision("wait", 0.9, "room")] for _ in range(4)]))
    human_provider = FakeProvider("human:keyboard", [Decision("move_up", 1.0, "human_input")])
    assigned[0] = PlayerAssignment(
        0,
        "human leader",
        human_provider,
        participant_type="human",
    )
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=tuple(assigned),
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=1, max_model_calls=3),
        telemetry=telemetry,
    )

    result = runner.run()

    kinds = {name: kind for name, kind, _attributes in telemetry.spans}
    assert kinds["player_1_human_input"] == "TOOL"
    assert result.model_calls == 3


def test_live_human_action_executes_while_agents_are_reasoning() -> None:
    port = HumanAwarePort()
    source = HumanActionSource()
    source.submit("move_up")
    assigned = (
        PlayerAssignment(
            0,
            "human leader",
            HumanDecisionProvider(source),
            participant_type="human",
        ),
        *(
            PlayerAssignment(
                player,
                f"agent {player + 1}",
                WaitForHumanProvider(f"model-{player}", port.human_moved),
            )
            for player in range(1, 4)
        ),
    )
    runner = LinkedAgentRunner(
        session=LinkedSession(port),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=1, max_model_calls=3),
    )

    result = runner.run()

    assert port.human_moved.is_set()
    assert result.model_calls == 3
    assert result.turns[0].players[0].action == "move_up"


def test_perceptual_stuck_detector_ignores_small_animation_but_detects_movement() -> None:
    previous = np.zeros((160, 240, 3), dtype=np.uint8)
    small_animation = previous.copy()
    small_animation[20:25, 20:25] = 255
    moved_sprite = previous.copy()
    moved_sprite[40:90, 40:90] = 255

    assert not _meaningful_visual_change(previous, small_animation)
    assert _meaningful_visual_change(previous, moved_sprite)


def test_blocked_repeated_movement_escalates_recovery_actions() -> None:
    repeated_move = [Decision("move_down", 1.0, "blocked") for _ in range(3)]
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assignments([repeated_move.copy() for _ in range(4)]),
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=3, max_model_calls=12),
    )

    result = runner.run()

    assert [turn.players[0].action for turn in result.turns] == [
        "move_down",
        "move_up",
        "move_right",
    ]


def test_blocked_repeated_attack_stops_until_the_scene_changes() -> None:
    repeated_attack = [Decision("attack_down", 1.0, "blocked") for _ in range(3)]
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assignments([repeated_attack.copy() for _ in range(4)]),
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=3, max_model_calls=12),
    )

    result = runner.run()

    assert [turn.players[0].action for turn in result.turns] == [
        "attack_down",
        "wait",
        "wait",
    ]


def test_room_plan_is_reused_until_visual_stagnation_requires_replanning() -> None:
    source = HumanActionSource()
    captain = FakeTeamProvider(stage="explore_room")
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=3, max_model_calls=2),
        team_provider=captain,
        visual_plan_override=lambda _stage, _views: {},
    )
    runner._team_stage = "explore_room"

    result = runner.run()

    assert result.model_calls == 2
    assert len(captain.requests) == 2


def test_npc_autopilot_executes_only_the_captains_human_dialogue_plan() -> None:
    source = HumanActionSource()
    captain = FakeTeamProvider()
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    result = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(
            max_turns=1,
            max_model_calls=1,
            autoplay_human_dialogue=True,
        ),
        team_provider=captain,
    ).run()

    assert result.turns[0].players[0].plan == ("interact",)


def test_visual_dialogue_close_advances_to_single_scout_stage() -> None:
    source = HumanActionSource()
    captain = FakeTeamProvider(stage="advance_dialogue")
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    dialogue_states = iter((True, False))
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(
            max_turns=1,
            max_model_calls=1,
            autoplay_human_dialogue=True,
        ),
        team_provider=captain,
        dialogue_detector=lambda _frame: next(dialogue_states),
    )

    result = runner.run()

    assert result.turns[0].team_stage == "probe_target"


def test_visual_probe_overlap_releases_all_agents() -> None:
    source = HumanActionSource()
    captain = FakeTeamProvider(stage="reach_warps")
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=1, max_model_calls=1),
        team_provider=captain,
        probe_target_detector=lambda _views: True,
    )
    runner._team_stage = "probe_target"

    result = runner.run()

    assert captain.requests[0].stage == "reach_warps"
    assert result.turns[0].team_stage == "reach_warps"


def test_model_cannot_complete_before_visual_target_confirmation() -> None:
    source = HumanActionSource()
    captain = FakeTeamProvider(stage="complete")
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=1, max_model_calls=1),
        team_provider=captain,
        team_target_detector=lambda _views: False,
    )
    runner._team_stage = "reach_warps"

    result = runner.run()

    assert result.status == "turn_limit"
    assert result.turns[0].team_stage == "reach_warps"


def test_team_captain_assigns_three_actions_with_one_model_call() -> None:
    telemetry = RecordingTelemetry()
    source = HumanActionSource()
    captain = FakeTeamProvider()
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=1, max_model_calls=1),
        team_provider=captain,
        telemetry=telemetry,
    )

    result = runner.run()

    assert result.model_calls == 1
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert [player.action for player in result.turns[0].players] == [
        "wait",
        "move_left",
        "interact",
        "move_right",
    ]
    assert result.turns[0].team_instruction == "P1 hold the left switch."
    assert result.turns[0].captain == "vertex:captain"
    assert captain.requests[0].ai_players == (1, 2, 3)
    plan_trace = telemetry.annotations["rotating_team_captain_plan"]
    plan_input = json.loads(cast(str, plan_trace["neatlogs.agent.input"]))
    plan_output = json.loads(cast(str, plan_trace["neatlogs.agent.output"]))
    assert plan_input["goal"] == FOUR_SWORDS_POLICY.goal
    assert plan_input["observation"]["sha256"]
    assert plan_output["objective"] == "occupy both switches"
    assert plan_output["decisions"]["P2"]["decision_reason"] == ("Follow the joint plan.")


def test_visual_motor_steps_reuse_one_team_plan_for_the_stage() -> None:
    source = HumanActionSource()
    captain = FakeTeamProvider(stage="reach_npc")
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, StaticComposite()),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=3, max_model_calls=1),
        team_provider=captain,
        visual_plan_override=lambda _stage, _views: {
            1: ("move_up",),
            2: ("wait",),
            3: ("wait",),
        },
    )

    result = runner.run()

    assert result.status == "turn_limit"
    assert result.model_calls == 1
    assert len(result.turns) == 3
    assert len(captain.requests) == 1
    assert [turn.players[1].plan for turn in result.turns] == [
        ("move_up",),
        ("move_up",),
        ("move_up",),
    ]


def test_human_portal_transition_resumes_one_shared_model() -> None:
    source = HumanActionSource()
    source.submit("move_up")
    captain = FakeTeamProvider(stage="reach_warps")
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    reset_calls: list[bool] = []
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(
            FrameLatch,
            SequenceComposite([0, 0, 0, 255, 0, 255, 255]),
        ),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(
            max_turns=4,
            max_model_calls=2,
            continue_after_team_target=True,
        ),
        team_provider=captain,
        visual_plan_override=lambda _stage, _views: {
            1: ("wait",),
            2: ("wait",),
            3: ("wait",),
        },
        team_target_detector=lambda _views: True,
        gameplay_detector=lambda frame: frame.data[0] == 255,
        on_next_room=lambda: reset_calls.append(True),
    )
    runner._team_stage = "reach_warps"

    result = runner.run()

    assert result.status == "turn_limit"
    assert result.model_calls == 2
    assert len(captain.requests) == 2
    assert captain.requests[1].stage == "explore_room"
    assert reset_calls == [True]
    assert result.turns[0].team_stage == "wait_for_human_warp"
    assert result.turns[-1].team_stage == "explore_room"


def test_human_portal_transition_can_complete_one_puzzle() -> None:
    source = HumanActionSource()
    source.submit("move_up")
    captain = FakeTeamProvider(stage="reach_warps")
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    reset_calls: list[bool] = []
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(
            FrameLatch,
            SequenceComposite([0, 0, 0, 255, 0, 255, 255]),
        ),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(
            max_turns=4,
            max_model_calls=2,
            continue_after_team_target=True,
            complete_after_room_transition=True,
        ),
        team_provider=captain,
        visual_plan_override=lambda _stage, _views: {
            1: ("wait",),
            2: ("wait",),
            3: ("wait",),
        },
        team_target_detector=lambda _views: True,
        gameplay_detector=lambda frame: frame.data[0] == 255,
        on_next_room=lambda: reset_calls.append(True),
    )
    runner._team_stage = "reach_warps"

    result = runner.run()

    assert result.status == "completed"
    assert result.model_calls == 1
    assert len(captain.requests) == 1
    assert reset_calls == []
    assert result.turns[-1].team_stage == "wait_for_human_warp"


def test_human_portal_wait_does_not_treat_a_large_camera_change_as_transition() -> None:
    source = HumanActionSource()
    source.submit("move_right")
    assigned = (
        PlayerAssignment(
            0,
            "human",
            HumanDecisionProvider(source, timeout_seconds=0),
            participant_type="human",
        ),
        *assignments([[Decision("wait", 1.0, "unused")] for _ in range(3)]),
    )
    assigned = tuple(
        PlayerAssignment(index, item.role, item.provider, item.participant_type)
        for index, item in enumerate(assigned)
    )
    lost_views: list[tuple[RawFrame, ...]] = []
    runner = LinkedAgentRunner(
        session=LinkedSession(RecordingPort()),
        frames=cast(FrameLatch, SequenceComposite([255] * 20)),
        assignments=assigned,
        policy=FOUR_SWORDS_POLICY,
        config=LinkedRunnerConfig(max_turns=1, max_model_calls=1),
        team_target_stability_detector=lambda _views: False,
        on_team_target_lost=lost_views.append,
    )
    runner._warp_reference = tuple(np.zeros((160, 240, 3), dtype=np.uint8) for _ in range(4))

    result = runner._wait_for_human_warp()

    assert result == "repair"
    assert len(lost_views) == 1
