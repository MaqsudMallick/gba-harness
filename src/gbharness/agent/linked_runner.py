"""Cost-bounded orchestration for multiple visual agents in one linked game."""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from gbharness.actions import Action
from gbharness.agent.contracts import Decision, DecisionProvider, DecisionRequest, GamePolicy
from gbharness.agent.human import HumanDecisionProvider
from gbharness.agent.team import TeamDecision, TeamDecisionProvider, TeamDecisionRequest
from gbharness.multiplayer import LinkedActionResult, LinkedSession
from gbharness.observability import NullTelemetry, RunTelemetry
from gbharness.vision.capture import FrameLatch, RawFrame
from gbharness.vision.encode import to_png, to_rgb
from gbharness.vision.multiplayer import split_player_frames

LinkedRunStatus = Literal["completed", "turn_limit", "stopped", "call_budget"]


@dataclass(frozen=True, slots=True)
class PlayerAssignment:
    """One model, private viewport, and cooperation role."""

    player: int
    role: str
    provider: DecisionProvider
    participant_type: Literal["agent", "human"] = "agent"


@dataclass(frozen=True, slots=True)
class LinkedRunnerConfig:
    max_turns: int = 10
    max_model_calls: int = 40
    confidence_threshold: float = 0.65
    terminal_confirmations: int = 2
    max_consecutive_errors: int = 3
    autoplay_human_dialogue: bool = False

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.terminal_confirmations < 1:
            raise ValueError("terminal_confirmations must be positive")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be positive")


@dataclass(frozen=True, slots=True)
class PlayerTurnRecord:
    player: int
    provider: str
    action: str
    confidence: float
    scene: str
    terminal: bool
    progress: float | None
    note: str
    repeated_observations: int
    input_tokens: int
    output_tokens: int
    plan: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LinkedTurnRecord:
    turn: int
    frame: int
    frames_advanced: int
    players: tuple[PlayerTurnRecord, ...]
    team_instruction: str = ""
    captain: str = ""
    execution_steps: int = 1
    progress_detected: bool = False
    changed_players: tuple[int, ...] = ()
    team_stage: str = "explore"


@dataclass(frozen=True, slots=True)
class LinkedEpisodeResult:
    status: LinkedRunStatus
    turns: tuple[LinkedTurnRecord, ...]
    total_frames: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    providers: tuple[str, ...]


@dataclass(slots=True)
class LinkedAgentRunner:
    """Observe four private screens, decide concurrently, then act atomically."""

    session: LinkedSession
    frames: FrameLatch
    assignments: tuple[PlayerAssignment, ...]
    policy: GamePolicy
    config: LinkedRunnerConfig = field(default_factory=LinkedRunnerConfig)
    team_provider: TeamDecisionProvider | None = None
    telemetry: RunTelemetry = field(default_factory=NullTelemetry)
    stop_event: threading.Event = field(default_factory=threading.Event)
    on_turn: Callable[[LinkedTurnRecord], None] | None = None
    on_team_plan: Callable[[str], None] | None = None
    dialogue_detector: Callable[[RawFrame], bool] | None = None
    visual_plan_override: (
        Callable[[str, tuple[RawFrame, ...]], dict[int, tuple[str, ...]]] | None
    ) = None
    probe_target_detector: Callable[[tuple[RawFrame, ...]], bool] | None = None
    team_target_detector: Callable[[tuple[RawFrame, ...]], bool] | None = None
    _team_instruction: str = field(default="", init=False, repr=False)
    _team_captain: str = field(default="", init=False, repr=False)
    _team_input_tokens: int = field(default=0, init=False, repr=False)
    _team_output_tokens: int = field(default=0, init=False, repr=False)
    _team_model_calls: int = field(default=0, init=False, repr=False)
    _team_plans: dict[int, tuple[str, ...]] = field(default_factory=dict, init=False, repr=False)
    _team_human_plan: tuple[str, ...] = field(default=(), init=False, repr=False)
    _team_objective: str = field(default="not assigned", init=False, repr=False)
    _team_stage: str = field(default="reach_npc", init=False, repr=False)
    _dialogue_seen: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        players = self.session.port.players
        assigned = [assignment.player for assignment in self.assignments]
        if len(self.assignments) != players or sorted(assigned) != list(range(players)):
            raise ValueError(
                f"assignments must cover each linked player exactly once; got {assigned}"
            )
        if "wait" not in self.policy.actions:
            raise ValueError("linked agent policy needs a safe 'wait' action")

    def run(self) -> LinkedEpisodeResult:
        records: list[LinkedTurnRecord] = []
        recent: list[deque[str]] = [deque(maxlen=6) for _ in self.assignments]
        previous_views: list[np.ndarray | None] = [None] * len(self.assignments)
        repeated = [0] * len(self.assignments)
        errors = [0] * len(self.assignments)
        calls = 0
        input_tokens = 0
        output_tokens = 0
        terminal_count = 0
        status: LinkedRunStatus = "turn_limit"
        model_decisions_per_turn = (
            int(getattr(self.team_provider, "calls_per_plan", 1))
            if self.team_provider is not None
            else sum(assignment.participant_type == "agent" for assignment in self.assignments)
        )

        with self.telemetry.span(
            "coordinate_four_player_decisions",
            "CHAIN",
            input_value={
                "game": self.policy.name,
                "goal": self.policy.goal,
                "participants": [
                    {
                        "player": assignment.player + 1,
                        "type": assignment.participant_type,
                        "provider": assignment.provider.name,
                        "role": assignment.role,
                    }
                    for assignment in self.assignments
                ],
                "turn_limit": self.config.max_turns,
                "model_call_limit": self.config.max_model_calls,
            },
            attributes={
                "game": self.policy.name,
                "participant.count": len(self.assignments),
                "turn.limit": self.config.max_turns,
                "model_call.limit": self.config.max_model_calls,
            },
        ) as episode_span:
            self.telemetry.log(
                "linked episode started for {game} with {players} players",
                game=self.policy.name,
                players=len(self.assignments),
            )
            for turn in range(1, self.config.max_turns + 1):
                if self.stop_event.is_set():
                    status = "stopped"
                    break
                if calls + model_decisions_per_turn > self.config.max_model_calls:
                    status = "call_budget"
                    self.telemetry.log(
                        "model call budget reached at {calls} calls",
                        level="warning",
                        calls=calls,
                    )
                    break

                composite = self.frames.require()
                dialogue_before = (
                    self.dialogue_detector(composite)
                    if self.dialogue_detector is not None
                    else False
                )
                self._dialogue_seen = self._dialogue_seen or dialogue_before
                views = split_player_frames(composite, len(self.assignments))
                if (
                    self._team_stage == "probe_target"
                    and self.probe_target_detector is not None
                    and self.probe_target_detector(views)
                ):
                    self._team_stage = "reach_warps"
                    self.telemetry.log(
                        "Visual P2 target overlap advanced the team to reach_warps"
                    )
                rgbs = tuple(to_rgb(view) for view in views)
                pngs = tuple(to_png(view) for view in views)
                for player, rgb in enumerate(rgbs):
                    previous = previous_views[player]
                    repeated[player] = (
                        repeated[player] + 1
                        if previous is not None and not _meaningful_visual_change(previous, rgb)
                        else 0
                    )
                    previous_views[player] = rgb.copy()

                requests = tuple(
                    DecisionRequest(
                        goal=self.policy.goal,
                        state=self._state_summary(
                            assignment,
                            turn,
                            repeated[assignment.player],
                            recent,
                            records[-1] if records else None,
                        ),
                        image_png=pngs[assignment.player],
                        actions=self.policy.options,
                    )
                    for assignment in self.assignments
                )
                outcomes = self._decide_all(requests, turn, to_png(composite))
                calls += (
                    self._team_model_calls
                    if self.team_provider is not None
                    else model_decisions_per_turn
                )

                decisions: list[Decision] = []
                for assignment, outcome in zip(self.assignments, outcomes, strict=True):
                    player = assignment.player
                    if isinstance(outcome, BaseException):
                        errors[player] += 1
                        self.telemetry.log(
                            "P{player} decision failed on turn {turn}: {error}",
                            level="error",
                            player=player + 1,
                            turn=turn,
                            error=type(outcome).__name__,
                        )
                        if errors[player] >= self.config.max_consecutive_errors:
                            raise RuntimeError(
                                f"P{player + 1} failed {errors[player]} consecutive decisions"
                            ) from outcome
                        decisions.append(
                            Decision(
                                action="wait",
                                confidence=0.0,
                                scene="decision_error",
                                note=f"Recovered from {type(outcome).__name__}.",
                            )
                        )
                    else:
                        errors[player] = 0
                        decisions.append(outcome)
                        input_tokens += outcome.input_tokens
                        output_tokens += outcome.output_tokens
                        self.telemetry.log(
                            "P{player} decided {action} with confidence {confidence}; "
                            "scene={scene}; terminal={terminal}; note={note}",
                            player=player + 1,
                            action=outcome.action,
                            confidence=outcome.confidence,
                            scene=outcome.scene,
                            terminal=outcome.terminal,
                            note=outcome.note,
                        )

                if self.team_provider is not None:
                    input_tokens += self._team_input_tokens
                    output_tokens += self._team_output_tokens

                with self.telemetry.span(
                    "verify_team_completion",
                    "GUARDRAIL",
                    input_value={
                        "question": "do all agents report the objective complete?",
                        "votes": {
                            f"P{assignment.player + 1}": decision.terminal
                            for assignment, decision in zip(
                                self.assignments, decisions, strict=True
                            )
                            if assignment.participant_type == "agent"
                        },
                        "confirmations_required": self.config.terminal_confirmations,
                    },
                    attributes={"turn": turn},
                ) as guardrail_span:
                    terminal_votes = [
                        decision.terminal
                        for assignment, decision in zip(self.assignments, decisions, strict=True)
                        if assignment.participant_type == "agent"
                    ]
                    unanimous = bool(terminal_votes) and all(terminal_votes)
                    terminal_count = terminal_count + 1 if unanimous else 0
                    passed = terminal_count >= self.config.terminal_confirmations
                    self.telemetry.annotate(guardrail_span, {"neatlogs.guardrail.passed": passed})
                    self.telemetry.record_output(
                        guardrail_span,
                        "GUARDRAIL",
                        {
                            "passed": passed,
                            "consecutive_unanimous": terminal_count,
                            "confirmations_required": self.config.terminal_confirmations,
                        },
                    )
                if passed:
                    status = "completed"
                    break

                action_keys = tuple(
                    self._safe_action(
                        decision,
                        player=index,
                        repeated=repeated[index],
                        recent=recent[index],
                    )
                    for index, decision in enumerate(decisions)
                )
                action_plans = self._action_plans(action_keys, decisions)
                if self.visual_plan_override is not None:
                    overrides = self.visual_plan_override(self._team_stage, views)
                    action_plans = tuple(
                        overrides.get(assignment.player, action_plans[index])
                        for index, assignment in enumerate(self.assignments)
                    )
                action_summary = {
                    f"P{assignment.player + 1}": list(action_plans[index])
                    for index, assignment in enumerate(self.assignments)
                }
                execution_stage = self._team_stage
                frame_before = self.session.frame
                controller_input = json.dumps(
                    {
                        "turn": turn,
                        "stage": execution_stage,
                        "frame_before": frame_before,
                        "plans": action_summary,
                    },
                    sort_keys=True,
                )
                with self.telemetry.span(
                    "execute_synchronized_controller_actions",
                    "TOOL",
                    input_value=controller_input,
                    attributes={
                        "neatlogs.tool.name": "linked_controller",
                        "turn": turn,
                        "frame.before": frame_before,
                    },
                ) as controller_span:
                    result, executed_plans = self._execute_plans(action_plans)
                    self.telemetry.annotate(
                        controller_span,
                        {
                            "frame.after": result.total_frames,
                            "frames.advanced": result.frames_advanced,
                            "emulation.ended": result.emulation_ended,
                            "plan.steps": max(len(plan) for plan in executed_plans),
                        },
                    )
                    self.telemetry.record_output(
                        controller_span,
                        "TOOL",
                        {
                            "frame_after": result.total_frames,
                            "frames_advanced": result.frames_advanced,
                            "emulation_ended": result.emulation_ended,
                            "executed_plans": {
                                f"P{assignment.player + 1}": list(executed_plans[index])
                                for index, assignment in enumerate(self.assignments)
                            },
                        },
                    )
                recorded_plans = tuple(
                    (action_keys[index],)
                    if assignment.participant_type == "human"
                    and not self.config.autoplay_human_dialogue
                    else executed_plans[index]
                    for index, assignment in enumerate(self.assignments)
                )
                after_frame = self.frames.require()
                after_views = split_player_frames(after_frame, len(self.assignments))
                after_rgbs = tuple(to_rgb(view) for view in after_views)
                after_pngs = tuple(to_png(view) for view in after_views)
                dialogue_after = (
                    self.dialogue_detector(after_frame)
                    if self.dialogue_detector is not None
                    else False
                )
                if (
                    self._team_stage == "advance_dialogue"
                    and self._dialogue_seen
                    and not dialogue_after
                ):
                    self._team_stage = "probe_target"
                    self.telemetry.log(
                        "Visual dialogue close advanced the team to probe_target"
                    )
                self._dialogue_seen = self._dialogue_seen or dialogue_after
                visual_team_complete = (
                    self._team_stage == "reach_warps"
                    and self.team_target_detector is not None
                    and self.team_target_detector(after_views)
                )
                if visual_team_complete:
                    self._team_stage = "complete"
                    self.telemetry.log(
                        "Visual target overlap confirmed P2 through P4 completion"
                    )
                changed_players = tuple(
                    player + 1
                    for player, (before, after) in enumerate(zip(rgbs, after_rgbs, strict=True))
                    if _meaningful_visual_change(before, after)
                )
                progress_detected = bool(changed_players)
                progress_passed = progress_detected or visual_team_complete
                progress_input = json.dumps(
                    {
                        "turn": turn,
                        "stage": execution_stage,
                        "frame": frame_before,
                        "screens": _image_evidence(pngs),
                        "plans": action_summary,
                    },
                    sort_keys=True,
                )
                progress_output = json.dumps(
                    {
                        "turn": turn,
                        "stage_before": execution_stage,
                        "stage_after": self._team_stage,
                        "frame": result.total_frames,
                        "screens": _image_evidence(after_pngs),
                        "changed_players": list(changed_players),
                        "team_target_complete": visual_team_complete,
                    },
                    sort_keys=True,
                )
                with self.telemetry.span(
                    "verify_visual_plan_progress",
                    "GUARDRAIL",
                    input_value=progress_input,
                    output_value=progress_output,
                    attributes={
                        "turn": turn,
                        "plan.steps": max(len(plan) for plan in executed_plans),
                        "changed.players": ",".join(map(str, changed_players)),
                        "stage.before": execution_stage,
                        "stage.after": self._team_stage,
                    },
                ) as progress_span:
                    self.telemetry.annotate(
                        progress_span,
                        {
                            "neatlogs.guardrail.passed": progress_passed,
                        },
                    )
                record = self._record(
                    turn,
                    tuple(plan[0] for plan in recorded_plans),
                    decisions,
                    repeated,
                    result,
                    recorded_plans,
                    progress_detected,
                    changed_players,
                )
                records.append(record)
                for player, action in enumerate(action_keys):
                    recent[player].append(action)
                self.telemetry.log(
                    "linked turn {turn}: {actions}",
                    turn=turn,
                    actions=", ".join(
                        f"P{index + 1}={action}" for index, action in enumerate(action_keys)
                    ),
                )
                if self.on_turn is not None:
                    self.on_turn(record)
                if visual_team_complete:
                    status = "completed"
                    break
                if result.emulation_ended:
                    status = "stopped"
                    break

            self.telemetry.annotate(
                episode_span,
                {
                    "episode.status": status,
                    "episode.turns": len(records),
                    "episode.model_calls": calls,
                    "episode.input_tokens": input_tokens,
                    "episode.output_tokens": output_tokens,
                },
            )
            self.telemetry.record_output(
                episode_span,
                "CHAIN",
                {
                    "status": status,
                    "turns": len(records),
                    "total_frames": self.session.frame,
                    "model_calls": calls,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "final_stage": self._team_stage,
                    "final_actions": {
                        f"P{player.player + 1}": list(player.plan)
                        for player in records[-1].players
                    }
                    if records
                    else {},
                },
            )

        return LinkedEpisodeResult(
            status=status,
            turns=tuple(records),
            total_frames=self.session.frame,
            model_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            providers=tuple(assignment.provider.name for assignment in self.assignments),
        )

    def _decide_all(
        self,
        requests: tuple[DecisionRequest, ...],
        turn: int,
        composite_png: bytes,
    ) -> tuple[Decision | BaseException, ...]:
        if self.team_provider is not None:
            return self._decide_as_team(requests, turn, composite_png)

        def call(assignment: PlayerAssignment, request: DecisionRequest) -> Decision:
            is_human = assignment.participant_type == "human"
            span_name = (
                f"player_{assignment.player + 1}_human_input"
                if is_human
                else f"player_{assignment.player + 1}_decision"
            )
            span_kind = "TOOL" if is_human else "AGENT"
            attributes: dict[str, object] = {
                "player.number": assignment.player + 1,
                "player.role": assignment.role,
                "participant.type": assignment.participant_type,
                "agent.provider": assignment.provider.name,
                "turn": turn,
                "observation.bytes": len(request.image_png),
                "allowed.actions": ",".join(option.key for option in request.actions),
            }
            if is_human:
                attributes["neatlogs.tool.name"] = "human_keyboard_controller"
            with self.telemetry.span(
                span_name,
                span_kind,
                input_value={
                    "goal": request.goal,
                    "state": request.state,
                    "role": assignment.role,
                    "observation_bytes": len(request.image_png),
                    "allowed_actions": [option.key for option in request.actions],
                },
                attributes=attributes,
            ) as decision_span:
                decision = assignment.provider.decide(request)
                self.telemetry.annotate(
                    decision_span,
                    {
                        "decision.action": decision.action,
                        "decision.confidence": decision.confidence,
                        "decision.scene": decision.scene,
                        "decision.terminal": decision.terminal,
                        "decision.progress": (
                            decision.progress if decision.progress is not None else -1.0
                        ),
                        "decision.note": decision.note,
                        "usage.input_tokens": decision.input_tokens,
                        "usage.output_tokens": decision.output_tokens,
                    },
                )
                self.telemetry.record_output(
                    decision_span,
                    span_kind,
                    {
                        "action": decision.action,
                        "confidence": decision.confidence,
                        "scene": decision.scene,
                        "terminal": decision.terminal,
                        "progress": decision.progress,
                        "note": decision.note,
                        "input_tokens": decision.input_tokens,
                        "output_tokens": decision.output_tokens,
                    },
                )
                return decision

        live_humans = {
            index: assignment
            for index, assignment in enumerate(self.assignments)
            if assignment.participant_type == "human"
            and isinstance(assignment.provider, HumanDecisionProvider)
        }
        async_indexes = [
            index for index in range(len(self.assignments)) if index not in live_humans
        ]
        outcomes: list[Decision | BaseException | None] = [None] * len(self.assignments)
        live_actions: dict[int, list[str]] = {index: [] for index in live_humans}

        with ThreadPoolExecutor(
            max_workers=max(1, len(async_indexes)), thread_name_prefix="linked-agent"
        ) as executor:
            futures = {
                index: executor.submit(
                    contextvars.copy_context().run,
                    call,
                    self.assignments[index],
                    requests[index],
                )
                for index in async_indexes
            }
            while futures:
                for index, assignment in live_humans.items():
                    provider = assignment.provider
                    assert isinstance(provider, HumanDecisionProvider)
                    action = provider.poll_action(timeout=0.05)
                    if action is None:
                        continue
                    if action not in self.policy.actions:
                        action = "wait"
                    with self.telemetry.span(
                        f"player_{assignment.player + 1}_live_action",
                        "TOOL",
                        input_value={
                            "action": action,
                            "while": "teammate agents are reasoning",
                        },
                        attributes={
                            "neatlogs.tool.name": "human_keyboard_controller",
                            "player.number": assignment.player + 1,
                            "turn": turn,
                            "frame.before": self.session.frame,
                        },
                    ) as human_span:
                        result = self.session.submit(
                            {assignment.player: self.policy.resolve(action)}
                        )
                        self.telemetry.annotate(
                            human_span,
                            {
                                "frame.after": result.total_frames,
                                "frames.advanced": result.frames_advanced,
                                "emulation.ended": result.emulation_ended,
                            },
                        )
                        self.telemetry.record_output(
                            human_span,
                            "TOOL",
                            {
                                "frame_after": result.total_frames,
                                "frames_advanced": result.frames_advanced,
                                "emulation_ended": result.emulation_ended,
                            },
                        )
                    live_actions[index].append(action)
                    self.telemetry.log(
                        "P{player} human live action {action} on turn {turn}",
                        player=assignment.player + 1,
                        action=action,
                        turn=turn,
                    )

                completed = [index for index, future in futures.items() if future.done()]
                for index in completed:
                    future = futures.pop(index)
                    try:
                        outcomes[index] = future.result()
                    except BaseException as exc:
                        outcomes[index] = exc

        for index, assignment in live_humans.items():
            actions = live_actions[index]
            last_action = actions[-1] if actions else "wait"
            with self.telemetry.span(
                f"player_{assignment.player + 1}_human_input",
                "TOOL",
                input_value={"actions": actions},
                attributes={
                    "neatlogs.tool.name": "human_keyboard_controller",
                    "player.number": assignment.player + 1,
                    "turn": turn,
                    "human.action_count": len(actions),
                },
            ) as human_summary_span:
                decision = Decision(
                    action=last_action,
                    confidence=1.0,
                    scene="human_live_control",
                    note=f"Executed {len(actions)} live keyboard actions during agent reasoning.",
                )
                self.telemetry.annotate(
                    human_summary_span,
                    {
                        "decision.action": decision.action,
                        "decision.confidence": decision.confidence,
                        "decision.scene": decision.scene,
                        "decision.note": decision.note,
                    },
                )
                self.telemetry.record_output(
                    human_summary_span,
                    "TOOL",
                    {
                        "action": decision.action,
                        "action_count": len(actions),
                        "scene": decision.scene,
                        "note": decision.note,
                    },
                )
                outcomes[index] = decision

        assert all(outcome is not None for outcome in outcomes)
        return tuple(outcome for outcome in outcomes if outcome is not None)

    def _decide_as_team(
        self,
        requests: tuple[DecisionRequest, ...],
        turn: int,
        composite_png: bytes,
    ) -> tuple[Decision | BaseException, ...]:
        team_provider = self.team_provider
        assert team_provider is not None
        ai_players = tuple(
            assignment.player
            for assignment in self.assignments
            if assignment.participant_type == "agent"
        )
        shared_state = "\n\n".join(
            f"P{assignment.player + 1}:\n{requests[index].state}"
            for index, assignment in enumerate(self.assignments)
        )
        team_request = TeamDecisionRequest(
            goal=self.policy.goal,
            state=shared_state,
            image_png=composite_png,
            actions=self.policy.options,
            ai_players=ai_players,
            turn=turn,
            player_images=tuple(request.image_png for request in requests),
            player_roles=tuple(assignment.role for assignment in self.assignments),
            stage=self._team_stage,
        )

        self._team_model_calls = int(getattr(team_provider, "calls_per_plan", 1))
        self._team_plans = {}
        self._team_human_plan = ()
        fallback_input_json = json.dumps(
            {
                "goal": team_request.goal,
                "stage_before": team_request.stage,
                "turn": team_request.turn,
                "shared_state": team_request.state,
                "allowed_actions": [option.key for option in team_request.actions],
                "ai_players": [player + 1 for player in team_request.ai_players],
                "observation": {
                    "mime_type": "image/png",
                    "bytes": len(composite_png),
                    "sha256": hashlib.sha256(composite_png).hexdigest(),
                },
            },
            sort_keys=True,
        )

        def plan() -> TeamDecision:
            with self.telemetry.span(
                "rotating_team_captain_plan",
                "AGENT",
                input_value=fallback_input_json,
                attributes={
                    "team.provider": team_provider.name,
                    "neatlogs.agent.name": "four_swords_team_captain",
                    "neatlogs.agent.role": (
                        "Observe the current game state and choose one coordinated team plan"
                    ),
                    "turn": turn,
                    "ai.players": ",".join(f"P{player + 1}" for player in ai_players),
                    "observation.bytes": len(composite_png),
                },
            ) as plan_span:
                decision = team_provider.decide_team(team_request)
                self.telemetry.annotate(
                    plan_span,
                    {
                        "neatlogs.agent.input": decision.trace_input_json
                        or fallback_input_json,
                        "team.captain": decision.captain,
                        "team.objective": decision.objective,
                        "team.stage": decision.stage,
                        "team.human_instruction": decision.human_instruction,
                        "team.actions": json.dumps(
                            {
                                f"P{player + 1}": item.action
                                for player, item in decision.decisions.items()
                            },
                            sort_keys=True,
                        ),
                        "team.plans": json.dumps(
                            {
                                f"P{player + 1}": list(plan)
                                for player, plan in decision.plans.items()
                            },
                            sort_keys=True,
                        ),
                        "team.human_plan": json.dumps(decision.human_plan),
                        "team.observer_count": len(decision.proposals),
                        "team.model_calls": decision.model_calls,
                        "usage.input_tokens": decision.input_tokens,
                        "usage.output_tokens": decision.output_tokens,
                    },
                )
                self.telemetry.record_output(
                    plan_span,
                    "AGENT",
                    decision.trace_output_json
                    or _team_decision_json(decision, team_request.stage),
                )
                return decision

        live_humans = {
            index: assignment
            for index, assignment in enumerate(self.assignments)
            if assignment.participant_type == "human"
            and isinstance(assignment.provider, HumanDecisionProvider)
        }
        live_actions: dict[int, list[str]] = {index: [] for index in live_humans}
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="team-captain") as executor:
            future = executor.submit(contextvars.copy_context().run, plan)
            while not future.done():
                for index, assignment in live_humans.items():
                    provider = assignment.provider
                    assert isinstance(provider, HumanDecisionProvider)
                    action = provider.poll_action(timeout=0.05)
                    if action is None:
                        continue
                    if action not in self.policy.actions:
                        action = "wait"
                    with self.telemetry.span(
                        f"player_{assignment.player + 1}_live_action",
                        "TOOL",
                        input_value={"action": action, "while": "team captain planning"},
                        attributes={
                            "neatlogs.tool.name": "human_keyboard_controller",
                            "player.number": assignment.player + 1,
                            "turn": turn,
                            "frame.before": self.session.frame,
                        },
                    ) as human_span:
                        result = self.session.submit(
                            {assignment.player: self.policy.resolve(action)}
                        )
                        self.telemetry.annotate(
                            human_span,
                            {
                                "frame.after": result.total_frames,
                                "frames.advanced": result.frames_advanced,
                            },
                        )
                        self.telemetry.record_output(
                            human_span,
                            "TOOL",
                            {
                                "frame_after": result.total_frames,
                                "frames_advanced": result.frames_advanced,
                            },
                        )
                    live_actions[index].append(action)

            team: TeamDecision | BaseException
            try:
                team = future.result()
            except BaseException as exc:
                team = exc

        outcomes: list[Decision | BaseException] = []
        for index, assignment in enumerate(self.assignments):
            if index in live_humans:
                actions = live_actions[index]
                outcomes.append(
                    Decision(
                        action=actions[-1] if actions else "wait",
                        confidence=1.0,
                        scene="human_live_control",
                        note=f"Executed {len(actions)} live keyboard actions.",
                    )
                )
                continue
            if isinstance(team, BaseException):
                outcomes.append(team)
                continue
            decision = team.decisions[assignment.player]
            assigned_input = json.dumps(
                {
                    "turn": turn,
                    "player": assignment.player + 1,
                    "role": assignment.role,
                    "team_captain": team.captain,
                    "team_objective": team.objective,
                    "team_stage": team.stage,
                    "scene": decision.scene,
                    "decision_reason": decision.note,
                    "assigned_plan": list(
                        team.plans.get(assignment.player, (decision.action,))
                    ),
                },
                sort_keys=True,
            )
            assigned_output = json.dumps(
                {
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "terminal": decision.terminal,
                },
                sort_keys=True,
            )
            with self.telemetry.span(
                f"player_{assignment.player + 1}_assigned_team_action",
                "AGENT",
                input_value=assigned_input,
                output_value=assigned_output,
                attributes={
                    "player.number": assignment.player + 1,
                    "team.captain": team.captain,
                    "decision.action": decision.action,
                    "decision.confidence": decision.confidence,
                    "decision.note": decision.note,
                    "turn": turn,
                },
            ):
                pass
            outcomes.append(decision)

        if isinstance(team, BaseException):
            self._team_instruction = "Team captain failed; AI players will wait this turn."
            self._team_captain = team_provider.name
            self._team_input_tokens = 0
            self._team_output_tokens = 0
        else:
            self._team_instruction = team.human_instruction
            self._team_captain = team.captain
            self._team_input_tokens = team.input_tokens
            self._team_output_tokens = team.output_tokens
            self._team_model_calls = team.model_calls
            self._team_plans = team.plans or {
                player: (decision.action,) for player, decision in team.decisions.items()
            }
            self._team_human_plan = team.human_plan
            self._team_objective = team.objective
            self._team_stage = self._next_team_stage(team.stage)
            if self.on_team_plan is not None:
                self.on_team_plan(team.human_instruction)
            self.telemetry.log(
                "Captain {captain} objective={objective}; P1 instruction={instruction}",
                captain=team.captain,
                objective=team.objective,
                instruction=team.human_instruction,
            )
        return tuple(outcomes)

    def _next_team_stage(self, proposed: str) -> str:
        """Accept only forward stage changes from the visual team planner."""
        stages = (
            "reach_npc",
            "advance_dialogue",
            "probe_target",
            "reach_warps",
            "complete",
        )
        if proposed not in stages or self._team_stage not in stages:
            return proposed
        if proposed == "complete" and self.team_target_detector is not None:
            return self._team_stage
        current = stages.index(self._team_stage)
        candidate = stages.index(proposed)
        if candidate <= current:
            return self._team_stage
        return stages[min(current + 1, len(stages) - 1)]

    def _action_plans(
        self,
        action_keys: tuple[str, ...],
        decisions: list[Decision],
    ) -> tuple[tuple[str, ...], ...]:
        plans: list[tuple[str, ...]] = []
        for index, assignment in enumerate(self.assignments):
            if assignment.participant_type == "human":
                plans.append(
                    self._team_human_plan
                    if self.config.autoplay_human_dialogue
                    else ("wait",)
                )
                continue
            proposed = self._team_plans.get(assignment.player, (action_keys[index],))
            if decisions[index].confidence < self.config.confidence_threshold:
                proposed = ("wait",)
            elif action_keys[index] != decisions[index].action:
                proposed = (action_keys[index], "wait")
            plans.append(
                tuple(action if action in self.policy.actions else "wait" for action in proposed)
                or ("wait",)
            )
        return tuple(plans)

    def _execute_plans(
        self,
        plans: tuple[tuple[str, ...], ...],
    ) -> tuple[LinkedActionResult, tuple[tuple[str, ...], ...]]:
        """Execute a short team horizon locally without another model round trip."""
        steps = max(len(plan) for plan in plans)
        executed: list[list[str]] = [[] for _ in self.assignments]
        total_advanced = 0
        ended = False
        for step in range(steps):
            actions: dict[int, Action] = {}
            for index, assignment in enumerate(self.assignments):
                action = plans[index][step] if step < len(plans[index]) else "wait"
                if assignment.participant_type == "human":
                    provider = assignment.provider
                    if isinstance(provider, HumanDecisionProvider):
                        queued = provider.poll_action(timeout=0)
                        if queued is not None:
                            action = queued
                        elif not self.config.autoplay_human_dialogue:
                            action = "wait"
                    else:
                        action = "wait"
                if action not in self.policy.actions:
                    action = "wait"
                executed[index].append(action)
                actions[assignment.player] = self.policy.resolve(action)
            step_result = self.session.submit(actions)
            total_advanced += step_result.frames_advanced
            ended = step_result.emulation_ended
            if ended or self.stop_event.is_set():
                break
        return (
            LinkedActionResult(
                frames_advanced=total_advanced,
                total_frames=self.session.frame,
                emulation_ended=ended,
            ),
            tuple(tuple(plan) for plan in executed),
        )

    def _safe_action(
        self,
        decision: Decision,
        *,
        player: int,
        repeated: int,
        recent: deque[str],
    ) -> str:
        if decision.confidence < self.config.confidence_threshold:
            return "wait"

        action = decision.action
        if (
            repeated < 1
            or not recent
            or action not in tuple(recent)[-3:]
            or not action.startswith("move_")
        ):
            return action

        direction = action.removeprefix("move_")
        opposite = {
            "up": "move_down",
            "down": "move_up",
            "left": "move_right",
            "right": "move_left",
        }
        alternatives = {
            "up": ("move_left", "move_right"),
            "down": ("move_right", "move_left"),
            "left": ("move_down", "move_up"),
            "right": ("move_up", "move_down"),
        }
        if repeated == 1:
            recovery = opposite[direction]
        elif repeated == 2:
            recovery = alternatives[direction][player % 2]
        else:
            recovery = alternatives[direction][(player + 1) % 2]
        if recovery not in self.policy.actions:
            return "wait"
        self.telemetry.log(
            "P{player} blocked-action guard changed {requested} to {recovery} after "
            "{stagnant} stagnant observations",
            level="warning",
            player=player + 1,
            requested=action,
            recovery=recovery,
            stagnant=repeated,
        )
        return recovery

    def _state_summary(
        self,
        assignment: PlayerAssignment,
        turn: int,
        repeated: int,
        recent: list[deque[str]],
        previous: LinkedTurnRecord | None,
    ) -> str:
        last_team_actions = "none"
        if previous is not None:
            last_team_actions = ", ".join(
                f"P{record.player + 1}={record.action}" for record in previous.players
            )
        own_recent = ", ".join(recent[assignment.player]) or "none"
        return (
            f"Game adapter: {self.policy.name}. Use mechanics from this exact game only.\n"
            f"You are Player {assignment.player + 1} of {len(self.assignments)}.\n"
            f"Role: {assignment.role}\n"
            f"Turn: {turn}\n"
            f"Shared emulated frame: {self.session.frame}\n"
            f"Your recent actions: {own_recent}\n"
            f"Team actions last turn: {last_team_actions}\n"
            f"Persistent team objective: {self._team_objective}\n"
            f"Persistent team stage: {self._team_stage}\n"
            f"Consecutive identical private observations: {repeated}\n"
            "A non-zero identical-observation count means your last action did not move the "
            "scene meaningfully. Treat that direction as blocked. Do not attack a solid "
            "wall or ledge. Back away, then route perpendicular to it.\n"
            + (
                "Player 1 is human-controlled. Use your note for one concrete P1 instruction "
                "that complements your chosen action.\n"
                if any(item.participant_type == "human" for item in self.assignments)
                else ""
            )
            + "Coordinate from visible teammate positions and shared game cues. "
            "Choose only one allowed controller macro."
        )

    def _record(
        self,
        turn: int,
        action_keys: tuple[str, ...],
        decisions: list[Decision],
        repeated: list[int],
        result: LinkedActionResult,
        action_plans: tuple[tuple[str, ...], ...],
        progress_detected: bool,
        changed_players: tuple[int, ...],
    ) -> LinkedTurnRecord:
        players = tuple(
            PlayerTurnRecord(
                player=assignment.player,
                provider=assignment.provider.name,
                action=action_keys[index],
                confidence=decisions[index].confidence,
                scene=decisions[index].scene,
                terminal=decisions[index].terminal,
                progress=decisions[index].progress,
                note=decisions[index].note,
                repeated_observations=repeated[assignment.player],
                input_tokens=decisions[index].input_tokens,
                output_tokens=decisions[index].output_tokens,
                plan=action_plans[index],
            )
            for index, assignment in enumerate(self.assignments)
        )
        return LinkedTurnRecord(
            turn=turn,
            frame=result.total_frames,
            frames_advanced=result.frames_advanced,
            players=players,
            team_instruction=self._team_instruction,
            captain=self._team_captain,
            execution_steps=max(len(plan) for plan in action_plans),
            progress_detected=progress_detected,
            changed_players=changed_players,
            team_stage=self._team_stage,
        )


def _meaningful_visual_change(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    changed_pixel_fraction: float = 0.04,
) -> bool:
    """Detect meaningful scene/camera progress, not ordinary sprite animation."""
    if previous.shape != current.shape:
        return True
    difference = np.max(
        np.abs(current.astype(np.int16) - previous.astype(np.int16)),
        axis=2,
    )
    changed = float(np.count_nonzero(difference > 32)) / float(difference.size)
    return changed >= changed_pixel_fraction


def _image_evidence(images: tuple[bytes, ...]) -> list[dict[str, object]]:
    """Create compact evidence that links each visual state to its captured image."""
    return [
        {
            "player": player + 1,
            "mime_type": "image/png",
            "bytes": len(image),
            "sha256": hashlib.sha256(image).hexdigest(),
        }
        for player, image in enumerate(images)
    ]


def _team_decision_json(decision: TeamDecision, stage_before: str) -> str:
    """Serialize a team decision when a provider does not supply trace JSON."""
    return json.dumps(
        {
            "captain": decision.captain,
            "scene": next(
                (item.scene for item in decision.decisions.values()),
                "unknown",
            ),
            "stage_before": stage_before,
            "stage_after": decision.stage,
            "objective": decision.objective,
            "human_instruction": decision.human_instruction,
            "human_plan": list(decision.human_plan),
            "player_plans": {
                f"P{player + 1}": list(plan) for player, plan in decision.plans.items()
            },
            "decisions": {
                f"P{player + 1}": {
                    "action": item.action,
                    "confidence": item.confidence,
                    "scene": item.scene,
                    "decision_reason": item.note,
                    "terminal": item.terminal,
                }
                for player, item in decision.decisions.items()
            },
            "usage": {
                "input_tokens": decision.input_tokens,
                "output_tokens": decision.output_tokens,
            },
        },
        sort_keys=True,
    )
