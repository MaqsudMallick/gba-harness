"""Cost-bounded orchestration for multiple visual agents in one linked game."""

from __future__ import annotations

import contextvars
import json
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from gbharness.agent.contracts import Decision, DecisionProvider, DecisionRequest, GamePolicy
from gbharness.agent.human import HumanDecisionProvider
from gbharness.agent.team import TeamDecision, TeamDecisionProvider, TeamDecisionRequest
from gbharness.multiplayer import LinkedActionResult, LinkedSession
from gbharness.observability import NullTelemetry, RunTelemetry
from gbharness.vision.capture import FrameLatch
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


@dataclass(frozen=True, slots=True)
class LinkedTurnRecord:
    turn: int
    frame: int
    frames_advanced: int
    players: tuple[PlayerTurnRecord, ...]
    team_instruction: str = ""
    captain: str = ""


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
    _team_instruction: str = field(default="", init=False, repr=False)
    _team_captain: str = field(default="", init=False, repr=False)
    _team_input_tokens: int = field(default=0, init=False, repr=False)
    _team_output_tokens: int = field(default=0, init=False, repr=False)

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
            1
            if self.team_provider is not None
            else sum(assignment.participant_type == "agent" for assignment in self.assignments)
        )

        with self.telemetry.span(
            "coordinate_four_player_decisions",
            "CHAIN",
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
                views = split_player_frames(composite, len(self.assignments))
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
                calls += model_decisions_per_turn

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
                    attributes={
                        "neatlogs.guardrail.input": json.dumps(
                            [
                                decision.terminal
                                for assignment, decision in zip(
                                    self.assignments, decisions, strict=True
                                )
                                if assignment.participant_type == "agent"
                            ]
                        ),
                        "turn": turn,
                    },
                ) as guardrail_span:
                    terminal_votes = [
                        decision.terminal
                        for assignment, decision in zip(self.assignments, decisions, strict=True)
                        if assignment.participant_type == "agent"
                    ]
                    unanimous = bool(terminal_votes) and all(terminal_votes)
                    terminal_count = terminal_count + 1 if unanimous else 0
                    passed = terminal_count >= self.config.terminal_confirmations
                    self.telemetry.annotate(
                        guardrail_span,
                        {
                            "neatlogs.guardrail.passed": passed,
                            "neatlogs.guardrail.output": (
                                f"{terminal_count}/{self.config.terminal_confirmations} "
                                "consecutive unanimous confirmations"
                            ),
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
                actions = {
                    assignment.player: self.policy.resolve(
                        "wait" if assignment.participant_type == "human" else action_keys[index]
                    )
                    for index, assignment in enumerate(self.assignments)
                }
                action_summary = {
                    f"P{assignment.player + 1}": (
                        f"live:{action_keys[index]}"
                        if assignment.participant_type == "human"
                        else action_keys[index]
                    )
                    for index, assignment in enumerate(self.assignments)
                }
                with self.telemetry.span(
                    "execute_synchronized_controller_actions",
                    "TOOL",
                    attributes={
                        "neatlogs.tool.name": "linked_controller",
                        "neatlogs.tool.description": (
                            "Execute all four bounded controller macros on one emulator clock"
                        ),
                        "neatlogs.tool.parameters": json.dumps(action_summary, sort_keys=True),
                        "turn": turn,
                        "frame.before": self.session.frame,
                    },
                ) as controller_span:
                    result = self.session.submit(actions)
                    self.telemetry.annotate(
                        controller_span,
                        {
                            "frame.after": result.total_frames,
                            "frames.advanced": result.frames_advanced,
                            "emulation.ended": result.emulation_ended,
                        },
                    )
                record = self._record(turn, action_keys, decisions, repeated, result)
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
                attributes.update(
                    {
                        "neatlogs.tool.name": "human_keyboard_controller",
                        "neatlogs.tool.description": "Capture one bounded player action",
                        "neatlogs.tool.parameters": "keyboard action queue",
                    }
                )
            with self.telemetry.span(
                span_name,
                span_kind,
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
                        attributes={
                            "neatlogs.tool.name": "human_keyboard_controller",
                            "neatlogs.tool.description": (
                                "Execute a human action while teammate agents are reasoning"
                            ),
                            "neatlogs.tool.parameters": action,
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
                attributes={
                    "neatlogs.tool.name": "human_keyboard_controller",
                    "neatlogs.tool.description": "Summarize live human control during reasoning",
                    "neatlogs.tool.parameters": json.dumps(actions),
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
        )

        def plan() -> TeamDecision:
            with self.telemetry.span(
                "rotating_team_captain_plan",
                "AGENT",
                attributes={
                    "team.provider": team_provider.name,
                    "turn": turn,
                    "ai.players": ",".join(f"P{player + 1}" for player in ai_players),
                    "observation.bytes": len(composite_png),
                },
            ) as plan_span:
                decision = team_provider.decide_team(team_request)
                self.telemetry.annotate(
                    plan_span,
                    {
                        "team.captain": decision.captain,
                        "team.objective": decision.objective,
                        "team.human_instruction": decision.human_instruction,
                        "team.actions": json.dumps(
                            {
                                f"P{player + 1}": item.action
                                for player, item in decision.decisions.items()
                            },
                            sort_keys=True,
                        ),
                        "usage.input_tokens": decision.input_tokens,
                        "usage.output_tokens": decision.output_tokens,
                    },
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
                        attributes={
                            "neatlogs.tool.name": "human_keyboard_controller",
                            "neatlogs.tool.parameters": action,
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
            with self.telemetry.span(
                f"player_{assignment.player + 1}_assigned_team_action",
                "AGENT",
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
            self.telemetry.log(
                "Captain {captain} objective={objective}; P1 instruction={instruction}",
                captain=team.captain,
                objective=team.objective,
                instruction=team.human_instruction,
            )
        return tuple(outcomes)

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
        if repeated == 1:
            recovery = f"attack_{direction}"
        elif repeated == 2:
            recovery = "interact"
        else:
            alternatives = {
                "up": ("move_left", "move_right"),
                "down": ("move_right", "move_left"),
                "left": ("move_down", "move_up"),
                "right": ("move_up", "move_down"),
            }
            recovery = alternatives[direction][player % 2]
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
            f"Consecutive identical private observations: {repeated}\n"
            "A non-zero identical-observation count means your last action did not move the "
            "scene meaningfully. Do not repeat a blocked movement: try the matching attack, "
            "interact with the obstacle, or route perpendicular to it.\n"
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
