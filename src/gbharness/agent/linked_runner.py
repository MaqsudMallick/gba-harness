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

from gbharness.agent.contracts import Decision, DecisionProvider, DecisionRequest, GamePolicy
from gbharness.multiplayer import LinkedActionResult, LinkedSession
from gbharness.observability import NullTelemetry, RunTelemetry
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_png
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
    telemetry: RunTelemetry = field(default_factory=NullTelemetry)
    stop_event: threading.Event = field(default_factory=threading.Event)
    on_turn: Callable[[LinkedTurnRecord], None] | None = None

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
        previous_digests: list[str | None] = [None] * len(self.assignments)
        repeated = [0] * len(self.assignments)
        errors = [0] * len(self.assignments)
        calls = 0
        input_tokens = 0
        output_tokens = 0
        terminal_count = 0
        status: LinkedRunStatus = "turn_limit"
        model_decisions_per_turn = sum(
            assignment.participant_type == "agent" for assignment in self.assignments
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

                views = split_player_frames(self.frames.require(), len(self.assignments))
                pngs = tuple(to_png(view) for view in views)
                for player, png in enumerate(pngs):
                    digest = hashlib.sha256(png).hexdigest()
                    repeated[player] = (
                        repeated[player] + 1 if digest == previous_digests[player] else 0
                    )
                    previous_digests[player] = digest

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
                outcomes = self._decide_all(requests, turn)
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

                with self.telemetry.span(
                    "verify_team_completion",
                    "GUARDRAIL",
                    attributes={
                        "neatlogs.guardrail.input": json.dumps(
                            [decision.terminal for decision in decisions]
                        ),
                        "turn": turn,
                    },
                ) as guardrail_span:
                    unanimous = all(decision.terminal for decision in decisions)
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

                action_keys = tuple(self._safe_action(decision) for decision in decisions)
                actions = {
                    assignment.player: self.policy.resolve(action_keys[index])
                    for index, assignment in enumerate(self.assignments)
                }
                action_summary = {
                    f"P{assignment.player + 1}": action_keys[index]
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
        self, requests: tuple[DecisionRequest, ...], turn: int
    ) -> tuple[Decision | BaseException, ...]:
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

        with ThreadPoolExecutor(
            max_workers=len(self.assignments), thread_name_prefix="linked-agent"
        ) as executor:
            futures = [
                executor.submit(
                    contextvars.copy_context().run,
                    call,
                    assignment,
                    request,
                )
                for assignment, request in zip(self.assignments, requests, strict=True)
            ]
            outcomes: list[Decision | BaseException] = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except BaseException as exc:
                    outcomes.append(exc)
            return tuple(outcomes)

    def _safe_action(self, decision: Decision) -> str:
        if decision.confidence >= self.config.confidence_threshold:
            return decision.action
        return "wait"

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
            f"You are Player {assignment.player + 1} of {len(self.assignments)}.\n"
            f"Role: {assignment.role}\n"
            f"Turn: {turn}\n"
            f"Shared emulated frame: {self.session.frame}\n"
            f"Your recent actions: {own_recent}\n"
            f"Team actions last turn: {last_team_actions}\n"
            f"Consecutive identical private observations: {repeated}\n"
            "Coordinate from visible teammate positions and shared game cues. "
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
        )
