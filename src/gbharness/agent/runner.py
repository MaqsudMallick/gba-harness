"""Long-horizon observe-decide-act loop independent of game and model provider."""

from __future__ import annotations

import hashlib
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from gbharness.agent.contracts import Decision, DecisionProvider, DecisionRequest, GamePolicy
from gbharness.observability import NullTelemetry, RunTelemetry
from gbharness.session import ActionResult, InputSession
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_png

RunStatus = Literal["completed", "turn_limit", "stopped"]


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    max_turns: int = 500
    confidence_threshold: float = 0.70
    terminal_confirmations: int = 2
    max_consecutive_errors: int = 3

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.terminal_confirmations < 1:
            raise ValueError("terminal_confirmations must be positive")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be positive")


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn: int
    frame: int
    action: str
    confidence: float
    scene: str
    terminal: bool
    progress: float | None
    note: str
    frames_advanced: int
    repeated_observations: int


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    status: RunStatus
    turns: tuple[TurnRecord, ...]
    total_frames: int
    provider: str


@dataclass(slots=True)
class AgentRunner:
    session: InputSession
    frames: FrameLatch
    provider: DecisionProvider
    policy: GamePolicy
    config: RunnerConfig = field(default_factory=RunnerConfig)
    telemetry: RunTelemetry = field(default_factory=NullTelemetry)
    stop_event: threading.Event = field(default_factory=threading.Event)
    on_turn: Callable[[TurnRecord], None] | None = None

    def run(self) -> EpisodeResult:
        records: list[TurnRecord] = []
        recent: deque[str] = deque(maxlen=6)
        previous_digest: str | None = None
        repeated = 0
        terminal_count = 0
        errors = 0
        status: RunStatus = "turn_limit"

        with self.telemetry.span(
            "game_agent_episode",
            "WORKFLOW",
            input_value={
                "game": self.policy.name,
                "goal": self.policy.goal,
                "provider": self.provider.name,
                "turn_limit": self.config.max_turns,
            },
        ) as episode_span:
            self.telemetry.log(
                "episode started for {game} with {provider}",
                game=self.policy.name,
                provider=self.provider.name,
            )
            for turn in range(1, self.config.max_turns + 1):
                if self.stop_event.is_set():
                    status = "stopped"
                    break

                png = to_png(self.frames.require())
                digest = hashlib.sha256(png).hexdigest()
                repeated = repeated + 1 if digest == previous_digest else 0
                previous_digest = digest
                state = self._state_summary(turn, repeated, recent)
                request = DecisionRequest(
                    goal=self.policy.goal,
                    state=state,
                    image_png=png,
                    actions=self.policy.options,
                )

                try:
                    with self.telemetry.span(
                        "choose_action",
                        "AGENT",
                        input_value={
                            "goal": request.goal,
                            "state": request.state,
                            "allowed_actions": [option.key for option in request.actions],
                        },
                    ) as decision_span:
                        decision = self.provider.decide(request)
                        self.telemetry.record_output(
                            decision_span,
                            "AGENT",
                            {
                                "action": decision.action,
                                "confidence": decision.confidence,
                                "scene": decision.scene,
                                "note": decision.note,
                                "terminal": decision.terminal,
                            },
                        )
                    errors = 0
                except Exception as exc:
                    errors += 1
                    self.telemetry.log(
                        "decision failed on turn {turn}: {error}",
                        level="error",
                        turn=turn,
                        error=type(exc).__name__,
                    )
                    if errors >= self.config.max_consecutive_errors:
                        raise
                    self._recover()
                    continue

                if decision.terminal:
                    terminal_count += 1
                    if terminal_count >= self.config.terminal_confirmations:
                        status = "completed"
                        self.telemetry.log(
                            "episode completed after {turn} turns", turn=turn
                        )
                        break
                else:
                    terminal_count = 0

                action_key = self._safe_action(decision)
                with self.telemetry.span(
                    "execute_controller_action",
                    "TOOL",
                    input_value={"action": action_key},
                    attributes={"neatlogs.tool.name": "gba_controller"},
                ) as controller_span:
                    result = self.session.submit(self.policy.resolve(action_key))
                    self.telemetry.record_output(
                        controller_span,
                        "TOOL",
                        {
                            "frames_advanced": result.frames_advanced,
                            "frame": result.total_frames,
                        },
                    )
                recent.append(action_key)
                record = self._record(turn, action_key, decision, result, repeated)
                records.append(record)
                self.telemetry.log(
                    "turn {turn}: {action} at {confidence:.2f}",
                    turn=turn,
                    action=action_key,
                    confidence=decision.confidence,
                )
                if self.on_turn is not None:
                    self.on_turn(record)
            self.telemetry.record_output(
                episode_span,
                "WORKFLOW",
                {
                    "status": status,
                    "turns": len(records),
                    "frames": self.session.frame,
                    "last_action": records[-1].action if records else "",
                },
            )
        return EpisodeResult(
            status=status,
            turns=tuple(records),
            total_frames=self.session.frame,
            provider=self.provider.name,
        )

    def _safe_action(self, decision: Decision) -> str:
        if decision.confidence >= self.config.confidence_threshold:
            return decision.action
        if "wait" in self.policy.actions:
            self.telemetry.log(
                "low confidence {confidence:.2f}; using wait",
                level="warning",
                confidence=decision.confidence,
            )
            return "wait"
        raise RuntimeError(
            f"decision confidence {decision.confidence:.2f} is below "
            f"{self.config.confidence_threshold:.2f}, and this policy has no wait action"
        )

    def _recover(self) -> None:
        if "wait" not in self.policy.actions:
            return
        with self.telemetry.span(
            "decision_error_recovery",
            "TOOL",
            input_value={"action": "wait", "reason": "decision provider raised"},
            output_value={"recovered": "submitted one wait action"},
            attributes={"neatlogs.tool.name": "gba_controller"},
        ):
            self.session.submit(self.policy.resolve("wait"))

    def _state_summary(self, turn: int, repeated: int, recent: deque[str]) -> str:
        actions = ", ".join(recent) if recent else "none"
        return (
            f"Turn: {turn}\n"
            f"Emulated frames elapsed: {self.session.frame}\n"
            f"Recent actions: {actions}\n"
            f"Consecutive identical observations: {repeated}"
        )

    @staticmethod
    def _record(
        turn: int,
        action: str,
        decision: Decision,
        result: ActionResult,
        repeated: int,
    ) -> TurnRecord:
        return TurnRecord(
            turn=turn,
            frame=result.total_frames,
            action=action,
            confidence=decision.confidence,
            scene=decision.scene,
            terminal=decision.terminal,
            progress=decision.progress,
            note=decision.note,
            frames_advanced=result.frames_advanced,
            repeated_observations=repeated,
        )
