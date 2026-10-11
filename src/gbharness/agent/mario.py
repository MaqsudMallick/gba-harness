"""Deterministic visual controller for the bundled Super Mario Bros. demo route.

The cloud model remains useful for unfamiliar screens and games, but a platformer needs
far more control decisions per second than a remote model quota can sustain.  This module
turns the model's high-level objective (finish World 1-1) into a local closed-loop skill:
it reads the same framebuffer as the model, identifies collision-relevant pixels, and
uses bounded input macros.  It never reads emulator RAM or restores a save state.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from gbharness.actions import Action, press, release, tap, wait
from gbharness.agent.runner import EpisodeResult, RunnerConfig, TurnRecord
from gbharness.observability import NullTelemetry, RunTelemetry
from gbharness.session import InputSession
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_rgb

SKY = np.array((123, 146, 255), dtype=np.uint8)
BLACK = np.array((0, 0, 0), dtype=np.uint8)
WHITE = np.array((255, 251, 255), dtype=np.uint8)

LOCAL_ACTIONS: dict[str, Action] = {
    "wait": wait(4),
    "start": tap("start", hold_frames=4, gap_frames=10),
    "jump": tap("a", hold_frames=34, gap_frames=2),
    # A slightly longer release window is what lets Mario re-arm Jump after landing on
    # the late staircases instead of consuming the next edge while still airborne.
    "finish_jump": tap("a", hold_frames=34, gap_frames=18),
}


@dataclass(frozen=True, slots=True)
class MarioObservation:
    """Collision cues extracted from one native 240x160 framebuffer."""

    gameplay: bool
    gap_at: int | None
    danger_at: int | None
    pipe_at: int | None
    staircase: bool
    flagpole: bool
    next_level_card: bool
    title: bool


def observe_mario(rgb: np.ndarray, *, goal_seen: bool = False) -> MarioObservation:
    """Extract the small set of visual cues needed to complete World 1-1.

    Distances are measured from fixed near-field scan lines.  Once the screen scrolls,
    Mario stays near x=104, so the probes at x=110 and x=130 remain player-relative.
    Exact matching is used only for the stable NES palette's sky/text colors; collision
    objects use broad red/green ranges so sprite palette animation does not hide them.
    """
    if rgb.shape != (160, 240, 3):
        raise ValueError(f"expected a 160x240 RGB frame, got {rgb.shape!r}")

    sky = np.all(rgb == SKY, axis=2)
    black = np.all(rgb == BLACK, axis=2)
    white = np.all(rgb == WHITE, axis=2)
    gameplay = float(sky[40:145].mean()) > 0.35

    scan_start, scan_end = 110, 230
    gap_columns = sky[144:160, scan_start:scan_end].mean(axis=0) > 0.75
    gap_at = int(np.argmax(gap_columns)) if np.any(gap_columns) else None

    danger = rgb[120:144, 130:scan_end]
    red = (danger[:, :, 0] > 200) & (danger[:, :, 1] < 130)
    danger_columns = red.sum(axis=0) >= 2
    danger_at = int(np.argmax(danger_columns)) if np.any(danger_columns) else None

    pipe_region = rgb[96:144, scan_start:scan_end]
    green = (pipe_region[:, :, 1] > 170) & (pipe_region[:, :, 0] < 190)
    pipe_columns = green.sum(axis=0) >= 12
    pipe_at = int(np.argmax(pipe_columns)) if np.any(pipe_columns) else None

    lower = rgb[70:144]
    lower_red = (lower[:, :, 0] > 200) & (lower[:, :, 1] < 130)
    staircase = int(lower_red.sum()) > 900

    whole_green = (rgb[20:145, :, 1] > 170) & (rgb[20:145, :, 0] < 190)
    flagpole = int(whole_green.sum(axis=0).max(initial=0)) >= 80
    upper = rgb[10:90]
    upper_red = (upper[:, :, 0] > 200) & (upper[:, :, 1] < 130)
    title = int(upper_red.sum()) > 2_000
    next_level_card = (
        goal_seen
        and float(black.mean()) > 0.85
        and int(white[40:110].sum()) > 100
    )
    return MarioObservation(
        gameplay=gameplay,
        gap_at=gap_at,
        danger_at=danger_at,
        pipe_at=pipe_at,
        staircase=staircase,
        flagpole=flagpole,
        next_level_card=next_level_card,
        title=title,
    )


@dataclass(frozen=True, slots=True)
class MarioControlDecision:
    action: Literal["wait", "start", "jump", "finish_jump"]
    scene: str
    terminal: bool = False
    progress: float | None = None
    note: str = ""


@dataclass(slots=True)
class MarioVisionController:
    """Stateful, closed-loop World 1-1 policy."""

    # The opening hop is part of the deterministic route: besides clearing the first
    # enemy, it establishes the landing phase used by the later obstacle cadence.
    next_jump_frame: int = 0
    next_start_frame: int = 0
    staircase_seen_frame: int | None = None
    finishing: bool = False
    goal_seen: bool = False

    def decide(self, rgb: np.ndarray, frame: int) -> MarioControlDecision:
        observation = observe_mario(rgb, goal_seen=self.goal_seen)
        self.goal_seen = self.goal_seen or observation.flagpole
        if observation.next_level_card:
            return MarioControlDecision(
                "wait",
                "world-1-2-card",
                terminal=True,
                progress=1.0,
                note="World 1-2 card appeared after the flag and castle sequence.",
            )

        if observation.title:
            self.finishing = False
            self.goal_seen = False
            self.next_jump_frame = frame + 80
            self.staircase_seen_frame = None
            if frame < self.next_start_frame:
                return MarioControlDecision(
                    "wait",
                    "title",
                    progress=0.0,
                    note="Wait for the previous Start press to take effect.",
                )
            self.next_start_frame = frame + 180
            return MarioControlDecision(
                "start",
                "title",
                progress=0.0,
                note="Restart after a game over.",
            )

        # A death transition before the flag invalidates the one-way late-level phase.
        # Clear it so the next life uses near-field obstacle avoidance from the start.
        if self.finishing and not self.goal_seen and not observation.gameplay:
            self.finishing = False
            self.next_jump_frame = frame + 80
            self.staircase_seen_frame = None

        if observation.staircase and self.staircase_seen_frame is None:
            self.staircase_seen_frame = frame
        staircase_ready = (
            self.staircase_seen_frame is not None
            and frame - self.staircase_seen_frame >= 48
        )
        self.finishing = self.finishing or staircase_ready or self.goal_seen
        if self.finishing:
            return MarioControlDecision(
                "finish_jump",
                "flag-sequence" if self.goal_seen else "late-staircase",
                progress=0.95 if self.goal_seen else 0.80,
                note="Use a landing-safe jump cadence through the staircase and flag route.",
            )

        gap_close = observation.gap_at is not None and observation.gap_at < 82
        danger_close = observation.danger_at is not None and observation.danger_at < 60
        pipe_close = observation.pipe_at is not None and observation.pipe_at < 60
        if (
            observation.gameplay
            and frame >= self.next_jump_frame
            and (gap_close or danger_close or pipe_close)
        ):
            self.next_jump_frame = frame + 40
            reasons = [
                name
                for name, present in (
                    ("gap", gap_close),
                    ("enemy-or-stair", danger_close),
                    ("pipe", pipe_close),
                )
                if present
            ]
            return MarioControlDecision(
                "jump",
                "level-1-1",
                progress=min(0.75, frame / 1_900),
                note=f"Jump for {', '.join(reasons)}.",
            )
        return MarioControlDecision(
            "wait",
            "level-1-1" if observation.gameplay else "transition",
            progress=min(0.75, frame / 1_900),
            note="Keep sprinting right and scan the near field.",
        )


@dataclass(slots=True)
class MarioAutopilot:
    """Run the visual World 1-1 skill through the standard episode interface."""

    session: InputSession
    frames: FrameLatch
    config: RunnerConfig = field(default_factory=RunnerConfig)
    telemetry: RunTelemetry = field(default_factory=NullTelemetry)
    stop_event: threading.Event = field(default_factory=threading.Event)
    on_turn: Callable[[TurnRecord], None] | None = None
    controller: MarioVisionController = field(default_factory=MarioVisionController)

    def run(self) -> EpisodeResult:
        records: list[TurnRecord] = []
        status: Literal["completed", "turn_limit", "stopped"] = "turn_limit"
        with self.telemetry.span(
            "game_agent_episode",
            "WORKFLOW",
            input_value={
                "game": "super-mario-bros-world-1-1",
                "goal": "clear World 1-1 and reach the World 1-2 card",
                "provider": "local-vision",
                "turn_limit": self.config.max_turns,
            },
        ) as episode_span:
            self.telemetry.log(
                "episode started for {game} with {provider}",
                game="super-mario-bros-world-1-1",
                provider="local-vision",
            )
            with self.telemetry.span(
                "boot_and_start_level",
                "TOOL",
                input_value={"sequence": "wait 180, tap start, wait 150, hold right+b"},
                output_value={"started": "level running with right+b held"},
                attributes={"neatlogs.tool.name": "gba_controller"},
            ):
                self.session.submit(wait(180))
                self.session.submit(tap("start", hold_frames=4, gap_frames=10))
                self.session.submit(wait(150))
                self.session.submit(press(["right", "b"]))
            try:
                for turn in range(1, self.config.max_turns + 1):
                    if self.stop_event.is_set():
                        status = "stopped"
                        break
                    decision = self.controller.decide(
                        to_rgb(self.frames.require()), self.session.frame
                    )
                    if decision.terminal:
                        status = "completed"
                        self.telemetry.log(
                            "episode completed after {turn} turns at frame {frame}",
                            turn=turn,
                            frame=self.session.frame,
                        )
                        break
                    with self.telemetry.span(
                        "execute_controller_action",
                        "TOOL",
                        input_value={"action": decision.action, "scene": decision.scene},
                        attributes={"neatlogs.tool.name": "gba_controller"},
                    ) as controller_span:
                        result = self.session.submit(LOCAL_ACTIONS[decision.action])
                        self.telemetry.record_output(
                            controller_span,
                            "TOOL",
                            {
                                "frames_advanced": result.frames_advanced,
                                "frame": result.total_frames,
                            },
                        )
                    record = TurnRecord(
                        turn=turn,
                        frame=result.total_frames,
                        action=decision.action,
                        confidence=1.0,
                        scene=decision.scene,
                        terminal=False,
                        progress=decision.progress,
                        note=decision.note,
                        frames_advanced=result.frames_advanced,
                        repeated_observations=0,
                    )
                    records.append(record)
                    self.telemetry.log(
                        "turn {turn}: {action} in {scene}",
                        turn=turn,
                        action=decision.action,
                        scene=decision.scene,
                    )
                    if self.on_turn is not None:
                        self.on_turn(record)
            finally:
                self.session.submit(release(["right", "b"]))
                self.telemetry.record_output(
                    episode_span,
                    "WORKFLOW",
                    {
                        "status": status,
                        "turns": len(records),
                        "frames": self.session.frame,
                        "last_scene": records[-1].scene if records else "",
                    },
                )

        return EpisodeResult(
            status=status,
            turns=tuple(records),
            total_frames=self.session.frame,
            provider="local-vision",
        )
