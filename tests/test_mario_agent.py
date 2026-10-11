"""Local Mario controller vision and state transitions."""

from __future__ import annotations

import numpy as np

from gbharness.agent.mario import MarioVisionController, observe_mario

SKY = (123, 146, 255)
GROUND = (231, 89, 16)
GREEN = (57, 186, 49)
WHITE = (255, 251, 255)


def gameplay_frame() -> np.ndarray:
    frame = np.full((160, 240, 3), SKY, dtype=np.uint8)
    frame[:40] = (0, 0, 0)
    frame[144:] = GROUND
    return frame


def test_observation_finds_near_field_gap_enemy_and_pipe() -> None:
    frame = gameplay_frame()
    frame[144:, 150:160] = SKY
    frame[132:140, 170:176] = GROUND
    frame[110:144, 190:198] = GREEN

    observation = observe_mario(frame)

    assert observation.gameplay
    assert observation.gap_at == 40
    assert observation.danger_at == 40
    assert observation.pipe_at == 80


def test_controller_jumps_for_close_hazard_then_honors_cooldown() -> None:
    frame = gameplay_frame()
    frame[144:, 150:160] = SKY
    controller = MarioVisionController()

    first = controller.decide(frame, frame=500)
    second = controller.decide(frame, frame=520)

    assert first.action == "jump"
    assert "gap" in first.note
    assert second.action == "wait"


def test_opening_hop_establishes_the_route_cadence() -> None:
    frame = gameplay_frame()
    frame[144:, 110:120] = SKY
    controller = MarioVisionController()

    decision = controller.decide(frame, frame=348)

    assert decision.action == "jump"


def test_staircase_switches_to_landing_safe_finish_cadence() -> None:
    frame = gameplay_frame()
    frame[80:130, 20:40] = GROUND
    controller = MarioVisionController()

    approach = controller.decide(frame, frame=1_400)
    decision = controller.decide(frame, frame=1_448)

    assert approach.action != "finish_jump"
    assert decision.action == "finish_jump"
    assert decision.scene == "late-staircase"
    assert controller.finishing


def test_next_level_card_is_terminal_only_after_flag_was_seen() -> None:
    flag = gameplay_frame()
    flag[20:140, 136] = GREEN
    card = np.zeros((160, 240, 3), dtype=np.uint8)
    card[50:70, 60:180] = WHITE
    controller = MarioVisionController()

    at_flag = controller.decide(flag, frame=2_000)
    completed = controller.decide(card, frame=2_500)

    assert at_flag.scene == "flag-sequence"
    assert not at_flag.terminal
    assert completed.terminal
    assert completed.progress == 1.0


def test_title_screen_restarts_and_clears_stale_finish_phase() -> None:
    title = gameplay_frame()
    title[10:80, 30:210] = GROUND
    controller = MarioVisionController(finishing=True)

    decision = controller.decide(title, frame=3_000)

    assert decision.action == "start"
    assert not controller.finishing
