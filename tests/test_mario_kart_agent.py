from __future__ import annotations

import numpy as np

from gbharness.agent.mario_kart import MarioKartVisionAgent, observe_mario_kart
from gbharness.agent.multiplayer import AutonomousPlayer
from gbharness.buttons import Button
from gbharness.vision.capture import RawFrame


def race_frame(center: int = 120) -> np.ndarray:
    frame = np.full((160, 240, 3), (20, 190, 35), dtype=np.uint8)
    frame[:82] = (80, 175, 245)
    for row in range(82, 145):
        half_width = 16 + (row - 82)
        left = max(0, center - half_width)
        right = min(240, center + half_width)
        frame[row, left:right] = (132, 132, 132)
    return frame


def rgb565_frame(rgb: np.ndarray) -> RawFrame:
    pixels = rgb.astype(np.uint16)
    packed = (
        ((pixels[:, :, 0] >> 3) << 11)
        | ((pixels[:, :, 1] >> 2) << 5)
        | (pixels[:, :, 2] >> 3)
    ).astype("<u2")
    height, width, _ = rgb.shape
    return RawFrame(packed.tobytes(), width, height, width * 2, 1)


def test_observation_finds_visible_road_center() -> None:
    observation = observe_mario_kart(race_frame(center=148))

    assert observation.racing
    assert observation.road_center is not None
    assert observation.road_center == 141
    assert observation.road_width >= 50
    assert observation.confidence > 0.5


def test_agent_accelerates_and_steers_toward_road() -> None:
    clockwise = MarioKartVisionAgent(
        smoothed_center=85,
        smoothed_heading=-30,
    ).decide(race_frame(center=92), frame=0)
    counter_clockwise = MarioKartVisionAgent(
        smoothed_center=170,
        smoothed_heading=30,
    ).decide(race_frame(center=148), frame=0)

    assert clockwise.buttons == {Button.A, Button.RIGHT}
    assert counter_clockwise.buttons == {Button.A, Button.LEFT}
    assert clockwise.scene == counter_clockwise.scene == "off-road-recovery"


def test_agent_accelerates_while_waiting_for_race() -> None:
    menu = np.full((160, 240, 3), (10, 180, 240), dtype=np.uint8)

    decision = MarioKartVisionAgent().decide(menu, frame=0)

    assert Button.A in decision.buttons
    assert len(decision.buttons) == 2
    assert decision.scene == "recovering"


def test_autonomous_player_crops_assigned_view_and_caches_decisions() -> None:
    composite = np.concatenate((race_frame(120), race_frame(148)), axis=1)
    controller = AutonomousPlayer(1, 2, MarioKartVisionAgent(decision_interval=3))

    first = controller.update(rgb565_frame(composite), frame=0)
    cached = controller.update(rgb565_frame(composite), frame=1)
    refreshed = controller.update(rgb565_frame(composite), frame=3)

    assert Button.A in first.buttons
    assert "near=" in first.note
    assert cached is first
    assert refreshed.scene == first.scene
    assert controller.decisions == 2
