from __future__ import annotations

import numpy as np

from gbharness.agent.mario_kart import MarioKartVisionAgent, observe_mario_kart
from gbharness.agent.multiplayer import AutonomousPlayer
from gbharness.buttons import Button
from gbharness.vision.capture import RawFrame


def race_frame(center: int = 120, far_center: int | None = None) -> np.ndarray:
    frame = np.full((160, 240, 3), (20, 190, 35), dtype=np.uint8)
    frame[:82] = (80, 175, 245)
    far_center = center if far_center is None else far_center
    for row in range(82, 145):
        half_width = 16 + (row - 82)
        progress = (row - 82) / (145 - 82)
        row_center = round(far_center * (1 - progress) + center * progress)
        left = max(0, row_center - half_width)
        right = min(240, row_center + half_width)
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


def test_agent_uses_three_point_escape_when_stranded() -> None:
    agent = MarioKartVisionAgent(
        stall_threshold=2,
        off_road_escape_threshold=99,
        reverse_escape_frames=2,
        forward_escape_frames=2,
    )
    grass = np.full((160, 240, 3), (20, 190, 35), dtype=np.uint8)

    agent.decide(grass, frame=0)
    agent.decide(grass, frame=1)
    reverse = agent.decide(grass, frame=2)
    reverse_last = agent.decide(grass, frame=3)
    forward = agent.decide(grass, frame=4)

    assert reverse.scene == reverse_last.scene == "unstuck-reverse"
    assert Button.B in reverse.buttons
    assert Button.A not in reverse.buttons
    assert len(reverse.buttons & {Button.LEFT, Button.RIGHT}) == 1
    assert forward.scene == "unstuck-forward"
    assert Button.A in forward.buttons
    assert Button.B not in forward.buttons
    assert (reverse.buttons & {Button.LEFT, Button.RIGHT}) != (
        forward.buttons & {Button.LEFT, Button.RIGHT}
    )


def test_wrong_way_recovery_keeps_accelerating_on_a_bend() -> None:
    frame = race_frame(center=150, far_center=92)
    frame[35:48, 80:175] = (230, 20, 20)
    frame[48:64, 80:175] = (245, 245, 245)

    decision = MarioKartVisionAgent().decide(frame, frame=8)

    assert observe_mario_kart(frame).reverse
    assert decision.scene == "reverse-recovery"
    assert Button.A in decision.buttons
