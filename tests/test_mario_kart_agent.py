from __future__ import annotations

import numpy as np

from gbharness.agent.mario_kart import MarioKartVisionAgent, observe_mario_kart
from gbharness.agent.mario_kart_minimap import (
    MarioKartMinimapNavigator,
    TrackProfile,
)
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
        ((pixels[:, :, 0] >> 3) << 11) | ((pixels[:, :, 1] >> 2) << 5) | (pixels[:, :, 2] >> 3)
    ).astype("<u2")
    height, width, _ = rgb.shape
    return RawFrame(packed.tobytes(), width, height, width * 2, 1)


def minimap_frame(marker: tuple[int, int], *, white: bool) -> np.ndarray:
    frame = race_frame()
    x, y = marker
    frame[y - 2 : y + 4, x - 2 : x + 4] = (12, 12, 12)
    frame[y : y + 2, x : x + 2] = (245, 245, 245) if white else (230, 40, 35)
    return frame


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

    assert clockwise.buttons == {Button.A, Button.LEFT}
    assert counter_clockwise.buttons == {Button.A, Button.RIGHT}
    assert clockwise.scene == counter_clockwise.scene == "off-road-recovery"


def test_agent_accelerates_while_waiting_for_race() -> None:
    menu = np.full((160, 240, 3), (10, 180, 240), dtype=np.uint8)

    decision = MarioKartVisionAgent().decide(menu, frame=0)

    assert Button.A in decision.buttons
    assert len(decision.buttons) == 2
    assert decision.scene == "recovering"


def test_recovery_continues_the_last_visible_turn() -> None:
    menu = np.full((160, 240, 3), (10, 180, 240), dtype=np.uint8)

    decision = MarioKartVisionAgent(last_steer=Button.LEFT).decide(menu, frame=0)

    assert decision.buttons == {Button.A, Button.LEFT}


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

    agent = MarioKartVisionAgent()
    decision = agent.decide(frame, frame=8)

    assert observe_mario_kart(frame).reverse
    assert decision.scene == "reverse-recovery"
    assert Button.A in decision.buttons

    # The banner flashes, so recovery must remain active on an intervening frame
    # where the red REVERSE text is not rendered.
    following = agent.decide(race_frame(center=150, far_center=92), frame=9)
    assert following.scene == "reverse-recovery"


def test_agent_realigns_when_crossing_road_sideways() -> None:
    frame = np.full((160, 240, 3), (20, 190, 35), dtype=np.uint8)
    frame[:82] = (80, 175, 245)
    frame[82:154] = (132, 132, 132)
    agent = MarioKartVisionAgent(last_steer=Button.LEFT, needs_realign=True)

    decision = agent.decide(frame, frame=0)

    assert observe_mario_kart(frame).perspective_expansion == 1.0
    assert decision.scene == "road-realignment"
    assert decision.buttons == {Button.A, Button.LEFT}


def test_minimap_identifies_self_from_white_red_blink() -> None:
    profile = TrackProfile(
        "test-course",
        ((50, 40), (50, 20), (30, 20), (30, 40)),
        lookahead=12,
    )
    navigator = MarioKartMinimapNavigator(profile=profile)

    # Coordinates here are in the full 240x160 player view. The first white frame
    # starts the identity handshake; a red fill at the same location confirms self.
    first = navigator.update(minimap_frame((202, 118), white=True), frame=0)
    acquired = navigator.update(minimap_frame((202, 117), white=False), frame=5)
    navigator.update(minimap_frame((202, 114), white=True), frame=10)
    navigator.update(minimap_frame((202, 111), white=False), frame=15)
    navigator.update(minimap_frame((202, 108), white=True), frame=20)
    turning = navigator.update(minimap_frame((202, 105), white=False), frame=25)

    assert first.position is None
    assert acquired.position == (50.5, 39.5)
    assert turning.position == (50.5, 27.5)
    assert turning.velocity == (0.0, -12.0)
    assert turning.steering_error is not None
    assert turning.steering_error < -20
    assert turning.moving_forward


def test_minimap_does_not_mistake_red_opponent_for_self() -> None:
    navigator = MarioKartMinimapNavigator()

    observation = navigator.update(minimap_frame((190, 120), white=False), frame=0)

    assert observation.position is None


def test_minimap_counts_forward_start_line_crossing() -> None:
    profile = TrackProfile("tiny-loop", ((10, 10), (20, 10), (20, 20), (10, 20)))
    navigator = MarioKartMinimapNavigator(profile=profile)

    navigator._last_progress = 0.9
    # A direct projection check is sufficient here; marker extraction and identity are
    # exercised above. The public counter changes only across the profile wrap.
    navigator.position = (10.5, 12.5)
    navigator._history.append((0, (10.5, 12.5)))
    frame = minimap_frame((162, 88), white=True)
    navigator._pending_white.append((0, (10.5, 10.5)))
    navigator.update(frame, frame=1)

    assert navigator.completed_laps == 1
