from __future__ import annotations

from typing import cast

import numpy as np

from gbharness.agent.zelda import (
    FourSwordsVisualTargetNavigator,
    bootstrap_four_swords,
    first_training_reward_visible,
    four_swords_dialogue_visible,
    four_swords_gameplay_visible,
    four_swords_probe_target_reached,
    four_swords_visual_target_plan,
)
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession
from gbharness.vision.capture import FrameLatch, RawFrame


class FourPlayerRecordingPort:
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
        return True


class StaticFrames:
    def __init__(self, frame: RawFrame) -> None:
        self.frame = frame

    def require(self) -> RawFrame:
        return self.frame


def _composite(*, gameplay: bool, stage_select: bool = False) -> RawFrame:
    rgb = np.zeros((320, 480, 3), dtype=np.uint8)
    for x, y in ((0, 0), (240, 0), (0, 160), (240, 160)):
        if gameplay:
            rgb[y + 3 : y + 17, x + 120 : x + 130] = (255, 0, 0)
        if stage_select:
            rgb[y + 3 : y + 17, x + 3 : x + 20] = (255, 0, 0)
    packed = (
        ((rgb[:, :, 0].astype(np.uint16) >> 3) << 11)
        | ((rgb[:, :, 1].astype(np.uint16) >> 2) << 5)
        | (rgb[:, :, 2].astype(np.uint16) >> 3)
    )
    return RawFrame(
        data=packed.astype("<u2").tobytes(),
        width=480,
        height=320,
        pitch=960,
        index=0,
    )


def test_four_swords_gameplay_requires_heart_hud_on_every_view() -> None:
    assert not four_swords_gameplay_visible(_composite(gameplay=False), 4)
    assert not four_swords_gameplay_visible(_composite(gameplay=False, stage_select=True), 4)
    assert four_swords_gameplay_visible(_composite(gameplay=True), 4)


def test_four_swords_dialogue_detector_uses_player_one_text_box() -> None:
    rgb = np.full((320, 480, 3), (130, 180, 110), dtype=np.uint8)

    def frame() -> RawFrame:
        packed = (
            ((rgb[:, :, 0].astype(np.uint16) >> 3) << 11)
            | ((rgb[:, :, 1].astype(np.uint16) >> 2) << 5)
            | (rgb[:, :, 2].astype(np.uint16) >> 3)
        )
        return RawFrame(
            data=packed.astype("<u2").tobytes(),
            width=480,
            height=320,
            pitch=960,
            index=0,
        )

    assert not four_swords_dialogue_visible(frame())
    rgb[98:142, 10:230] = (8, 8, 8)
    assert four_swords_dialogue_visible(frame())


def test_visual_target_plan_moves_red_scout_to_detected_orange_tile() -> None:
    rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
    rgb[105:136, 105:136] = (250, 100, 20)
    rgb[111:130, 111:130] = (245, 245, 220)

    def view(player_top: int) -> RawFrame:
        current = rgb.copy()
        current[player_top : player_top + 14, 116:125] = (220, 30, 25)
        packed = (
            ((current[:, :, 0].astype(np.uint16) >> 3) << 11)
            | ((current[:, :, 1].astype(np.uint16) >> 2) << 5)
            | (current[:, :, 2].astype(np.uint16) >> 3)
        )
        return RawFrame(
            data=packed.astype("<u2").tobytes(),
            width=240,
            height=160,
            pitch=480,
            index=0,
        )

    far = view(64)
    assert four_swords_visual_target_plan(
        "probe_target", (far, far, far, far)
    ) == {1: ("move_down", "move_down")}

    near = view(102)
    assert four_swords_visual_target_plan(
        "probe_target", (near, near, near, near)
    ) == {1: ("wait",)}
    assert four_swords_probe_target_reached((near, near, near, near))
    assert not four_swords_probe_target_reached((far, far, far, far))
    assert four_swords_visual_target_plan("reach_warps", (far, far, far, far)) == {}


def test_visual_target_navigator_locks_distinct_targets_and_waits_on_arrival() -> None:
    targets = ((55, 50), (175, 50), (55, 120), (175, 120))

    def view(
        player: int,
        position: tuple[int, int],
        visible_targets: tuple[tuple[int, int], ...] = targets,
    ) -> RawFrame:
        rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
        for x, y in visible_targets:
            rgb[y - 15 : y + 16, x - 15 : x + 16] = (250, 100, 20)
            rgb[y - 9 : y + 10, x - 9 : x + 10] = (245, 245, 220)
        colours = {
            1: (220, 30, 25),
            2: (30, 70, 220),
            3: (180, 40, 190),
        }
        x, y = position
        rgb[y - 13 : y + 1, x - 4 : x + 5] = colours[player]
        packed = (
            ((rgb[:, :, 0].astype(np.uint16) >> 3) << 11)
            | ((rgb[:, :, 1].astype(np.uint16) >> 2) << 5)
            | (rgb[:, :, 2].astype(np.uint16) >> 3)
        )
        return RawFrame(
            data=packed.astype("<u2").tobytes(),
            width=240,
            height=160,
            pitch=480,
            index=0,
        )

    navigator = FourSwordsVisualTargetNavigator()
    positions = {1: (60, 75), 2: (120, 75), 3: (180, 75)}
    views = tuple(
        view(max(index, 1), positions.get(index, (120, 80))) for index in range(4)
    )

    plans = navigator("reach_warps", views)

    assert set(plans) == {1, 2, 3}
    assert len(set(navigator.assignments.values())) == 3

    p4_target = navigator.tracked_targets[3]
    shifted_p4_target = (p4_target[0] - 24, p4_target[1] - 18)
    partial_views = list(views)
    partial_views[3] = view(3, (120, 80), (shifted_p4_target,))
    assert navigator("reach_warps", tuple(partial_views))[3] != ("wait",)

    reached_views = list(views)
    for player, target_index in navigator.assignments.items():
        reached_views[player] = view(player, targets[target_index])

    reached_plans = navigator("reach_warps", tuple(reached_views))
    assert reached_plans == {1: ("wait",), 2: ("wait",), 3: ("wait",)}
    assert navigator.all_assigned_players_reached(tuple(reached_views))


def test_bootstrap_is_deterministic_and_stops_at_visible_gameplay() -> None:
    port = FourPlayerRecordingPort()
    session = LinkedSession(port, max_step_frames=1800)
    frames = cast(FrameLatch, StaticFrames(_composite(gameplay=True)))

    result = bootstrap_four_swords(session, frames)

    assert result.frames_advanced == 3042
    assert result.handshake_confirms == 1
    assert session.frame == 3042
    assert all(not buttons for buttons in port.held)


def test_first_training_reward_needs_both_visible_chests() -> None:
    rgb = np.zeros((320, 480, 3), dtype=np.uint8)

    def frame() -> RawFrame:
        packed = (
            ((rgb[:, :, 0].astype(np.uint16) >> 3) << 11)
            | ((rgb[:, :, 1].astype(np.uint16) >> 2) << 5)
            | (rgb[:, :, 2].astype(np.uint16) >> 3)
        )
        return RawFrame(
            data=packed.astype("<u2").tobytes(),
            width=480,
            height=320,
            pitch=960,
            index=0,
        )

    assert not first_training_reward_visible(frame())
    rgb[80:90, 30:40] = (255, 230, 0)
    assert not first_training_reward_visible(frame())
    rgb[80:90, 190:200] = (255, 230, 0)
    assert first_training_reward_visible(frame())
