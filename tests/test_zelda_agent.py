from __future__ import annotations

from typing import cast

import numpy as np

from gbharness.agent.zelda import (
    FourSwordsVisualTargetNavigator,
    _forward_marked_target_grid,
    _forward_portal_search_step,
    _link_position,
    _order_target_grid,
    _target_grids,
    _visible_reached_teammate_step,
    _visible_wall_opening,
    bootstrap_four_swords,
    first_training_reward_visible,
    four_swords_dialogue_visible,
    four_swords_gameplay_visible,
    four_swords_npc_adjacent,
    four_swords_probe_target_reached,
    four_swords_visual_target_plan,
)
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession
from gbharness.vision.capture import FrameLatch, RawFrame
from gbharness.vision.multiplayer import split_player_frames


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


def test_four_swords_dialogue_detector_uses_any_player_text_box() -> None:
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
    # Put the dialogue box only in the P2 viewport.
    rgb[98:142, 250:470] = (8, 8, 8)
    assert four_swords_dialogue_visible(frame())


def test_four_swords_npc_adjacent_uses_p2_visible_pixels() -> None:
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

    views = tuple(split_player_frames(frame(), 4))
    assert not four_swords_npc_adjacent(views)
    # Add Purple Link next to P2. Its pink clothes must not match the NPC.
    rgb[64:70, 385:397] = (190, 40, 190)
    rgb[79:84, 387:395] = (190, 40, 190)
    rgb[70:79, 385:397] = (245, 190, 35)
    rgb[45:55, 354:364] = (222, 40, 165)
    views = tuple(split_player_frames(frame(), 4))
    assert not four_swords_npc_adjacent(views)
    rgb[45:55, 354:364] = (130, 180, 110)
    rgb[68:78, 354:364] = (222, 40, 165)
    views = tuple(split_player_frames(frame(), 4))
    assert four_swords_npc_adjacent(views)


def test_visual_target_plan_moves_red_scout_to_detected_orange_tile() -> None:
    def view(target_y: int, *, activate: bool = False) -> RawFrame:
        current = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
        for x in (72, 120):
            for y in (target_y, target_y + 48):
                current[y - 15 : y + 16, x - 15 : x + 16] = (250, 100, 20)
                current[y - 9 : y + 10, x - 9 : x + 10] = (245, 245, 220)
        current[target_y + 19 : target_y + 29, 91:101] = (222, 40, 165)
        if activate:
            y = target_y + 48
            current[y - 15 : y + 16, 105:136] = (173, 219, 255)
            current[y - 9 : y + 10, 111:130] = (255, 251, 255)
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

    far = view(125)
    assert four_swords_visual_target_plan("probe_target", (far, far, far, far)) == {1: ("move_up",)}

    near = view(40, activate=True)
    assert four_swords_visual_target_plan("probe_target", (near, near, near, near)) == {
        1: ("wait",)
    }
    assert four_swords_probe_target_reached((near, near, near, near))
    assert not four_swords_probe_target_reached((far, far, far, far))
    assert four_swords_visual_target_plan("reach_warps", (far, far, far, far)) == {}

    blank = _composite(gameplay=False)
    blank_views = tuple(split_player_frames(blank, 4))
    assert four_swords_visual_target_plan("probe_target", blank_views) == {1: ("move_up",)}


def test_visual_target_navigator_locks_distinct_targets_and_waits_on_arrival() -> None:
    targets = ((70, 40), (120, 40), (70, 88), (120, 88))

    def view(
        player: int,
        position: tuple[int, int],
        visible_targets: tuple[tuple[int, int], ...] = targets,
        active_targets: tuple[tuple[int, int], ...] = (),
        frame_index: int = 0,
    ) -> RawFrame:
        rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
        for x, y in visible_targets:
            rgb[y - 15 : y + 16, x - 15 : x + 16] = (250, 100, 20)
            rgb[y - 9 : y + 10, x - 9 : x + 10] = (245, 245, 220)
        for x, y in active_targets:
            rgb[y - 15 : y + 16, x - 15 : x + 16] = (173, 219, 255)
            rgb[y - 9 : y + 10, x - 9 : x + 10] = (255, 251, 255)
        all_targets = _merge_test_targets(visible_targets, active_targets)
        if len(all_targets) == 4:
            centre_x = sum(point[0] for point in all_targets) // 4
            centre_y = sum(point[1] for point in all_targets) // 4
            rgb[centre_y - 5 : centre_y + 5, centre_x - 5 : centre_x + 5] = (
                222,
                40,
                165,
            )
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
            index=frame_index,
        )

    navigator = FourSwordsVisualTargetNavigator()
    positions = {1: (120, 80), 2: (120, 80), 3: (120, 80)}
    views = tuple(view(max(index, 1), positions.get(index, (120, 80))) for index in range(4))
    p2_unreached = FourSwordsVisualTargetNavigator()
    p2_target_below = list(views)
    p2_target_below[1] = view(1, (120, 80), ((120, 125),))
    assert p2_unreached("reach_warps", tuple(p2_target_below))[1] == ("move_up",)
    assert 1 not in p2_unreached.reached_players

    views = (
        views[0],
        view(1, (120, 80), targets[:-1], ((120, 88),)),
        views[2],
        views[3],
    )

    plans = navigator("reach_warps", views)

    assert set(plans) == {1, 2, 3}
    assert len(set(navigator.assignments.values())) == 3
    assert plans[1] == ("wait",)

    shifted_p4_target = (160, 120)
    partial_views = list(views)
    partial_views[3] = view(3, (120, 80), (shifted_p4_target,))
    assert navigator("reach_warps", tuple(partial_views))[3] != ("wait",)

    reached_views = list(views)
    reached_views[1] = view(1, (120, 80), targets[:-1], ((120, 88),))
    reached_views[2] = view(
        2,
        (120, 80),
        ((170, 88), (120, 136), (170, 136)),
        ((120, 88),),
    )
    reached_views[3] = view(
        3,
        (120, 80),
        ((120, 40), (170, 40), (170, 88)),
        ((120, 88),),
    )

    navigator("reach_warps", tuple(reached_views))
    reached_views = [
        view(
            player,
            (120, 80),
            (
                targets[:-1]
                if player == 1
                else (
                    ((170, 88), (120, 136), (170, 136))
                    if player == 2
                    else ((120, 40), (170, 40), (170, 88))
                )
            ),
            ((120, 88),),
            frame_index=1,
        )
        if player in (1, 2, 3)
        else reached_views[player]
        for player in range(4)
    ]
    reached_plans = navigator("reach_warps", tuple(reached_views))
    assert reached_plans == {1: ("wait",), 2: ("wait",), 3: ("wait",)}
    assert navigator.all_assigned_players_reached(tuple(reached_views))


def _merge_test_targets(
    visible: tuple[tuple[int, int], ...],
    active: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    result = list(visible)
    for target in active:
        if target not in result:
            result.append(target)
    return tuple(result)


def test_forward_grid_rejects_the_return_portal_marker() -> None:
    rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
    targets = ((70, 40), (120, 40), (70, 88), (120, 88))
    for x, y in targets:
        rgb[y - 15 : y + 16, x - 15 : x + 16] = (250, 100, 20)
        rgb[y - 9 : y + 10, x - 9 : x + 10] = (245, 245, 220)

    # The red U-shaped return marker is not the pink forward marker.
    rgb[58:68, 90:100] = (190, 30, 25)
    assert _forward_marked_target_grid(rgb) is None

    rgb[58:68, 90:100] = (222, 40, 165)
    assert _forward_marked_target_grid(rgb) is not None


def test_portal_search_moves_down_from_a_visible_return_grid() -> None:
    rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
    for x, y in ((70, 40), (120, 40), (70, 88), (120, 88)):
        rgb[y - 15 : y + 16, x - 15 : x + 16] = (250, 100, 20)
        rgb[y - 9 : y + 10, x - 9 : x + 10] = (245, 245, 220)
    rgb[58:68, 90:100] = (190, 30, 25)

    assert _forward_portal_search_step(rgb, (150, 75)) == ("move_down",)


def test_navigator_keeps_leaving_after_return_grid_is_partially_hidden() -> None:
    navigator = FourSwordsVisualTargetNavigator()

    def frame(with_grid: bool, index: int) -> RawFrame:
        rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
        if with_grid:
            for x, y in ((70, 40), (120, 40), (70, 88), (120, 88)):
                rgb[y - 15 : y + 16, x - 15 : x + 16] = (250, 100, 20)
                rgb[y - 9 : y + 10, x - 9 : x + 10] = (245, 245, 220)
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
            index=index,
        )

    return_view = frame(True, 0)
    blank_view = frame(False, 1)
    first = navigator("reach_warps", (return_view,) * 4)
    second = navigator("reach_warps", (blank_view,) * 4)

    assert first == {1: ("move_down",), 2: ("move_down",), 3: ("move_down",)}
    assert second == first


def test_visible_reached_teammate_ignores_a_false_colour_match() -> None:
    rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)

    # Only the red teammate is on an active portal pad. The blue detection is
    # a false match and must not change the route.
    rgb[29:60, 127:158] = (173, 219, 255)
    rgb[35:54, 133:152] = (255, 251, 255)

    def draw_link(player: int, position: tuple[int, int]) -> None:
        colours = {
            1: (220, 30, 25),
            2: (30, 70, 220),
            3: (180, 40, 190),
        }
        x, y = position
        rgb[y - 13 : y + 1, x - 4 : x + 5] = colours[player]
        rgb[y - 21 : y - 13, x - 4 : x + 5] = (245, 190, 35)

    draw_link(1, (145, 44))
    draw_link(2, (114, 121))
    draw_link(3, (119, 86))

    assert _visible_reached_teammate_step(
        rgb,
        3,
        (119, 86),
        {1, 2},
    ) == ("move_up",)


def test_target_grid_order_ignores_small_vertical_jitter() -> None:
    targets = ((160, 91), (110, 92), (160, 140), (110, 141))

    assert _order_target_grid(targets) == (
        (110, 92),
        (160, 91),
        (110, 141),
        (160, 140),
    )


def test_target_grid_rejects_two_rows_from_different_warp_groups() -> None:
    return_row = ((80, 120), (128, 119))
    forward_row = ((80, 30), (128, 31))

    assert _target_grids((*return_row, *forward_row)) == ()


def test_wall_opening_detector_finds_an_off_centre_gap() -> None:
    rgb = np.full((160, 240, 3), (181, 211, 173), dtype=np.uint8)
    rgb[55:76, :145] = (173, 121, 115)
    rgb[55:76, 193:] = (173, 121, 115)

    opening = _visible_wall_opening(rgb, (120, 88))

    assert opening is not None
    assert 165 <= opening[0] <= 173


def test_link_position_does_not_include_a_nearby_target_ring() -> None:
    rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
    rgb[78:84, 116:124] = (220, 30, 25)
    rgb[70:78, 116:124] = (245, 190, 35)
    rgb[91:119, 106:110] = (220, 30, 25)
    rgb[91:119, 130:134] = (220, 30, 25)
    rgb[91:95, 106:134] = (220, 30, 25)
    rgb[115:119, 106:134] = (220, 30, 25)

    assert _link_position(rgb, 1) == (119, 87)


def test_visual_target_navigator_keeps_p2_on_a_covered_probe_target() -> None:
    navigator = FourSwordsVisualTargetNavigator()

    def views(
        targets: tuple[tuple[int, int], ...],
        active_targets: tuple[tuple[int, int], ...] = (),
    ) -> tuple[RawFrame, ...]:
        rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
        for x, y in targets:
            rgb[y - 15 : y + 16, x - 15 : x + 16] = (250, 100, 20)
            rgb[y - 9 : y + 10, x - 9 : x + 10] = (245, 245, 220)
        for x, y in active_targets:
            rgb[y - 15 : y + 16, x - 15 : x + 16] = (173, 219, 255)
            rgb[y - 9 : y + 10, x - 9 : x + 10] = (255, 251, 255)
        all_targets = _merge_test_targets(targets, active_targets)
        if len(all_targets) == 4:
            centre_x = sum(point[0] for point in all_targets) // 4
            centre_y = sum(point[1] for point in all_targets) // 4
            rgb[centre_y - 5 : centre_y + 5, centre_x - 5 : centre_x + 5] = (
                222,
                40,
                165,
            )
        packed = (
            ((rgb[:, :, 0].astype(np.uint16) >> 3) << 11)
            | ((rgb[:, :, 1].astype(np.uint16) >> 2) << 5)
            | (rgb[:, :, 2].astype(np.uint16) >> 3)
        )
        frame = RawFrame(
            data=packed.astype("<u2").tobytes(),
            width=240,
            height=160,
            pitch=480,
            index=0,
        )
        return (frame, frame, frame, frame)

    assert navigator("probe_target", views(((140, 88),))) == {1: ("move_up",)}
    inactive = ((72, 40), (120, 40), (72, 88), (120, 88))
    assert navigator("probe_target", views(inactive)) == {1: ("micro_down",)}
    active = ((72, 40), (120, 40), (72, 88))
    assert navigator.probe_target_reached(views(active, ((120, 88),)))
    assert navigator("probe_target", views(active, ((120, 88),))) == {1: ("wait",)}


def test_visual_target_navigator_faces_npc_and_recovers_from_failed_dialogue() -> None:
    navigator = FourSwordsVisualTargetNavigator()
    rgb = np.full((160, 240, 3), (130, 180, 110), dtype=np.uint8)
    rgb[65:75, 115:125] = (222, 40, 165)

    def frame(current: np.ndarray) -> RawFrame:
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

    views = (frame(rgb),) * 4
    assert navigator("reach_npc", views) == {1: ("wait",)}
    assert navigator("advance_dialogue", views)[1][0] == "face_up"
    navigator("advance_dialogue", views)
    assert navigator("advance_dialogue", views)[1][:3] == (
        "move_down",
        "face_up",
        "interact",
    )

    dialogue = rgb.copy()
    dialogue[92:148, 4:236] = (8, 8, 8)
    assert navigator("advance_dialogue", (frame(dialogue),) * 4)[1] == (
        "interact",
        "wait",
        "wait",
        "wait",
        "wait",
        "wait",
    )


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
