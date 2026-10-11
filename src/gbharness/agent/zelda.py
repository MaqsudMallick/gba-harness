"""Bounded controller vocabulary for Zelda: Four Swords cooperative play."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Final

import numpy as np

from gbharness.actions import Action, hold, tap, wait
from gbharness.agent.contracts import GamePolicy
from gbharness.multiplayer import LinkedSession
from gbharness.observability import NullTelemetry, RunTelemetry
from gbharness.vision.capture import FrameLatch, RawFrame
from gbharness.vision.encode import to_rgb
from gbharness.vision.multiplayer import split_player_frames

ZELDA_ACTIONS: Final[dict[str, Action]] = {
    "move_up": hold("up", 18),
    "move_down": hold("down", 18),
    "move_left": hold("left", 18),
    "move_right": hold("right", 18),
    "nudge_up": hold("up", 5),
    "nudge_down": hold("down", 5),
    "nudge_left": hold("left", 5),
    "nudge_right": hold("right", 5),
    "micro_up": hold("up", 2),
    "micro_down": hold("down", 2),
    "micro_left": hold("left", 2),
    "micro_right": hold("right", 2),
    "face_up": tap("up", hold_frames=2, gap_frames=2),
    "face_down": tap("down", hold_frames=2, gap_frames=2),
    "face_left": tap("left", hold_frames=2, gap_frames=2),
    "face_right": tap("right", hold_frames=2, gap_frames=2),
    "attack_up": hold(["up", "b"], 10),
    "attack_down": hold(["down", "b"], 10),
    "attack_left": hold(["left", "b"], 10),
    "attack_right": hold(["right", "b"], 10),
    "interact": tap("a", hold_frames=3, gap_frames=9),
    "use_sword": tap("b", hold_frames=3, gap_frames=9),
    "start": tap("start", hold_frames=3, gap_frames=12),
    "select": tap("select", hold_frames=3, gap_frames=12),
    "wait": wait(18),
}

ZELDA_ACTION_DESCRIPTIONS: Final[dict[str, str]] = {
    "move_up": "Move north for a short, controlled distance.",
    "move_down": "Move south for a short, controlled distance.",
    "move_left": "Move west for a short, controlled distance.",
    "move_right": "Move east for a short, controlled distance.",
    "nudge_up": "Move north by one fine positioning step.",
    "nudge_down": "Move south by one fine positioning step.",
    "nudge_left": "Move west by one fine positioning step.",
    "nudge_right": "Move east by one fine positioning step.",
    "micro_up": "Move north by one minimal positioning step.",
    "micro_down": "Move south by one minimal positioning step.",
    "micro_left": "Move west by one minimal positioning step.",
    "micro_right": "Move east by one minimal positioning step.",
    "face_up": "Face north with a minimal direction tap.",
    "face_down": "Face south with a minimal direction tap.",
    "face_left": "Face west with a minimal direction tap.",
    "face_right": "Face east with a minimal direction tap.",
    "attack_up": "Face north, move briefly, and attack with B.",
    "attack_down": "Face south, move briefly, and attack with B.",
    "attack_left": "Face west, move briefly, and attack with B.",
    "attack_right": "Face east, move briefly, and attack with B.",
    "interact": "Tap A to confirm a menu choice, interact, lift, or use the A item.",
    "use_sword": "Tap B to swing the sword or cancel a menu.",
    "start": "Tap Start to begin from a title/menu screen or open the pause screen.",
    "select": "Tap Select only when a menu visibly requires it.",
    "wait": "Release all controls briefly while an animation, link, or dialogue advances.",
}

FOUR_SWORDS_POLICY: Final = GamePolicy(
    name="zelda-four-swords",
    goal=(
        "Cooperate with all four players: keep the linked party together, solve the "
        "current room, defeat threats, and visibly reach the next room. "
        "Report terminal only after the whole party has crossed the room boundary or a "
        "clear completion/result screen is visible."
    ),
    actions=ZELDA_ACTIONS,
    descriptions=ZELDA_ACTION_DESCRIPTIONS,
)

FOUR_SWORDS_ROLES: Final[tuple[str, ...]] = (
    "Green Link: lead route selection and wait at gates until the team is assembled.",
    "Red Link: protect the group and help operate cooperative switches or formations.",
    "Blue Link: explore nearby side paths, then return promptly to the visible team.",
    "Purple Link: collect nearby resources and provide rear guard without splitting away.",
)


@dataclass(frozen=True, slots=True)
class FourSwordsBootstrapResult:
    """Evidence that deterministic setup reached the linked dungeon HUD."""

    frames_advanced: int
    handshake_confirms: int


@dataclass(frozen=True, slots=True)
class FirstTrainingPuzzleResult:
    """Visible evidence from the fast cooperative tutorial solution."""

    frames_advanced: int
    warp_sweep_steps: int
    reward_chests_visible: bool


_FOUR_SWORDS_SETUP: Final[tuple[Action, ...]] = (
    # Blank file -> name keyboard. Enter a one-letter name (A), move to END,
    # and leave enough time for all four save-file animations to settle.
    tap("a", hold_frames=3, gap_frames=9),
    wait(120),
    tap("a", hold_frames=3, gap_frames=9),
    tap("up", hold_frames=3, gap_frames=9),
    tap("right", hold_frames=3, gap_frames=9),
    tap("right", hold_frames=3, gap_frames=9),
    tap("right", hold_frames=3, gap_frames=9),
    tap("a", hold_frames=3, gap_frames=9),
    wait(600),
    # Select the new file, choose Four Swords, skip the opening, and enter the
    # four-system rendezvous screen. Inputs are synchronized on one core clock.
    tap("a", hold_frames=3, gap_frames=9),
    wait(360),
    tap("down", hold_frames=3, gap_frames=9),
    tap("a", hold_frames=3, gap_frames=9),
    wait(600),
    tap("start", hold_frames=3, gap_frames=12),
    wait(480),
    tap("start", hold_frames=3, gap_frames=12),
    wait(600),
)


def four_swords_gameplay_visible(frame: RawFrame, players: int) -> bool:
    """Recognize the in-dungeon heart HUD on every private viewport.

    This is deliberately a visual check, not a RAM address or emulator hook. The
    top HUD contains a stable row of saturated red hearts in gameplay; the title,
    linking, stage-select, and black transition screens do not.
    """
    for view in split_player_frames(frame, players):
        rgb = to_rgb(view)
        # Hearts occupy the top-centre band. Restricting the probe horizontally
        # avoids the stage-select volcano, whose saturated reds sit at far left.
        hud = rgb[3:17, 120:190]
        red = (hud[:, :, 0] > 180) & (hud[:, :, 1] < 120) & (hud[:, :, 2] < 120)
        if int(np.count_nonzero(red)) < 120:
            return False
    return True


def four_swords_dialogue_visible(frame: RawFrame) -> bool:
    """Detect a large black dialogue box in any player screen."""
    for frame_view in split_player_frames(frame, 4):
        if _private_dialogue_visible(to_rgb(frame_view)):
            return True
    return False


def four_swords_npc_adjacent(views: tuple[RawFrame, ...]) -> bool:
    """Detect the pink NPC next to the P2 screen centre from visible pixels."""
    if len(views) < 2:
        return False
    offset = _pink_npc_offset(to_rgb(views[1]))
    return offset is not None and offset[0] ** 2 + offset[1] ** 2 <= 30 * 30


def four_swords_visual_target_plan(
    stage: str,
    views: tuple[RawFrame, ...],
) -> dict[int, tuple[str, ...]]:
    """Move the P2 scout to a visible circular target from pixels only."""
    if stage != "probe_target" or len(views) < 2:
        return {}
    rgb = to_rgb(views[1])
    player = _link_position(rgb, 1) or (120, 88)
    grid = _forward_target_grid(rgb)
    if grid is None:
        return {1: _forward_exploration_step(rgb, player)}

    target = grid[3]
    if any(
        _squared_distance(player, active) <= 12 * 12
        and _squared_distance(target, active) <= 12 * 12
        for active in _active_target_centres(rgb)
    ):
        return {1: ("wait",)}
    return {1: _precision_route_step(player, target)}


def four_swords_probe_target_reached(views: tuple[RawFrame, ...]) -> bool:
    """Confirm that the P2 scout overlaps a detected circular target."""
    return four_swords_visual_target_plan("probe_target", views).get(1) == ("wait",)


@dataclass(slots=True)
class FourSwordsVisualTargetNavigator:
    """Keep each AI Link on a distinct screenshot-derived floor target."""

    assignments: dict[int, int] = field(default_factory=dict)
    tracked_targets: dict[int, tuple[int, int]] = field(default_factory=dict)
    reached_players: set[int] = field(default_factory=set)
    near_players: set[int] = field(default_factory=set)
    leaving_return_players: set[int] = field(default_factory=set)
    forward_grid_players: set[int] = field(default_factory=set)
    reached_misses: dict[int, int] = field(default_factory=dict)
    return_grid_absent_counts: dict[int, int] = field(default_factory=dict)
    cleared_return_players: set[int] = field(default_factory=set)
    portal_search_started: bool = False
    active_counts: dict[int, int] = field(default_factory=dict)
    active_frames: dict[int, int] = field(default_factory=dict)
    probe_reached: bool = False
    dialogue_attempts: int = 0

    def reset(self) -> None:
        """Clear first-room visual state before the team enters a new room."""
        self.assignments.clear()
        self.tracked_targets.clear()
        self.reached_players.clear()
        self.near_players.clear()
        self.leaving_return_players.clear()
        self.forward_grid_players.clear()
        self.reached_misses.clear()
        self.return_grid_absent_counts.clear()
        self.cleared_return_players.clear()
        self.portal_search_started = False
        self.active_counts.clear()
        self.active_frames.clear()
        self.probe_reached = False
        self.dialogue_attempts = 0

    def __call__(
        self,
        stage: str,
        views: tuple[RawFrame, ...],
    ) -> dict[int, tuple[str, ...]]:
        if stage == "reach_npc":
            return self._reach_npc_plan(views)
        if stage == "probe_target":
            return self._probe_plan(views)
        if stage == "advance_dialogue":
            return self._dialogue_plan(views)
        if stage != "reach_warps" or len(views) < 4:
            return {}

        if not self.portal_search_started:
            self.portal_search_started = True
            if self.probe_reached:
                self.forward_grid_players.add(1)
                self.leaving_return_players.update((2, 3))
            else:
                self.leaving_return_players.update((1, 2, 3))

        # Each private camera follows its Link near the screen centre. Move up
        # until the complete forward grid is visible. Do not use the return
        # grid below the starting position.
        preferred_slots = {1: 3, 2: 0, 3: 2}
        plans: dict[int, tuple[str, ...]] = {}
        for player in (1, 2, 3):
            grid: tuple[tuple[int, int], ...] | None = None
            rgb = to_rgb(views[player])
            position = _link_position(rgb, player) or (120, 88)
            active_targets = _active_target_centres(rgb)
            visible_grid = _forward_target_grid(rgb)
            marked_grid = _forward_marked_target_grid(rgb)
            visible_targets = _merge_target_centres(
                _orange_target_centres(rgb),
                active_targets,
            )
            if player in self.leaving_return_players:
                if player != 1 and 1 not in self.reached_players:
                    plans[player] = ("wait",)
                    continue
                portal_visible = len(visible_targets) >= 2
                if player not in self.cleared_return_players:
                    if portal_visible:
                        self.return_grid_absent_counts[player] = 0
                    else:
                        absent = self.return_grid_absent_counts.get(player, 0) + 1
                        self.return_grid_absent_counts[player] = absent
                        if absent >= 2:
                            self.cleared_return_players.add(player)
                    plans[player] = _forward_exploration_step(rgb, position)
                    continue
                if marked_grid is None:
                    teammate_step = _visible_reached_teammate_step(
                        rgb,
                        player,
                        position,
                        self.reached_players,
                    )
                    if teammate_step is not None:
                        plans[player] = teammate_step
                        continue
                    plans[player] = _forward_exploration_step(rgb, position)
                    continue
                if player != 1 and not _scout_occupies_grid(rgb, marked_grid):
                    plans[player] = _forward_exploration_step(rgb, position)
                    continue
                self.leaving_return_players.discard(player)
                self.forward_grid_players.add(player)
                grid = marked_grid
            else:
                grid = visible_grid if player in self.forward_grid_players else None
            if player in self.reached_players:
                if any(_squared_distance(position, active) <= 12 * 12 for active in active_targets):
                    self.reached_misses[player] = 0
                    plans[player] = ("wait",)
                    continue
                misses = self.reached_misses.get(player, 0) + 1
                self.reached_misses[player] = misses
                if misses < 6:
                    plans[player] = ("wait",)
                    continue
                self.reached_players.discard(player)
                self.reached_misses.pop(player, None)
                self.active_counts[player] = 0
                self.active_frames.pop(player, None)
            if grid is None:
                previous_target = self.tracked_targets.get(player)
                assigned_slot = self.assignments.get(player)
                partial_target = (
                    _partial_portal_target(
                        rgb,
                        assigned_slot,
                        previous_target,
                    )
                    if assigned_slot is not None
                    and (previous_target is not None or self.reached_players)
                    else None
                )
                if partial_target is not None:
                    self.tracked_targets[player] = partial_target
                    plans[player] = _precision_route_step(position, partial_target)
                    continue
                teammate_step = _visible_reached_teammate_step(
                    rgb,
                    player,
                    position,
                    self.reached_players,
                )
                if teammate_step is not None:
                    plans[player] = teammate_step
                    continue
                plans[player] = _forward_exploration_step(rgb, position)
                continue
            self.leaving_return_players.discard(player)
            slot = self.assignments.get(player)
            if slot is None:
                slot = _empty_target_slot(
                    grid,
                    active_targets,
                    position,
                    set(self.assignments.values()),
                    preferred_slots[player],
                )
                self.assignments[player] = slot
            target = grid[slot]
            active_now = any(
                _squared_distance(position, active) <= 12 * 12
                and _squared_distance(target, active) <= 12 * 12
                for active in active_targets
            )
            if active_now:
                frame_index = views[player].index
                if self.active_frames.get(player) != frame_index:
                    self.active_counts[player] = self.active_counts.get(player, 0) + 1
                    self.active_frames[player] = frame_index
                if self.active_counts[player] >= 2:
                    self.reached_players.add(player)
                    self.reached_misses[player] = 0
                plans[player] = ("wait",)
                continue
            self.active_counts[player] = 0
            self.active_frames.pop(player, None)

            self.tracked_targets[player] = target
            plans[player] = _precision_route_step(position, target)
        return plans

    def _reach_npc_plan(
        self,
        views: tuple[RawFrame, ...],
    ) -> dict[int, tuple[str, ...]]:
        if len(views) < 2:
            return {}
        offset = _pink_npc_offset(to_rgb(views[1]))
        if offset is None:
            return {1: ("move_up",)}
        if _squared_distance((0, 0), offset) <= 30 * 30:
            return {1: ("wait",)}
        return {1: _precision_route_step((0, 0), offset)}

    def _probe_plan(
        self,
        views: tuple[RawFrame, ...],
    ) -> dict[int, tuple[str, ...]]:
        if self.probe_reached:
            return {1: ("wait",)}
        if len(views) < 2:
            return {}

        rgb = to_rgb(views[1])
        position = _link_position(rgb, 1) or (120, 88)
        grid = _forward_target_grid(rgb)
        if grid is None:
            return {1: _forward_exploration_step(rgb, position)}

        target = grid[3]
        if any(
            _squared_distance(position, active) <= 12 * 12
            and _squared_distance(target, active) <= 12 * 12
            for active in _active_target_centres(rgb)
        ):
            self.probe_reached = True
            return {1: ("wait",)}
        return {1: _precision_route_step(position, target)}

    def _dialogue_plan(
        self,
        views: tuple[RawFrame, ...],
    ) -> dict[int, tuple[str, ...]]:
        if len(views) < 2:
            return {}
        rgb = to_rgb(views[1])
        if _private_dialogue_visible(rgb):
            self.dialogue_attempts = 0
            return {1: ("interact", "wait", "wait", "wait", "wait", "wait")}

        self.dialogue_attempts += 1
        if self.dialogue_attempts >= 3:
            self.dialogue_attempts = 0
            return {1: ("move_down", "face_up", "interact", "wait", "wait")}

        offset = _pink_npc_offset(rgb)
        if offset is None:
            return {1: ("face_up", "interact", "wait", "wait")}
        dx, dy = offset
        if abs(dx) > abs(dy):
            facing = "face_right" if dx > 0 else "face_left"
        else:
            facing = "face_down" if dy > 0 else "face_up"
        return {1: (facing, "interact", "wait", "wait")}

    def probe_target_reached(self, views: tuple[RawFrame, ...]) -> bool:
        """Confirm that P2 covered its selected floor target."""
        self._probe_plan(views)
        return self.probe_reached

    def all_assigned_players_reached(self, views: tuple[RawFrame, ...]) -> bool:
        """Confirm that P2 through P4 overlap their fixed visual targets."""
        plans = self("reach_warps", views)
        return self.reached_players == {1, 2, 3} and all(
            plans.get(player) == ("wait",) for player in (1, 2, 3)
        )


def _squared_distance(first: tuple[int, int], second: tuple[int, int]) -> int:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _private_dialogue_visible(rgb: np.ndarray) -> bool:
    region = rgb[92:148, 4:236]
    dark = np.max(region, axis=2) < 45
    return float(np.count_nonzero(dark)) / float(dark.size) >= 0.45


def _pink_npc_offset(rgb: np.ndarray) -> tuple[int, int] | None:
    npc = _pink_npc_position(rgb)
    if npc is None:
        return None
    player = _link_position(rgb, 1) or (120, 88)
    return npc[0] - player[0], npc[1] - player[1]


def _pink_npc_position(rgb: np.ndarray) -> tuple[int, int] | None:
    """Find the upper pink NPC without selecting Purple Link."""
    region_x = 70
    region_y = 15
    nearby = rgb[region_y:150, region_x:171]
    pink = (
        (nearby[:, :, 0] > 140)
        & (nearby[:, :, 2] > 100)
        & (nearby[:, :, 0] > nearby[:, :, 1] * 1.25)
        & (nearby[:, :, 2] > nearby[:, :, 1] * 1.10)
    )
    candidates: list[tuple[int, int]] = []
    for xs, ys in _components(pink):
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        # The NPC has one large connected pink body. Purple Link has two
        # smaller pink regions that are separated by yellow hair and skin.
        if not 55 <= len(xs) <= 180 or not 10 <= width <= 24 or not 8 <= height <= 28:
            continue
        candidates.append((int(np.mean(xs)) + region_x, int(np.mean(ys)) + region_y))
    if not candidates:
        return None
    player = _link_position(rgb, 1) or (120, 88)
    return min(candidates, key=lambda point: _squared_distance(point, player))


def _route_step(
    player: tuple[int, int],
    target: tuple[int, int],
) -> tuple[str, ...]:
    dx = target[0] - player[0]
    dy = target[1] - player[1]
    if dx * dx + dy * dy <= 16 * 16:
        return ("wait",)
    if abs(dx) > abs(dy):
        action = "move_right" if dx > 0 else "move_left"
        distance = abs(dx)
    else:
        action = "move_down" if dy > 0 else "move_up"
        distance = abs(dy)
    return (action, action) if distance > 28 else (action,)


def _single_route_step(
    player: tuple[int, int],
    target: tuple[int, int],
) -> tuple[str, ...]:
    """Move one controlled step toward a target."""
    dx = target[0] - player[0]
    dy = target[1] - player[1]
    if abs(dx) > abs(dy):
        return ("move_right" if dx > 0 else "move_left",)
    return ("move_down" if dy > 0 else "move_up",)


def _precision_route_step(
    player: tuple[int, int],
    target: tuple[int, int],
) -> tuple[str, ...]:
    """Use a short direction tap near a target to prevent overshoot."""
    dx = target[0] - player[0]
    dy = target[1] - player[1]
    distance = dx * dx + dy * dy
    if distance == 0:
        return ("micro_down",)
    if distance <= 10 * 10:
        prefix = "micro"
    elif distance <= 28 * 28:
        prefix = "nudge"
    else:
        prefix = "move"
    if abs(dx) > abs(dy):
        return (f"{prefix}_{'right' if dx > 0 else 'left'}",)
    return (f"{prefix}_{'down' if dy > 0 else 'up'}",)


def _visible_reached_teammate_step(
    rgb: np.ndarray,
    player: int,
    position: tuple[int, int],
    reached_players: set[int],
) -> tuple[str, ...] | None:
    """Move toward a visible teammate that reached the forward portal."""
    active_targets = _active_target_centres(rgb)
    candidates: list[tuple[int, tuple[int, int]]] = []
    for teammate in reached_players:
        if teammate == player:
            continue
        teammate_position = _link_position(rgb, teammate)
        if teammate_position is None:
            continue
        if not any(
            _squared_distance(teammate_position, target) <= 16 * 16 for target in active_targets
        ):
            continue
        distance = _squared_distance(position, teammate_position)
        if distance <= 24 * 24:
            continue
        candidates.append((distance, teammate_position))
    if not candidates:
        return None
    _, target = max(candidates)
    return _precision_route_step(position, target)


def _scout_occupies_grid(
    rgb: np.ndarray,
    grid: tuple[tuple[int, int], ...],
) -> bool:
    """Confirm that the red P2 scout occupies this visible portal grid."""
    scout = _link_position(rgb, 1)
    if scout is None:
        return False
    active_targets = _active_target_centres(rgb)
    return any(
        _squared_distance(scout, target) <= 18 * 18
        and any(_squared_distance(target, active) <= 12 * 12 for active in active_targets)
        for target in grid
    )


def _partial_portal_target(
    rgb: np.ndarray,
    slot: int,
    previous: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Infer an assigned target from one visible portal row or column."""
    if previous is None:
        orange_targets = _orange_target_centres(rgb)
        if not orange_targets:
            return None
        ordered = sorted(orange_targets, key=lambda point: (point[0], abs(point[1] - 88)))
        return ordered[0] if slot % 2 == 0 else ordered[-1]
    targets = _merge_target_centres(
        _orange_target_centres(rgb),
        _active_target_centres(rgb),
    )
    candidates: list[tuple[int, tuple[int, int]]] = []
    for first, second in combinations(targets, 2):
        dx = abs(first[0] - second[0])
        dy = abs(first[1] - second[1])
        if 28 <= dx <= 70 and dy <= 12:
            left, right = sorted((first, second))
            row_y = (left[1] + right[1]) // 2
            for top_y in (row_y, row_y - dx):
                grid = (
                    (left[0], top_y),
                    (right[0], top_y),
                    (left[0], top_y + dx),
                    (right[0], top_y + dx),
                )
                target = grid[slot]
                candidates.append((_squared_distance(target, previous), target))
        if 28 <= dy <= 70 and dx <= 12:
            upper, lower = sorted((first, second), key=lambda point: point[1])
            column_x = (upper[0] + lower[0]) // 2
            for left_x in (column_x, column_x - dy):
                grid = (
                    (left_x, upper[1]),
                    (left_x + dy, upper[1]),
                    (left_x, lower[1]),
                    (left_x + dy, lower[1]),
                )
                target = grid[slot]
                candidates.append((_squared_distance(target, previous), target))
    if not candidates:
        return None
    return min(candidates)[1]


def _orange_target_centres(rgb: np.ndarray) -> tuple[tuple[int, int], ...]:
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = (red > 210) & (green > 45) & (green < 210) & (blue < 105)
    centres: list[tuple[int, int]] = []
    for xs, ys in _components(mask):
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        if len(xs) < 55 or not 16 <= width <= 55 or not 16 <= height <= 55:
            continue
        ratio = width / height
        if 0.65 <= ratio <= 1.35 and int(ys.min()) >= 20:
            centres.append((int(np.mean(xs)), int(np.mean(ys))))
    return tuple(centres)


def _active_target_centres(rgb: np.ndarray) -> tuple[tuple[int, int], ...]:
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = (red > 145) & (red < 205) & (green > 195) & (blue > 230)
    white = (red > 230) & (green > 230) & (blue > 230)
    centres: list[tuple[int, int]] = []
    for xs, ys in _components(mask):
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        if len(xs) < 80 or not 16 <= width <= 45 or not 16 <= height <= 45:
            continue
        x0 = max(0, int(xs.min()) - 3)
        x1 = min(rgb.shape[1], int(xs.max()) + 4)
        y0 = max(0, int(ys.min()) - 3)
        y1 = min(rgb.shape[0], int(ys.max()) + 4)
        if int(np.count_nonzero(white[y0:y1, x0:x1])) < 25:
            continue
        centres.append((int(np.mean(xs)), int(np.mean(ys))))
    return tuple(centres)


def _merge_target_centres(
    first: tuple[tuple[int, int], ...],
    second: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for target in (*first, *second):
        if all(_squared_distance(target, existing) > 12 * 12 for existing in merged):
            merged.append(target)
    return tuple(merged)


def _order_target_grid(
    targets: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Order four targets as upper-left, upper-right, lower-left, lower-right."""
    if len(targets) != 4:
        return tuple(sorted(targets, key=lambda point: (point[1], point[0])))
    vertical = sorted(targets, key=lambda point: point[1])
    upper = sorted(vertical[:2], key=lambda point: point[0])
    lower = sorted(vertical[2:], key=lambda point: point[0])
    return tuple((*upper, *lower))


def _empty_target_slot(
    grid: tuple[tuple[int, int], ...],
    active_targets: tuple[tuple[int, int], ...],
    position: tuple[int, int],
    reserved_slots: set[int],
    preferred_slot: int,
) -> int:
    """Select one unoccupied and unreserved circle for one bot."""
    active_slots = {
        slot
        for slot, target in enumerate(grid)
        if any(_squared_distance(target, active) <= 12 * 12 for active in active_targets)
    }
    own_slot = min(range(len(grid)), key=lambda slot: _squared_distance(position, grid[slot]))
    if (
        own_slot in active_slots
        and own_slot not in reserved_slots
        and _squared_distance(position, grid[own_slot]) <= 14 * 14
    ):
        return own_slot

    available = [
        slot for slot in range(len(grid)) if slot not in active_slots and slot not in reserved_slots
    ]
    if preferred_slot in available:
        return preferred_slot
    if not available:
        available = [slot for slot in range(len(grid)) if slot not in reserved_slots]
    if not available:
        available = list(range(len(grid)))
    return min(available, key=lambda slot: _squared_distance(position, grid[slot]))


def _target_grids(
    targets: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Find compact two-by-two target grids in visible target centres."""
    grids: list[tuple[tuple[int, int], ...]] = []
    for candidate in combinations(targets, 4):
        upper_left, upper_right, lower_left, lower_right = _order_target_grid(candidate)
        horizontal = (
            upper_right[0] - upper_left[0],
            lower_right[0] - lower_left[0],
        )
        vertical = (
            lower_left[1] - upper_left[1],
            lower_right[1] - upper_right[1],
        )
        aligned = (
            abs(upper_left[1] - upper_right[1]) <= 12
            and abs(lower_left[1] - lower_right[1]) <= 12
            and abs(upper_left[0] - lower_left[0]) <= 12
            and abs(upper_right[0] - lower_right[0]) <= 12
        )
        if not aligned:
            continue
        if not all(28 <= gap <= 70 for gap in (*horizontal, *vertical)):
            continue
        grids.append((upper_left, upper_right, lower_left, lower_right))
    return tuple(
        sorted(
            grids,
            key=lambda grid: sum(point[1] for point in grid) / len(grid),
        )
    )


def _forward_target_grid(rgb: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    targets = _merge_target_centres(
        _orange_target_centres(rgb),
        _active_target_centres(rgb),
    )
    grids = _target_grids(targets)
    return grids[0] if grids else None


def _forward_marked_target_grid(
    rgb: np.ndarray,
) -> tuple[tuple[int, int], ...] | None:
    """Return a portal grid only when its red-orange forward marker is visible."""
    targets = _merge_target_centres(
        _orange_target_centres(rgb),
        _active_target_centres(rgb),
    )
    grids = _target_grids(targets)
    for grid in grids:
        centre_x = sum(point[0] for point in grid) // len(grid)
        centre_y = sum(point[1] for point in grid) // len(grid)
        marker = rgb[
            max(0, centre_y - 12) : min(rgb.shape[0], centre_y + 13),
            max(0, centre_x - 12) : min(rgb.shape[1], centre_x + 13),
        ]
        red = marker[:, :, 0]
        green = marker[:, :, 1]
        blue = marker[:, :, 2]
        forward_pixels = (red > 150) & (green < 120) & (blue < 110)
        return_pixels = (
            (red > 140)
            & (blue > 100)
            & (red.astype(np.int16) > green.astype(np.int16) * 1.25)
            & (blue.astype(np.int16) > green.astype(np.int16) * 1.10)
        )
        forward_count = int(np.count_nonzero(forward_pixels))
        return_count = int(np.count_nonzero(return_pixels))
        if forward_count >= 15 and forward_count > return_count:
            return grid
    return None


def _forward_exploration_step(
    rgb: np.ndarray,
    position: tuple[int, int],
) -> tuple[str, ...]:
    opening = _visible_wall_opening(rgb, position)
    if opening is None:
        return ("move_up",)
    dx = opening[0] - position[0]
    if abs(dx) <= 7:
        return ("move_up",)
    return _precision_route_step(position, (opening[0], position[1]))


def _visible_wall_opening(
    rgb: np.ndarray,
    position: tuple[int, int],
) -> tuple[int, int] | None:
    """Find a visible floor gap between two horizontal wall segments."""
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    wall = (
        (red > 80)
        & (green > 35)
        & (green < 235)
        & (blue > 20)
        & (blue < 190)
        & (red > green + 18)
        & (red > blue + 15)
    )
    candidates: list[tuple[int, int, int]] = []
    max_y = min(position[1] - 8, 104)
    for centre_y in range(28, max_y + 1):
        density = wall[max(0, centre_y - 8) : centre_y + 9].mean(axis=0)
        runs = _closed_runs(density >= 0.28, close_gap=15, min_length=30)
        for left, right in zip(runs, runs[1:], strict=False):
            gap_start = left[1] + 1
            gap_end = right[0] - 1
            gap_width = gap_end - gap_start + 1
            if not 24 <= gap_width <= 72:
                continue
            wall_width = left[1] - left[0] + right[1] - right[0] + 2
            candidates.append(((gap_start + gap_end) // 2, centre_y, wall_width))
    if not candidates:
        return None
    opening_x, opening_y, _ = max(
        candidates,
        key=lambda candidate: (
            -abs(candidate[0] - position[0]),
            candidate[2],
            candidate[1],
        ),
    )
    return opening_x, opening_y


def _closed_runs(
    mask: np.ndarray,
    *,
    close_gap: int,
    min_length: int,
) -> tuple[tuple[int, int], ...]:
    runs: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        if start is not None and (not bool(value) or index == len(mask) - 1):
            end = index if bool(value) and index == len(mask) - 1 else index - 1
            if runs and start - runs[-1][1] - 1 <= close_gap:
                runs[-1][1] = end
            else:
                runs.append([start, end])
            start = None
    return tuple((start_x, end_x) for start_x, end_x in runs if end_x - start_x + 1 >= min_length)


def _link_position(rgb: np.ndarray, player: int) -> tuple[int, int] | None:
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    if player == 1:
        mask = (red > 165) & (green < 100) & (blue < 100)
    elif player == 2:
        blue_wide = blue.astype(np.int16)
        red_wide = red.astype(np.int16)
        mask = (blue > 105) & (red < 115) & (green < 175) & (blue_wide > red_wide + 35)
    elif player == 3:
        mask = (red > 105) & (blue > 105) & (green < 125)
    else:
        raise ValueError(f"unsupported visual Link player index: {player}")
    yellow = (red > 200) & (green > 140) & (blue < 130)
    candidates: list[tuple[float, int, int]] = []
    for xs, ys in _components(mask):
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        if not 4 <= len(xs) <= 180 or width > 22 or height > 28 or int(ys.min()) < 20:
            continue
        x0 = max(0, int(xs.min()) - 10)
        x1 = min(rgb.shape[1], int(xs.max()) + 11)
        y0 = max(0, int(ys.min()) - 14)
        y1 = min(rgb.shape[0], int(ys.max()) + 9)
        if int(np.count_nonzero(yellow[y0:y1, x0:x1])) < 4:
            continue
        x = int(np.mean(xs))
        y = int(np.mean(ys))
        distance = float((x - 120) ** 2 + (y - 82) ** 2)
        candidates.append((distance, x, y))
    if not candidates:
        return None
    _, anchor_x, anchor_y = min(candidates)
    selected = min(
        _components(mask),
        key=lambda component: _squared_distance(
            (int(np.mean(component[0])), int(np.mean(component[1]))),
            (anchor_x, anchor_y),
        ),
    )
    selected_x, selected_y = selected
    return int(np.mean(selected_x)), int(selected_y.max() + 4)


def _components(mask: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return eight-connected pixel components for a small boolean mask."""
    pending = mask.copy()
    height, width = pending.shape
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for start_y, start_x in zip(*np.nonzero(pending), strict=True):
        if not pending[start_y, start_x]:
            continue
        stack = [(int(start_x), int(start_y))]
        pending[start_y, start_x] = False
        xs: list[int] = []
        ys: list[int] = []
        while stack:
            x, y = stack.pop()
            xs.append(x)
            ys.append(y)
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if pending[next_y, next_x]:
                        pending[next_y, next_x] = False
                        stack.append((next_x, next_y))
        result.append((np.asarray(xs), np.asarray(ys)))
    return tuple(result)


def bootstrap_four_swords(
    session: LinkedSession,
    frames: FrameLatch,
    *,
    max_handshake_confirms: int = 16,
) -> FourSwordsBootstrapResult:
    """Create four local saves and enter the first linked room without model calls.

    Four Swords' four-unit handshake finishes on a variable retry. Confirming at a
    bounded cadence and checking the visible HUD prevents both premature model calls
    and an unbounded menu loop.
    """
    if session.port.players != 4:
        raise ValueError("the Four Swords agent bootstrap requires exactly four players")
    if max_handshake_confirms < 1:
        raise ValueError("max_handshake_confirms must be positive")

    started = session.frame

    def every_player(action: Action) -> None:
        result = session.submit({player: action for player in range(4)})
        if result.emulation_ended:
            raise RuntimeError("emulation ended during Four Swords setup")

    for action in _FOUR_SWORDS_SETUP:
        every_player(action)

    for attempt in range(1, max_handshake_confirms + 1):
        every_player(tap("a", hold_frames=3, gap_frames=9))
        every_player(wait(120))
        if four_swords_gameplay_visible(frames.require(), 4):
            return FourSwordsBootstrapResult(
                frames_advanced=session.frame - started,
                handshake_confirms=attempt,
            )

    raise RuntimeError(
        "Four Swords did not reach four-player gameplay after "
        f"{max_handshake_confirms} synchronized handshake confirmations"
    )


def first_training_reward_visible(frame: RawFrame) -> bool:
    """Recognize both reward chests revealed by the first switch puzzle."""
    view = to_rgb(split_player_frames(frame, 4)[0])
    regions = (view[75:110, 25:55], view[75:110, 185:215])
    return all(
        int(
            np.count_nonzero(
                (region[:, :, 0] > 240) & (region[:, :, 1] > 210) & (region[:, :, 2] < 80)
            )
        )
        >= 50
        for region in regions
    )


def solve_first_training_puzzle(
    session: LinkedSession,
    frames: FrameLatch,
    *,
    on_phase: Callable[[str], None] | None = None,
    telemetry: RunTelemetry | None = None,
) -> FirstTrainingPuzzleResult:
    """Enter Basic Training 1 and reveal its two switch-reward chests.

    Models choose strategy; this bounded visual motor skill handles pixel-level
    movement at emulator speed. It verifies the warp and final rewards from pixels
    without reading RAM or restoring a save state.
    """
    if session.port.players != 4:
        raise ValueError("the first Four Swords training puzzle requires four players")

    emit = on_phase or (lambda _message: None)
    trace = telemetry or NullTelemetry()
    started = session.frame

    def one(player: int, action: Action) -> None:
        result = session.submit({player: action})
        if result.emulation_ended:
            raise RuntimeError("emulation ended during the first training puzzle")

    def team(actions: dict[int, Action]) -> None:
        result = session.submit(actions)
        if result.emulation_ended:
            raise RuntimeError("emulation ended during the first training puzzle")

    with trace.span(
        "solve_first_training_puzzle",
        "CHAIN",
        input_value={
            "puzzle": "Basic Training 1 outer switches",
            "participants": 4,
            "method": "deterministic local visual policy, no walkthrough access",
        },
        attributes={
            "puzzle.name": "Basic Training 1 outer switches",
            "puzzle.participants": 4,
            "motor_skill.source": "local_visual_policy",
            "runtime.walkthrough_access": False,
        },
    ) as puzzle_span:
        with trace.span(
            "unlock_tutorial_lobby",
            "TOOL",
            input_value={"P1": "right, up, interact, then dialogue confirms"},
            output_value={"lobby": "tutorial dialogue advanced for Green Link"},
            attributes={"neatlogs.tool.name": "bounded_controller_sequence"},
        ):
            emit("FAST MOTOR SKILL — unlocking the tutorial lobby")
            one(0, hold("right", 16))
            one(0, hold("up", 18))
            one(0, tap("a", hold_frames=3, gap_frames=9))
            one(0, wait(120))
            # Rapid taps leave the dialogue half-open and strand Green Link.
            for _ in range(12):
                one(0, tap("a", hold_frames=3, gap_frames=9))
                one(0, wait(120))

        with trace.span(
            "assemble_team_on_warp",
            "TOOL",
            input_value={
                "goal": "move all four Links onto the cooperative warp",
                "approach": "P1-P4 bounded approach vectors, then a 11x11 sweep for P4",
            },
            attributes={"neatlogs.tool.name": "four_player_warp_assembly"},
        ) as warp_span:
            emit("FAST MOTOR SKILL — assembling all four Links on the warp")
            approaches = (
                (0, 0, -24, 18),
                (1, -16, 24, 18),
                (2, 48, -24, 48),
                (3, -48, 24, 48),
            )
            for player, centre, side, depth in approaches:
                if centre:
                    one(player, hold("right" if centre > 0 else "left", abs(centre)))
                one(player, hold("up", 130))
                one(player, hold("right" if side > 0 else "left", abs(side)))
                one(player, hold("down", depth))

            one(1, hold("up", 15))
            reference = to_rgb(frames.require()).astype(np.int16)

            # Collision at the narrow door makes Purple's final coordinate vary. Sweep
            # a finite 44x44 box and stop on the visible room transition.
            one(3, hold("left", 20))
            one(3, hold("up", 20))
            sweep_steps = 0
            entered = False
            for row in range(11):
                direction = "right" if row % 2 == 0 else "left"
                for _column in range(11):
                    one(3, hold(direction, 4))
                    one(0, wait(20))
                    sweep_steps += 1
                    current = to_rgb(frames.require()).astype(np.int16)
                    if float(np.mean(np.abs(current - reference))) > 35:
                        entered = True
                        break
                if entered:
                    break
                one(3, hold("down", 4))
            trace.annotate(warp_span, {"warp.sweep_steps": sweep_steps, "warp.entered": entered})
            if not entered:
                raise RuntimeError("the four-player tutorial warp did not activate")

        with trace.span(
            "collect_rupees_and_regroup",
            "TOOL",
            input_value={"plan": "P1-P4 move south; P2 and P4 realign left"},
            output_value={"team": "four entry Rupees collected and team regrouped"},
            attributes={"neatlogs.tool.name": "synchronized_team_movement"},
        ):
            one(0, wait(500))
            emit("FAST MOTOR SKILL — collecting four Rupees and regrouping")
            team({player: hold("down", 120) for player in range(4)})
            one(0, wait(120))
            team({1: hold("left", 24), 3: hold("left", 24)})
            team({1: hold("down", 120), 3: hold("down", 120)})
            one(0, wait(180))

        with trace.span(
            "activate_outer_switches",
            "TOOL",
            input_value={"plan": "P1 left then down; P4 right then down; P2 and P3 hold"},
            output_value={"switches": "Green and Purple placed on both outer switches"},
            attributes={"neatlogs.tool.name": "cooperative_switch_activation"},
        ):
            emit("COOPERATIVE PUZZLE — Green and Purple taking both switches")
            team({0: hold("left", 16), 3: hold("right", 21)})
            team({0: hold("down", 55), 3: hold("down", 55)})
            one(0, wait(300))

        with trace.span(
            "verify_reward_chests_visible",
            "GUARDRAIL",
            input_value={
                "question": "did the puzzle reveal both reward chests?",
                "source": "P1 framebuffer reward-chest regions",
            },
        ) as verification_span:
            visible = first_training_reward_visible(frames.require())
            trace.annotate(verification_span, {"neatlogs.guardrail.passed": visible})
            trace.record_output(
                verification_span,
                "GUARDRAIL",
                "both reward chests visible" if visible else "reward chests absent",
            )
            if not visible:
                raise RuntimeError("the first puzzle did not reveal both reward chests")
        trace.annotate(
            puzzle_span,
            {
                "puzzle.completed": visible,
                "puzzle.frames_advanced": session.frame - started,
                "puzzle.warp_sweep_steps": sweep_steps,
            },
        )
        trace.record_output(
            puzzle_span,
            "CHAIN",
            {
                "completed": visible,
                "frames_advanced": session.frame - started,
                "warp_sweep_steps": sweep_steps,
            },
        )
        trace.log(
            "first Four Swords training puzzle solved in {frames} frames; "
            "reward chests are visible",
            frames=session.frame - started,
        )
        emit("PUZZLE 1 SOLVED — both reward chests are visible")

    return FirstTrainingPuzzleResult(
        frames_advanced=session.frame - started,
        warp_sweep_steps=sweep_steps,
        reward_chests_visible=visible,
    )
