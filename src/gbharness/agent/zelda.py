"""Bounded controller vocabulary for Zelda: Four Swords cooperative play."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    targets = _orange_target_centres(to_rgb(views[1]))
    if not targets:
        # P2 finishes the NPC dialogue near the top wall. Move away from the
        # NPC until one complete floor target enters the private viewport.
        return {1: ("move_down",)}

    player = (120, 88)
    target = min(
        targets,
        key=lambda point: (point[0] - player[0]) ** 2 + (point[1] - player[1]) ** 2,
    )
    if _squared_distance(player, target) <= 6 * 6:
        return {1: ("wait",)}
    return {1: _single_route_step(player, target)}


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
    active_counts: dict[int, int] = field(default_factory=dict)
    active_frames: dict[int, int] = field(default_factory=dict)
    probe_near_target: bool = False
    probe_reached: bool = False
    dialogue_attempts: int = 0

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

        # Each private camera follows its Link near the screen centre. Assign
        # P2 to the lower-right target, P3 to upper-left, and P4 to lower-left. Keep a Link
        # still only when an active blue target ring is under that Link.
        target_slots = {1: 3, 2: 0, 3: 2}
        plans: dict[int, tuple[str, ...]] = {}
        for player in (1, 2, 3):
            self.assignments.setdefault(player, target_slots[player])
            if player in self.reached_players:
                plans[player] = ("wait",)
                continue
            rgb = to_rgb(views[player])
            position = _link_position(rgb, player) or (120, 88)
            active_targets = _active_target_centres(rgb)
            active_now = any(
                _squared_distance(position, target) <= 12 * 12
                for target in active_targets
            )
            if active_now:
                frame_index = views[player].index
                if self.active_frames.get(player) != frame_index:
                    self.active_counts[player] = self.active_counts.get(player, 0) + 1
                    self.active_frames[player] = frame_index
                if self.active_counts[player] >= 2:
                    self.reached_players.add(player)
                plans[player] = ("wait",)
                continue
            self.active_counts[player] = 0
            self.active_frames.pop(player, None)

            orange_targets = _orange_target_centres(rgb)
            all_targets = _merge_target_centres(orange_targets, active_targets)
            ordered_targets = _order_target_grid(all_targets)
            if len(ordered_targets) >= 4:
                target = ordered_targets[self.assignments[player]]
            elif orange_targets:
                target = min(
                    orange_targets,
                    key=lambda point: _squared_distance(point, position),
                )
            else:
                plans[player] = ("move_down",)
                continue
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

        anchor = (120, 88)
        targets = _orange_target_centres(to_rgb(views[1]))
        if not targets:
            if self.probe_near_target:
                self.probe_reached = True
                return {1: ("wait",)}
            return {1: ("move_down",)}

        target = min(targets, key=lambda point: _squared_distance(point, anchor))
        distance = _squared_distance(anchor, target)
        if self.probe_near_target and distance > 30 * 30:
            self.probe_reached = True
            return {1: ("wait",)}
        self.probe_near_target = distance <= 20 * 20
        if distance <= 16 * 16:
            self.probe_reached = True
            return {1: ("wait",)}
        return {1: _single_route_step(anchor, target)}

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
    region_y = 25
    nearby = rgb[region_y:105, region_x:171]
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
        candidates.append(
            (int(np.mean(xs)) + region_x, int(np.mean(ys)) + region_y)
        )
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
    prefix = "nudge" if dx * dx + dy * dy <= 28 * 28 else "move"
    if abs(dx) > abs(dy):
        return (f"{prefix}_{'right' if dx > 0 else 'left'}",)
    return (f"{prefix}_{'down' if dy > 0 else 'up'}",)


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


def _link_position(rgb: np.ndarray, player: int) -> tuple[int, int] | None:
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    if player == 1:
        mask = (red > 165) & (green < 100) & (blue < 100)
    elif player == 2:
        blue_wide = blue.astype(np.int16)
        red_wide = red.astype(np.int16)
        mask = (
            (blue > 105)
            & (red < 115)
            & (green < 175)
            & (blue_wide > red_wide + 35)
        )
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
