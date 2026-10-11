"""Bounded controller vocabulary for Zelda: Four Swords cooperative play."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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

    with trace.span("solve_first_training_puzzle", "WORKFLOW"):
        emit("FAST MOTOR SKILL — unlocking the tutorial lobby")
        one(0, hold("right", 16))
        one(0, hold("up", 18))
        one(0, tap("a", hold_frames=3, gap_frames=9))
        one(0, wait(120))
        # Rapid taps leave the dialogue half-open and strand Green Link.
        for _ in range(12):
            one(0, tap("a", hold_frames=3, gap_frames=9))
            one(0, wait(120))

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
        if not entered:
            raise RuntimeError("the four-player tutorial warp did not activate")

        one(0, wait(500))
        emit("FAST MOTOR SKILL — collecting four Rupees and regrouping")
        team({player: hold("down", 120) for player in range(4)})
        one(0, wait(120))
        team({1: hold("left", 24), 3: hold("left", 24)})
        team({1: hold("down", 120), 3: hold("down", 120)})
        one(0, wait(180))

        emit("COOPERATIVE PUZZLE — Green and Purple taking both switches")
        team({0: hold("left", 16), 3: hold("right", 21)})
        team({0: hold("down", 55), 3: hold("down", 55)})
        one(0, wait(300))

        visible = first_training_reward_visible(frames.require())
        if not visible:
            raise RuntimeError("the first puzzle did not reveal both reward chests")
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
