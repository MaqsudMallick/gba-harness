"""Bounded controller vocabulary for Zelda: Four Swords cooperative play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from gbharness.actions import Action, hold, tap, wait
from gbharness.agent.contracts import GamePolicy
from gbharness.multiplayer import LinkedSession
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
