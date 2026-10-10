"""The action space, and the questions that choose from it.

A Game Boy Advance agent needs no tool calling. The action space is ten buttons and a few
durations -- small, discrete, enumerable -- so choosing from it is classification, which is
exactly what the Decisions API does (PLAN-agent-layer.md §0).

The model picks a *key*; this module owns what that key does. No model output is ever
interpreted as code or as free-form arguments, which removes a whole class of failure that
tool-calling designs have to guard against.
"""

from __future__ import annotations

from typing import Final

from openai.types.decision_create_params import (
    Question,
    QuestionQuestionParamChoice,
    QuestionQuestionParamPredicate,
)

from gbharness.actions import Action, hold, tap, wait

#: Choice value -> the input-layer action it performs. Durations are tuned so one turn is a
#: meaningful amount of game, not a twitch: crossing a screen should take a few turns, not
#: two hundred. Turn count drives both cost and wall-clock, so this table matters more than
#: any model setting (PLAN-agent-layer.md §6).
ACTIONS: Final[dict[str, Action]] = {
    "jump": tap("a", hold_frames=10, gap_frames=4),
    "jump_right": hold(["right", "a"], 22),
    "jump_left": hold(["left", "a"], 22),
    "run_right": hold(["right", "b"], 28),
    "run_left": hold(["left", "b"], 28),
    "walk_right": hold("right", 14),
    "walk_left": hold("left", 14),
    "crouch": hold("down", 10),
    "look_up": hold("up", 10),
    "start": tap("start", hold_frames=4, gap_frames=10),
    "select": tap("select", hold_frames=4, gap_frames=10),
    "wait": wait(24),
}

ACTION_DESCRIPTIONS: Final[dict[str, str]] = {
    "jump": "Jump straight up on the spot.",
    "jump_right": "Jump while moving right — use to clear a gap, a pipe, or an enemy ahead.",
    "jump_left": "Jump while moving left.",
    "run_right": "Run right at speed. The fastest way to make progress.",
    "run_left": "Run left at speed.",
    "walk_right": "Walk right slowly and carefully.",
    "walk_left": "Walk left slowly and carefully.",
    "crouch": "Duck down.",
    "look_up": "Look upward, or climb/enter a door above.",
    "start": "Press Start — begins the game from a title screen, or pauses during play.",
    "select": "Press Select — changes the highlighted option on a menu.",
    "wait": "Do nothing for a moment. Use while an animation, cutscene or death plays out.",
}

ACTION_QUESTION: Final[QuestionQuestionParamChoice] = {
    "type": "choice",
    "name": "action",
    "instructions": (
        "The image is the current screen of a Game Boy Advance game: Super Mario Bros. "
        "Choose the single best next input to make progress. "
        "Mario moves right through the level toward a flagpole. Jump over gaps, pipes and "
        "enemies; falling into a gap kills him. If a title screen or menu is showing, "
        "press Start to begin. If an animation or death is playing, wait."
    ),
    "choices": [{"value": k, "description": v} for k, v in ACTION_DESCRIPTIONS.items()],
}

SCREEN_QUESTION: Final[QuestionQuestionParamChoice] = {
    "type": "choice",
    "name": "screen_kind",
    "instructions": "What kind of screen is showing?",
    "choices": [
        {"value": "title", "description": "A title or attract screen before play begins."},
        {"value": "menu", "description": "A menu with selectable options."},
        {"value": "gameplay", "description": "Active play: Mario is on screen in a level."},
        {"value": "cutscene", "description": "A transition, score tally, or 'World 1-1' card."},
        {"value": "game_over", "description": "Mario has died or the game is over."},
    ],
}

WAITING_QUESTION: Final[QuestionQuestionParamPredicate] = {
    "type": "predicate",
    "name": "waiting_for_input",
    "instructions": (
        "Is the game idle and waiting for a button press, as opposed to an animation or "
        "movement being in progress?"
    ),
}

QUESTIONS: Final[list[Question]] = [
    ACTION_QUESTION,
    SCREEN_QUESTION,
    WAITING_QUESTION,
]


def resolve(choice: str) -> Action:
    """Map a chosen value to an input-layer action.

    Raises:
        KeyError: if the model returned a value outside the choice set, which should be
            impossible -- the endpoint constrains it -- but is worth failing loudly on.
    """
    try:
        return ACTIONS[choice]
    except KeyError:
        raise KeyError(
            f"model chose {choice!r}, which is not in the action table: "
            f"{sorted(ACTIONS)}"
        ) from None
