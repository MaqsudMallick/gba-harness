"""Legacy OpenAI Decisions schema for the Mario starter policy.

This adapter-specific representation is deliberately separate from ``policy.py`` so new
games and the shared runner do not import OpenAI types.
"""

from __future__ import annotations

from typing import Final

from openai.types.decision_create_params import (
    Question,
    QuestionQuestionParamChoice,
    QuestionQuestionParamPredicate,
)

from gbharness.agent.policy import ACTION_DESCRIPTIONS

ACTION_QUESTION: Final[QuestionQuestionParamChoice] = {
    "type": "choice",
    "name": "action",
    "instructions": (
        "The image is the current screen of Super Mario Bros. Choose the single best next "
        "input to make progress. Mario moves right toward a flagpole. Jump over gaps, pipes "
        "and enemies. Press Start on a title screen; wait during transitions or death."
    ),
    "choices": [
        {"value": key, "description": description}
        for key, description in ACTION_DESCRIPTIONS.items()
    ],
}

SCREEN_QUESTION: Final[QuestionQuestionParamChoice] = {
    "type": "choice",
    "name": "screen_kind",
    "instructions": "What kind of screen is showing?",
    "choices": [
        {"value": "title", "description": "A title or attract screen before play begins."},
        {"value": "menu", "description": "A menu with selectable options."},
        {"value": "gameplay", "description": "Active play in a level."},
        {"value": "cutscene", "description": "A transition, score tally, or level card."},
        {"value": "game_over", "description": "The character died or the game is over."},
    ],
}

WAITING_QUESTION: Final[QuestionQuestionParamPredicate] = {
    "type": "predicate",
    "name": "waiting_for_input",
    "instructions": (
        "Is the game idle and waiting for a button press, rather than playing an animation?"
    ),
}

QUESTIONS: Final[list[Question]] = [
    ACTION_QUESTION,
    SCREEN_QUESTION,
    WAITING_QUESTION,
]
