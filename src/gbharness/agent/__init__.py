"""The decision loop. Optional extra: `gbharness[agent]`."""

from gbharness.agent.contracts import (
    ActionOption,
    Decision,
    DecisionProvider,
    DecisionRequest,
    GamePolicy,
)
from gbharness.agent.decide import Decider, Verdict
from gbharness.agent.mario import MarioAutopilot, MarioVisionController, observe_mario
from gbharness.agent.mario_kart import (
    MarioKartObservation,
    MarioKartVisionAgent,
    observe_mario_kart,
    open_linked_lobby,
    start_quick_race,
)
from gbharness.agent.multiplayer import (
    AutonomousPlayer,
    PlayerDecision,
    VisualPlayerAgent,
)
from gbharness.agent.openai_questions import QUESTIONS
from gbharness.agent.policy import ACTION_DESCRIPTIONS, ACTIONS, resolve
from gbharness.agent.runner import AgentRunner, EpisodeResult, RunnerConfig, TurnRecord
from gbharness.agent.vertex import VertexDecisionProvider

__all__ = [
    "ACTIONS",
    "ACTION_DESCRIPTIONS",
    "QUESTIONS",
    "ActionOption",
    "Decider",
    "Decision",
    "DecisionProvider",
    "DecisionRequest",
    "EpisodeResult",
    "GamePolicy",
    "MarioAutopilot",
    "MarioKartObservation",
    "MarioKartVisionAgent",
    "MarioVisionController",
    "AgentRunner",
    "RunnerConfig",
    "TurnRecord",
    "Verdict",
    "VertexDecisionProvider",
    "AutonomousPlayer",
    "PlayerDecision",
    "VisualPlayerAgent",
    "observe_mario",
    "observe_mario_kart",
    "open_linked_lobby",
    "start_quick_race",
    "resolve",
]
