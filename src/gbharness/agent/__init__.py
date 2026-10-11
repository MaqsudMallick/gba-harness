"""The decision loop. Optional extra: `gbharness[agent]`."""

from gbharness.agent.contracts import (
    ActionOption,
    Decision,
    DecisionProvider,
    DecisionRequest,
    GamePolicy,
)
from gbharness.agent.decide import Decider, Verdict
from gbharness.agent.linked_runner import (
    LinkedAgentRunner,
    LinkedEpisodeResult,
    LinkedRunnerConfig,
    LinkedTurnRecord,
    PlayerAssignment,
    PlayerTurnRecord,
)
from gbharness.agent.mario import MarioAutopilot, MarioVisionController, observe_mario
from gbharness.agent.mario_kart import (
    MarioKartObservation,
    MarioKartVisionAgent,
    observe_mario_kart,
    open_linked_lobby,
    start_quick_race,
)
from gbharness.agent.mario_kart_minimap import (
    PEACH_CIRCUIT,
    MarioKartMinimapNavigator,
    MinimapObservation,
    TrackProfile,
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
from gbharness.agent.zelda import (
    FOUR_SWORDS_POLICY,
    FOUR_SWORDS_ROLES,
    ZELDA_ACTION_DESCRIPTIONS,
    ZELDA_ACTIONS,
    FirstTrainingPuzzleResult,
    FourSwordsBootstrapResult,
    bootstrap_four_swords,
    first_training_reward_visible,
    four_swords_gameplay_visible,
    solve_first_training_puzzle,
)

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
    "FOUR_SWORDS_POLICY",
    "FOUR_SWORDS_ROLES",
    "FirstTrainingPuzzleResult",
    "FourSwordsBootstrapResult",
    "GamePolicy",
    "LinkedAgentRunner",
    "LinkedEpisodeResult",
    "LinkedRunnerConfig",
    "LinkedTurnRecord",
    "MarioAutopilot",
    "MarioKartObservation",
    "MarioKartMinimapNavigator",
    "MarioKartVisionAgent",
    "MarioVisionController",
    "MinimapObservation",
    "PEACH_CIRCUIT",
    "AgentRunner",
    "RunnerConfig",
    "TurnRecord",
    "TrackProfile",
    "Verdict",
    "VertexDecisionProvider",
    "AutonomousPlayer",
    "PlayerDecision",
    "PlayerAssignment",
    "PlayerTurnRecord",
    "VisualPlayerAgent",
    "ZELDA_ACTIONS",
    "ZELDA_ACTION_DESCRIPTIONS",
    "bootstrap_four_swords",
    "first_training_reward_visible",
    "four_swords_gameplay_visible",
    "observe_mario",
    "observe_mario_kart",
    "open_linked_lobby",
    "start_quick_race",
    "solve_first_training_puzzle",
    "resolve",
]
