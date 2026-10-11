"""The decision loop. Optional extra: `gbharness[agent]`."""

from gbharness.agent.contracts import (
    ActionOption,
    Decision,
    DecisionProvider,
    DecisionRequest,
    GamePolicy,
)
from gbharness.agent.decide import Decider, Verdict
from gbharness.agent.human import HumanActionSource, HumanDecisionProvider
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
from gbharness.agent.team import (
    PlayerProposal,
    TeamDecision,
    TeamDecisionProvider,
    TeamDecisionRequest,
)
from gbharness.agent.vertex import VertexDecisionProvider
from gbharness.agent.vertex_team import (
    HierarchicalVertexTeamDecisionProvider,
    VertexTeamDecisionProvider,
)
from gbharness.agent.zelda import (
    FOUR_SWORDS_POLICY,
    FOUR_SWORDS_ROLES,
    ZELDA_ACTION_DESCRIPTIONS,
    ZELDA_ACTIONS,
    FirstTrainingPuzzleResult,
    FourSwordsBootstrapResult,
    FourSwordsVisualTargetNavigator,
    bootstrap_four_swords,
    first_training_reward_visible,
    four_swords_dialogue_visible,
    four_swords_gameplay_visible,
    four_swords_probe_target_reached,
    four_swords_visual_target_plan,
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
    "FourSwordsVisualTargetNavigator",
    "GamePolicy",
    "HumanActionSource",
    "HumanDecisionProvider",
    "HierarchicalVertexTeamDecisionProvider",
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
    "TeamDecision",
    "TeamDecisionProvider",
    "TeamDecisionRequest",
    "Verdict",
    "VertexDecisionProvider",
    "VertexTeamDecisionProvider",
    "AutonomousPlayer",
    "PlayerDecision",
    "PlayerAssignment",
    "PlayerProposal",
    "PlayerTurnRecord",
    "VisualPlayerAgent",
    "ZELDA_ACTIONS",
    "ZELDA_ACTION_DESCRIPTIONS",
    "bootstrap_four_swords",
    "first_training_reward_visible",
    "four_swords_dialogue_visible",
    "four_swords_gameplay_visible",
    "four_swords_probe_target_reached",
    "four_swords_visual_target_plan",
    "observe_mario",
    "observe_mario_kart",
    "open_linked_lobby",
    "start_quick_race",
    "solve_first_training_puzzle",
    "resolve",
]
