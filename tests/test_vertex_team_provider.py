from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from gbharness.agent.contracts import ActionOption
from gbharness.agent.team import TeamDecisionRequest
from gbharness.agent.vertex import VertexDecisionProvider
from gbharness.agent.vertex_team import VertexTeamDecisionProvider


class FakeModels:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.kwargs: dict[str, Any] = {}

    def generate_content(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(
            parsed=self.payload,
            text=None,
            usage_metadata=SimpleNamespace(prompt_token_count=50, candidates_token_count=10),
        )


def provider(model: str, models: FakeModels) -> VertexDecisionProvider:
    return VertexDecisionProvider(
        project="test",
        location="global",
        model=model,
        client=SimpleNamespace(models=models),
    )


def test_vertex_team_provider_rotates_captain_and_parses_joint_actions() -> None:
    first = FakeModels(
        {
            "scene": "switch room",
            "objective": "occupy both switches",
            "human_instruction": "P1 move left and hold.",
            "player_2_action": "move_left",
            "player_3_action": "interact",
            "player_4_action": "move_right",
            "confidence": 0.9,
            "terminal": False,
            "note": "Split across the room.",
        }
    )
    second = FakeModels(first.payload)
    team = VertexTeamDecisionProvider(
        (provider("gemini-first", first), provider("gemini-second", second))
    )
    request = TeamDecisionRequest(
        goal="solve",
        state="turn state",
        image_png=b"composite",
        actions=(
            ActionOption("move_left", "left"),
            ActionOption("move_right", "right"),
            ActionOption("interact", "interact"),
        ),
        ai_players=(1, 2, 3),
        turn=2,
    )

    decision = team.decide_team(request)

    assert decision.captain == "vertex:gemini-second"
    assert decision.human_instruction == "P1 move left and hold."
    assert [decision.decisions[player].action for player in request.ai_players] == [
        "move_left",
        "interact",
        "move_right",
    ]
    assert decision.input_tokens == 50
    assert decision.output_tokens == 10
    assert second.kwargs["model"] == "gemini-second"
