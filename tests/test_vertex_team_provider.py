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
            "stage": "explore",
            "objective": "occupy both switches",
            "human_instruction": "P1 move left and hold.",
            "human_dialogue_plan": ["wait"],
            "p1_facing_action": "wait",
            "dialogue_visible": False,
            "player_2_plan": ["move_left", "move_left"],
            "player_3_plan": ["interact", "wait"],
            "player_4_plan": ["move_right", "move_right"],
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
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=2,
    )

    decision = team.decide_team(request)

    assert decision.captain == "vertex:gemini-second"
    assert decision.human_instruction == "P1 move left and hold."
    assert decision.stage == "explore"
    assert decision.human_plan == ("wait",)
    assert decision.plans[1] == ("move_left", "move_left")
    assert [decision.decisions[player].action for player in request.ai_players] == [
        "move_left",
        "interact",
        "move_right",
    ]
    assert decision.input_tokens == 50
    assert decision.output_tokens == 10
    assert second.kwargs["model"] == "gemini-second"


def test_navigation_mode_uses_one_scout_and_does_not_skip_stages() -> None:
    models = FakeModels(
        {
            "scene": "colored tiles beyond the NPC",
            "stage": "reach_warps",
            "objective": "test one colored tile",
            "human_instruction": "P1 wait.",
            "human_dialogue_plan": ["move_down"],
            "p1_facing_action": "face_right",
            "dialogue_visible": False,
            "player_2_plan": ["move_down", "interact"],
            "player_3_plan": ["move_left"],
            "player_4_plan": ["move_right"],
            "confidence": 0.9,
            "terminal": False,
            "note": "Use one scout.",
        }
    )
    team = VertexTeamDecisionProvider(
        (provider("gemini", models),), navigation_only=True
    )
    request = TeamDecisionRequest(
        goal="discover the target",
        state="dialogue just ended",
        image_png=b"composite",
        actions=(
            ActionOption("move_left", "left"),
            ActionOption("move_right", "right"),
            ActionOption("move_down", "down"),
            ActionOption("face_right", "face right"),
            ActionOption("interact", "interact"),
            ActionOption("attack_down", "attack"),
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=1,
        player_images=(b"p1", b"p2", b"p3", b"p4"),
        stage="advance_dialogue",
    )

    decision = team.decide_team(request)

    assert decision.stage == "probe_target"
    assert decision.human_plan == ("wait",)
    assert decision.plans[1] == ("move_down", "interact")
    assert decision.plans[2] == ("wait",)
    assert decision.plans[3] == ("wait",)
    assert models.kwargs["contents"][1].inline_data.data == b"p1"

    models.payload["stage"] = "probe_target"
    probe_request = TeamDecisionRequest(
        goal=request.goal,
        state=request.state,
        image_png=request.image_png,
        actions=request.actions,
        ai_players=request.ai_players,
        turn=2,
        player_images=request.player_images,
        stage="probe_target",
    )
    team.decide_team(probe_request)
    assert models.kwargs["contents"][1].inline_data.data == b"p2"


def test_navigation_mode_waits_for_visible_dialogue_text() -> None:
    models = FakeModels(
        {
            "scene": "open dialogue box",
            "stage": "advance_dialogue",
            "objective": "finish NPC dialogue",
            "human_instruction": "P1 advances the dialogue.",
            "human_dialogue_plan": ["interact"],
            "p1_facing_action": "face_right",
            "dialogue_visible": True,
            "player_2_plan": ["move_down"],
            "player_3_plan": ["move_left"],
            "player_4_plan": ["move_right"],
            "confidence": 0.9,
            "terminal": False,
            "note": "Wait for the page to draw.",
        }
    )
    team = VertexTeamDecisionProvider(
        (provider("gemini", models),), navigation_only=True
    )
    request = TeamDecisionRequest(
        goal="finish the dialogue",
        state="dialogue is open",
        image_png=b"composite",
        actions=(
            ActionOption("move_left", "left"),
            ActionOption("move_right", "right"),
            ActionOption("move_down", "down"),
            ActionOption("face_right", "face right"),
            ActionOption("interact", "interact"),
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=1,
        stage="advance_dialogue",
    )

    decision = team.decide_team(request)

    assert decision.stage == "advance_dialogue"
    assert decision.human_plan == (
        "interact",
        "wait",
        "wait",
        "wait",
        "wait",
        "wait",
    )
    assert all(plan == ("wait",) for plan in decision.plans.values())
