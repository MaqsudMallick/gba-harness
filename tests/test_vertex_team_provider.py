from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace
from typing import Any

from PIL import Image

import gbharness.agent.vertex_team as vertex_team
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


class FlakyModels(FakeModels):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(payload)
        self.calls = 0

    def generate_content(self, **kwargs: Any) -> object:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary Vertex failure")
        return super().generate_content(**kwargs)


def provider(model: str, models: FakeModels) -> VertexDecisionProvider:
    return VertexDecisionProvider(
        project="test",
        location="global",
        model=model,
        client=SimpleNamespace(models=models),
    )


def blank_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (240, 160)).save(output, format="PNG")
    return output.getvalue()


def test_vertex_team_provider_retries_one_transient_failure(monkeypatch: Any) -> None:
    models = FlakyModels(
        {
            "scene": "switch room",
            "stage": "explore",
            "objective": "occupy the targets",
            "human_instruction": "P1 remains human-controlled.",
            "human_dialogue_plan": ["wait"],
            "p1_facing_action": "wait",
            "dialogue_visible": False,
            "player_2_plan": ["wait"],
            "player_3_plan": ["move_down"],
            "player_4_plan": ["move_down"],
            "confidence": 1.0,
            "terminal": False,
            "note": "Continue the current routes.",
        }
    )
    monkeypatch.setattr(vertex_team.time, "sleep", lambda _seconds: None)
    team = VertexTeamDecisionProvider((provider("gemini", models),))
    request = TeamDecisionRequest(
        goal="solve",
        state="turn state",
        image_png=b"composite",
        actions=(
            ActionOption("move_down", "down"),
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=1,
    )

    decision = team.decide_team(request)

    assert models.calls == 2
    assert decision.plans[1] == ("wait",)
    assert decision.plans[2] == ("move_down",)
    assert decision.plans[3] == ("move_down",)


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
    trace_input = json.loads(decision.trace_input_json)
    trace_output = json.loads(decision.trace_output_json)
    assert trace_input["system_instruction"]
    assert trace_input["user_prompt"].startswith("Goal: solve")
    assert trace_input["observation"]["sha256"]
    assert trace_output["provider_response"] == first.payload
    assert trace_output["effective_decision"]["decision_reason"] == ("Split across the room.")
    assert trace_output["effective_decision"]["player_plans"]["P2"] == [
        "move_left",
        "move_left",
    ]


def test_navigation_mode_uses_one_scout_and_does_not_skip_stages() -> None:
    models = FakeModels(
        {
            "scene": "colored tiles beyond the NPC",
            "stage": "reach_warps",
            "objective": "test one colored tile",
            "npc_actor_player": 2,
            "npc_facing_action": "face_right",
            "dialogue_visible": False,
            "player_2_plan": ["move_down", "interact"],
            "player_3_plan": ["move_left"],
            "player_4_plan": ["move_right"],
            "confidence": 0.9,
            "terminal": False,
            "note": "Use one scout.",
        }
    )
    team = VertexTeamDecisionProvider((provider("gemini", models),), navigation_only=True)
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
    assert decision.human_plan == ()
    assert decision.human_instruction == "P1 remains human-controlled."
    assert decision.plans[1] == ("move_down", "interact")
    assert decision.plans[2] == ("wait",)
    assert decision.plans[3] == ("wait",)
    assert models.kwargs["contents"][1].inline_data.data == b"composite"
    schema = models.kwargs["config"].response_json_schema
    assert "human_dialogue_plan" not in schema["properties"]
    assert "human_instruction" not in schema["properties"]
    assert schema["properties"]["npc_actor_player"]["enum"] == [2]

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
    assert models.kwargs["contents"][1].inline_data.data == b"composite"


def test_navigation_mode_waits_for_visible_dialogue_text() -> None:
    models = FakeModels(
        {
            "scene": "open dialogue box",
            "stage": "advance_dialogue",
            "objective": "finish NPC dialogue",
            "npc_actor_player": 2,
            "npc_facing_action": "face_right",
            "dialogue_visible": True,
            "player_2_plan": ["move_down"],
            "player_3_plan": ["move_left"],
            "player_4_plan": ["move_right"],
            "confidence": 0.9,
            "terminal": False,
            "note": "Wait for the page to draw.",
        }
    )
    team = VertexTeamDecisionProvider((provider("gemini", models),), navigation_only=True)
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
            ActionOption("advance_text", "advance text"),
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=1,
        stage="advance_dialogue",
    )

    decision = team.decide_team(request)

    assert decision.stage == "advance_dialogue"
    assert decision.human_plan == ()
    assert decision.plans[1] == (
        "advance_text",
        "wait",
        "wait",
        "wait",
        "wait",
        "wait",
    )
    assert decision.plans[2] == ("wait",)
    assert decision.plans[3] == ("wait",)


def test_post_portal_room_allows_visible_combat_actions() -> None:
    models = FakeModels(
        {
            "scene": "three lanes with visible enemies",
            "stage": "explore_room",
            "objective": "clear each lane",
            "room_map": "Four separate lanes contain visible targets.",
            "exploration_status": "target_confirmed",
            "npc_status": "all_visible_npcs_checked",
            "visible_npc_count": 2,
            "checked_npc_count": 2,
            "dialogue_visible": False,
            "human_plan": ["move_down"],
            "player_1_target": "P1 target in the left lane",
            "player_2_target": "P2 target in the second lane",
            "player_3_target": "P3 target in the third lane",
            "player_4_target": "P4 target in the right lane",
            "replan_trigger": "a target disappears or a Link stops moving",
            "player_2_plan": ["move_down", "attack_down"],
            "player_3_plan": ["attack_down"],
            "player_4_plan": ["move_down", "use_sword"],
            "confidence": 0.9,
            "terminal": False,
            "note": "Each AI Link clears its visible lane.",
        }
    )
    team = VertexTeamDecisionProvider((provider("gemini", models),), navigation_only=True)
    request = TeamDecisionRequest(
        goal="solve the next room",
        state="the portal transition is complete",
        image_png=b"composite",
        actions=(
            ActionOption("move_down", "down"),
            ActionOption("attack_down", "attack down"),
            ActionOption("use_sword", "use sword"),
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=5,
        stage="explore_room",
    )

    decision = team.decide_team(request)

    assert decision.stage == "explore_room"
    assert decision.human_instruction == "P1 target in the left lane"
    assert decision.human_plan == ("move_down",)
    assert decision.plans[1] == ("move_down", "attack_down")
    assert decision.plans[2] == ("attack_down",)
    assert decision.plans[3] == ("move_down", "use_sword")
    schema = models.kwargs["config"].response_json_schema
    assert schema["properties"]["stage"]["enum"] == ["explore_room"]
    assert "npc_actor_player" not in schema["properties"]
    assert "human_instruction" not in schema["properties"]
    assert "human_plan" in schema["properties"]
    assert "Do not return through the entrance portal" in models.kwargs["contents"][0]
    assert "use P2 as the scout" in models.kwargs["contents"][0]


def test_parallel_lanes_can_use_the_same_direction() -> None:
    models = FakeModels(
        {
            "scene": "four separate lanes",
            "stage": "explore_room",
            "objective": "reach one visible target in each lane",
            "room_map": "Each Link is in a separate vertical lane.",
            "exploration_status": "target_confirmed",
            "npc_status": "all_visible_npcs_checked",
            "visible_npc_count": 2,
            "checked_npc_count": 2,
            "dialogue_visible": False,
            "human_plan": ["move_down"],
            "player_1_target": "left lane target",
            "player_2_target": "second lane target",
            "player_3_target": "third lane target",
            "player_4_target": "right lane target",
            "player_2_plan": ["move_down"],
            "player_3_plan": ["move_down"],
            "player_4_plan": ["move_down"],
            "replan_trigger": "one target disappears",
            "confidence": 0.9,
            "terminal": False,
            "note": "Each Link moves inside its separate lane.",
        }
    )
    team = VertexTeamDecisionProvider((provider("gemini", models),), navigation_only=True)
    request = TeamDecisionRequest(
        goal="solve the visible room",
        state="four Links entered separate lanes",
        image_png=b"composite",
        actions=(ActionOption("move_down", "down"), ActionOption("wait", "wait")),
        ai_players=(1, 2, 3),
        turn=1,
        stage="explore_room",
    )

    decision = team.decide_team(request)

    assert decision.human_plan == ("move_down",)
    assert decision.plans == {
        1: ("move_down",),
        2: ("move_down",),
        3: ("move_down",),
    }


def test_explore_room_holds_followers_while_scouting() -> None:
    models = FakeModels(
        {
            "scene": "separate lanes and a solid wall",
            "stage": "explore_room",
            "objective": "inspect the reachable lane",
            "room_map": "P2 has a visible collectible above a solid wall.",
            "exploration_status": "scouting",
            "npc_status": "searching",
            "visible_npc_count": 2,
            "checked_npc_count": 0,
            "dialogue_visible": False,
            "human_plan": ["wait"],
            "player_1_target": "hold position",
            "player_2_target": "visible collectible",
            "player_3_target": "hold position",
            "player_4_target": "hold position",
            "player_2_plan": ["move_up"],
            "player_3_plan": ["move_down"],
            "player_4_plan": ["move_down"],
            "replan_trigger": "P2 reaches the collectible",
            "confidence": 0.9,
            "terminal": False,
            "note": "P2 scouts the lane.",
        }
    )
    team = VertexTeamDecisionProvider((provider("gemini", models),), navigation_only=True)
    request = TeamDecisionRequest(
        goal="solve the visible room",
        state="new room",
        image_png=b"composite",
        actions=(
            ActionOption("move_up", "up"),
            ActionOption("move_down", "down"),
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=1,
        player_images=(blank_png(), blank_png(), blank_png(), blank_png()),
        stage="explore_room",
    )

    decision = team.decide_team(request)

    assert decision.plans == {1: ("move_up",), 2: ("wait",), 3: ("wait",)}
    assert models.kwargs["contents"][1].inline_data.data == request.image_png
    assert "Attached observation: four-player composite view" in models.kwargs["contents"][0]


def test_explore_room_finishes_npc_dialogue_before_team_actions() -> None:
    models = FakeModels(
        {
            "scene": "P2 has an open NPC dialogue box",
            "stage": "explore_room",
            "objective": "finish all visible NPC dialogue",
            "room_map": "The right NPC is active. Two other NPCs remain unchecked.",
            "exploration_status": "target_confirmed",
            "npc_status": "dialogue_active",
            "visible_npc_count": 3,
            "checked_npc_count": 1,
            "dialogue_visible": True,
            "human_plan": ["move_down"],
            "player_1_target": "left switch",
            "player_2_target": "right NPC",
            "player_3_target": "centre switch",
            "player_4_target": "right switch",
            "player_2_plan": ["move_down"],
            "player_3_plan": ["move_down"],
            "player_4_plan": ["move_down"],
            "replan_trigger": "the dialogue box closes",
            "confidence": 0.9,
            "terminal": True,
            "note": "Finish the current NPC first.",
        }
    )
    team = VertexTeamDecisionProvider((provider("gemini", models),), navigation_only=True)
    request = TeamDecisionRequest(
        goal="solve the visible room",
        state="the team entered a new room",
        image_png=b"composite",
        actions=(
            ActionOption("move_down", "down"),
            ActionOption("advance_text", "advance text"),
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=1,
        stage="explore_room",
    )

    decision = team.decide_team(request)

    assert decision.human_plan == ("wait",)
    assert decision.plans == {
        1: ("advance_text", "wait", "wait", "wait", "wait", "wait"),
        2: ("wait",),
        3: ("wait",),
    }
    assert not any(item.terminal for item in decision.decisions.values())
    schema = models.kwargs["config"].response_json_schema
    assert "npc_status" in schema["required"]
    assert "visible_npc_count" in schema["required"]
    assert "checked_npc_count" in schema["required"]
    assert "dialogue_visible" in schema["required"]
    assert "including pink figures on blue tiles" in models.kwargs["contents"][0]


def test_explore_room_blocks_attacks_until_npc_checks_finish() -> None:
    models = FakeModels(
        {
            "scene": "P2 sees an unchecked pink NPC",
            "stage": "explore_room",
            "objective": "talk to the visible NPC",
            "room_map": "One pink NPC remains unchecked on the upper tile.",
            "exploration_status": "scouting",
            "npc_status": "searching",
            "visible_npc_count": 1,
            "checked_npc_count": 0,
            "dialogue_visible": False,
            "human_plan": ["wait"],
            "player_1_target": "hold position",
            "player_2_target": "upper NPC",
            "player_3_target": "hold position",
            "player_4_target": "hold position",
            "player_2_plan": ["attack_up", "use_sword"],
            "player_3_plan": ["attack_up"],
            "player_4_plan": ["use_sword"],
            "replan_trigger": "P2 reaches the NPC",
            "confidence": 0.9,
            "terminal": False,
            "note": "The NPC check is not complete.",
        }
    )
    team = VertexTeamDecisionProvider((provider("gemini", models),), navigation_only=True)
    request = TeamDecisionRequest(
        goal="solve the visible room",
        state="new room",
        image_png=b"composite",
        actions=(
            ActionOption("attack_up", "attack up"),
            ActionOption("use_sword", "use sword"),
            ActionOption("wait", "wait"),
        ),
        ai_players=(1, 2, 3),
        turn=1,
        stage="explore_room",
    )

    decision = team.decide_team(request)

    assert decision.plans == {1: ("wait", "wait"), 2: ("wait",), 3: ("wait",)}
