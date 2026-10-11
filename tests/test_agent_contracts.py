"""Provider-neutral game-agent contracts."""

from __future__ import annotations

import pytest

from gbharness.actions import wait
from gbharness.agent.contracts import GamePolicy


def test_game_policy_exposes_ordered_options_and_resolves_actions() -> None:
    action = wait(4)
    policy = GamePolicy(
        name="test",
        goal="win",
        actions={"idle": action},
        descriptions={"idle": "Do nothing briefly."},
    )
    assert policy.options[0].key == "idle"
    assert policy.resolve("idle") is action


def test_game_policy_rejects_mismatched_descriptions() -> None:
    with pytest.raises(ValueError, match="descriptions differ"):
        GamePolicy(
            name="bad",
            goal="win",
            actions={"idle": wait(4)},
            descriptions={"other": "Wrong key."},
        )

