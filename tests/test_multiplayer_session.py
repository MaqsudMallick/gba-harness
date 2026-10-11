from __future__ import annotations

import json

import pytest

from gbharness.backends import LibretroMultiplayerPort
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession


def test_linked_step_updates_players_before_one_shared_advance(
    test_rom_bytes, multiplayer_core
) -> None:
    with LibretroMultiplayerPort(
        test_rom_bytes, players=2, core=multiplayer_core
    ) as port:
        port.start_linked_session()
        session = LinkedSession(port)
        before = port.frames_run
        result = session.step({0: ["a", "right"], 1: ["b", "left"]}, 5)
        assert port.frames_run == before + 5
        assert result.held_by_player == (
            (Button.RIGHT, Button.A),
            (Button.LEFT, Button.B),
        )
        assert json.loads(json.dumps(result.to_dict()))["total_frames"] == 5


def test_partial_step_preserves_an_omitted_player(test_rom_bytes, multiplayer_core) -> None:
    with LibretroMultiplayerPort(test_rom_bytes, core=multiplayer_core) as port:
        session = LinkedSession(port)
        session.step({1: "a"}, 1)
        session.step({0: "right"}, 1)
        assert port.pressed_for(1) == {Button.A}


def test_opposed_directions_are_rejected_before_any_input(
    test_rom_bytes, multiplayer_core
) -> None:
    with LibretroMultiplayerPort(test_rom_bytes, core=multiplayer_core) as port:
        session = LinkedSession(port)
        with pytest.raises(ValueError, match="opposed"):
            session.step({1: ["up", "down"]}, 2)
        assert port.pressed_for(1) == frozenset()
