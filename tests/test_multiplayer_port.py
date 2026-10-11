from __future__ import annotations

import pytest

from gbharness.backends import LibretroMultiplayerPort
from gbharness.buttons import Button
from gbharness.vision.capture import FrameLatch
from gbharness.vision.multiplayer import split_player_frames


def test_multiplayer_core_polls_each_controller_port(test_rom_bytes, multiplayer_core) -> None:
    latch = FrameLatch()
    with LibretroMultiplayerPort(
        test_rom_bytes,
        players=2,
        core=multiplayer_core,
        frame_sink=latch.sink,
    ) as port:
        port.start_linked_session()
        port.press_for(1, Button.RIGHT)
        port.advance(3)
        assert port.pressed_for(0) == frozenset()
        assert port.pressed_for(1) == {Button.RIGHT}
        assert port.queried_ports >= {0, 1}
        assert len(split_player_frames(latch.require(), 2)) == 2


def test_multiplayer_controller_indices_are_checked(test_rom_bytes, multiplayer_core) -> None:
    with (
        LibretroMultiplayerPort(test_rom_bytes, core=multiplayer_core) as port,
        pytest.raises(IndexError, match="outside"),
    ):
        port.press_for(2, Button.A)


@pytest.mark.parametrize("players", [0, 1, 5])
def test_multiplayer_requires_two_to_four_players(
    test_rom_bytes, multiplayer_core, players: int
) -> None:
    with pytest.raises(ValueError, match="2-4"):
        LibretroMultiplayerPort(test_rom_bytes, players=players, core=multiplayer_core)
