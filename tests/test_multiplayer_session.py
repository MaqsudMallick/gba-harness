from __future__ import annotations

import json

import pytest

from gbharness.actions import hold, sequence, tap, wait
from gbharness.backends import LibretroMultiplayerPort
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession


class RecordingLinkedPort:
    players = 2

    def __init__(self) -> None:
        self.held = [set[Button](), set[Button]()]
        self.frame = 0
        self.snapshots: list[tuple[int, tuple[frozenset[Button], ...]]] = []

    def press_for(self, player: int, button: Button) -> None:
        self.held[player].add(button)

    def release_for(self, player: int, button: Button) -> None:
        self.held[player].discard(button)

    def pressed_for(self, player: int) -> frozenset[Button]:
        return frozenset(self.held[player])

    def advance(self, frames: int) -> bool:
        self.snapshots.append((self.frame, tuple(frozenset(buttons) for buttons in self.held)))
        self.frame += frames
        return True


def test_linked_step_updates_players_before_one_shared_advance(
    test_rom_bytes, multiplayer_core
) -> None:
    with LibretroMultiplayerPort(test_rom_bytes, players=2, core=multiplayer_core) as port:
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


def test_opposed_directions_are_rejected_before_any_input(test_rom_bytes, multiplayer_core) -> None:
    with LibretroMultiplayerPort(test_rom_bytes, core=multiplayer_core) as port:
        session = LinkedSession(port)
        with pytest.raises(ValueError, match="opposed"):
            session.step({1: ["up", "down"]}, 2)
        assert port.pressed_for(1) == frozenset()


def test_linked_macros_share_phase_boundaries_and_release_every_player() -> None:
    port = RecordingLinkedPort()
    session = LinkedSession(port)
    result = session.submit(
        {
            0: tap("a", hold_frames=2, gap_frames=4),
            1: sequence(hold("right", 3), wait(5)),
        }
    )

    assert result.frames_advanced == 8
    assert port.snapshots[:4] == [
        (0, (frozenset({Button.A}), frozenset({Button.RIGHT}))),
        (2, (frozenset(), frozenset({Button.RIGHT}))),
        (3, (frozenset(), frozenset())),
        (6, (frozenset(), frozenset())),
    ]
    assert port.pressed_for(0) == frozenset()
    assert port.pressed_for(1) == frozenset()


def test_linked_macros_reject_open_ended_actions() -> None:
    from gbharness.actions import press

    with pytest.raises(TypeError, match="finite"):
        LinkedSession(RecordingLinkedPort()).submit({0: press("a")})
