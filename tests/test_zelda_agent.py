from __future__ import annotations

from typing import cast

import numpy as np

from gbharness.agent.zelda import (
    bootstrap_four_swords,
    four_swords_gameplay_visible,
)
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession
from gbharness.vision.capture import FrameLatch, RawFrame


class FourPlayerRecordingPort:
    players = 4

    def __init__(self) -> None:
        self.held = [set[Button]() for _ in range(self.players)]

    def press_for(self, player: int, button: Button) -> None:
        self.held[player].add(button)

    def release_for(self, player: int, button: Button) -> None:
        self.held[player].discard(button)

    def pressed_for(self, player: int) -> frozenset[Button]:
        return frozenset(self.held[player])

    def advance(self, frames: int) -> bool:
        return True


class StaticFrames:
    def __init__(self, frame: RawFrame) -> None:
        self.frame = frame

    def require(self) -> RawFrame:
        return self.frame


def _composite(*, gameplay: bool, stage_select: bool = False) -> RawFrame:
    rgb = np.zeros((320, 480, 3), dtype=np.uint8)
    for x, y in ((0, 0), (240, 0), (0, 160), (240, 160)):
        if gameplay:
            rgb[y + 3 : y + 17, x + 120 : x + 130] = (255, 0, 0)
        if stage_select:
            rgb[y + 3 : y + 17, x + 3 : x + 20] = (255, 0, 0)
    packed = (
        ((rgb[:, :, 0].astype(np.uint16) >> 3) << 11)
        | ((rgb[:, :, 1].astype(np.uint16) >> 2) << 5)
        | (rgb[:, :, 2].astype(np.uint16) >> 3)
    )
    return RawFrame(
        data=packed.astype("<u2").tobytes(),
        width=480,
        height=320,
        pitch=960,
        index=0,
    )


def test_four_swords_gameplay_requires_heart_hud_on_every_view() -> None:
    assert not four_swords_gameplay_visible(_composite(gameplay=False), 4)
    assert not four_swords_gameplay_visible(_composite(gameplay=False, stage_select=True), 4)
    assert four_swords_gameplay_visible(_composite(gameplay=True), 4)


def test_bootstrap_is_deterministic_and_stops_at_visible_gameplay() -> None:
    port = FourPlayerRecordingPort()
    session = LinkedSession(port, max_step_frames=1800)
    frames = cast(FrameLatch, StaticFrames(_composite(gameplay=True)))

    result = bootstrap_four_swords(session, frames)

    assert result.frames_advanced == 3042
    assert result.handshake_confirms == 1
    assert session.frame == 3042
    assert all(not buttons for buttons in port.held)
