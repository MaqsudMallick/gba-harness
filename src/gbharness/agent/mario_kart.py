"""Small deterministic setup skills for Mario Kart: Super Circuit demos."""

from __future__ import annotations

from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession


def open_linked_lobby(session: LinkedSession) -> None:
    """Navigate every cartridge to the verified Multi-Pak 2-4 player lobby.

    This is setup automation, not a racing policy. It reproduces the game's documented
    multiplayer flow so a live demo starts at the shared Mario GP / VS / Battle menu.
    """
    everyone = range(session.port.players)

    def tap_all(button: Button, settle: int) -> None:
        session.step({player: (button,) for player in everyone}, 3)
        session.step({player: () for player in everyone}, settle)

    session.step({}, 900)
    tap_all(Button.DOWN, 12)  # SINGLE PLAYER -> MULTIPLAYER
    tap_all(Button.A, 180)
    tap_all(Button.A, 300)  # MULTI-PAK LINK
    tap_all(Button.UP, 12)  # cable confirmation: NO -> YES
    tap_all(Button.A, 1200)
