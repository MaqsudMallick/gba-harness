"""Run a real 2-4 player linked GBA session in one split-screen window.

Designed for Mario Kart: Super Circuit's multiplayer mode, but the backend is generic to
GBA games that use the link cable. ROM files remain local and gitignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from gbharness.agent.mario_kart import (
    MarioKartVisionAgent,
    open_linked_lobby,
    start_quick_race,
)
from gbharness.agent.multiplayer import AutonomousPlayer, PlayerDecision
from gbharness.backends import LibretroMultiplayerPort, find_multiplayer_core, validate_rom
from gbharness.buttons import Button, canon_sort
from gbharness.multiplayer import LinkedSession
from gbharness.timing import FRAME_RATE
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_png, to_rgb
from gbharness.vision.multiplayer import split_player_frames

P1_KEYS: dict[int, Button] = {
    pygame.K_UP: Button.UP,
    pygame.K_DOWN: Button.DOWN,
    pygame.K_LEFT: Button.LEFT,
    pygame.K_RIGHT: Button.RIGHT,
    pygame.K_n: Button.A,
    pygame.K_m: Button.B,
    pygame.K_k: Button.L,
    pygame.K_l: Button.R,
    pygame.K_RETURN: Button.START,
    pygame.K_RSHIFT: Button.SELECT,
}

P2_KEYS: dict[int, Button] = {
    pygame.K_w: Button.UP,
    pygame.K_s: Button.DOWN,
    pygame.K_a: Button.LEFT,
    pygame.K_d: Button.RIGHT,
    pygame.K_f: Button.A,
    pygame.K_g: Button.B,
    pygame.K_q: Button.L,
    pygame.K_e: Button.R,
    pygame.K_SPACE: Button.START,
    pygame.K_LSHIFT: Button.SELECT,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--rom", type=Path, required=True, help="path to a user-owned .gba ROM")
    result.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    result.add_argument("--core", type=Path, help="mGBA Splitscreen libretro core")
    result.add_argument("--scale", type=int, default=2)
    result.add_argument("--max-frames", type=int)
    result.add_argument("--headless", action="store_true")
    result.add_argument("--screenshot", type=Path)
    result.add_argument(
        "--mario-kart-lobby",
        action="store_true",
        help="automatically reach the linked 2P/3P/4P Mario Kart mode menu",
    )
    result.add_argument(
        "--quick-race",
        action="store_true",
        help="choose default racers and Mushroom Cup (implies --mario-kart-lobby)",
    )
    result.add_argument(
        "--agent",
        choices=("mario-kart",),
        help="local visual game adapter controlling one linked player",
    )
    result.add_argument(
        "--agent-player",
        type=int,
        default=2,
        help="one-based player controlled by --agent (default: 2)",
    )
    return result


def _save_screenshot(latch: FrameLatch, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_png(latch.require()))
    print(f"screenshot: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.scale < 1:
        raise SystemExit("--scale must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    if not 1 <= args.agent_player <= args.players:
        raise SystemExit(f"--agent-player must be between 1 and {args.players}")
    if not args.rom.is_file():
        raise SystemExit(f"ROM does not exist: {args.rom}")

    rom = args.rom.read_bytes()
    validate_rom(rom)
    core = find_multiplayer_core(args.core)
    latch = FrameLatch()

    port = LibretroMultiplayerPort(
        rom,
        players=args.players,
        core=core,
        frame_sink=latch.sink,
    )
    with port:
        port.start_linked_session()
        linked = LinkedSession(port, max_step_frames=1800)
        if args.mario_kart_lobby or args.quick_race:
            open_linked_lobby(linked)
        if args.quick_race:
            start_quick_race(
                linked,
                lambda: to_rgb(split_player_frames(latch.require(), args.players)[0]),
            )
        autonomous = (
            AutonomousPlayer(
                player=args.agent_player - 1,
                players=args.players,
                agent=MarioKartVisionAgent(),
            )
            if args.agent == "mario-kart"
            else None
        )
        if args.headless:
            for _ in range(args.max_frames or 600):
                headless_inputs: dict[int, object] = {}
                if autonomous is not None:
                    decision = autonomous.update(latch.require(), linked.frame)
                    headless_inputs[autonomous.player] = decision.buttons
                if linked.step(headless_inputs, 1).emulation_ended:
                    break
            _save_screenshot(latch, args.screenshot)
            print(
                f"linked session: {args.players} players, {port.frames_run} frames, "
                f"controller ports polled={sorted(port.queried_ports)}"
            )
            if autonomous is not None:
                print(
                    f"agent: P{autonomous.player + 1} {autonomous.agent.name}, "
                    f"decisions={autonomous.decisions}, "
                    f"scene={autonomous.last_decision.scene}, "
                    f"buttons={'+'.join(
                        button.value for button in canon_sort(
                            autonomous.last_decision.buttons
                        )
                    ) or 'none'}"
                )
                if autonomous.last_decision.note:
                    print(f"agent note: {autonomous.last_decision.note}")
            return 0

        pygame.init()
        frame = latch.require()
        native_size = (frame.width, frame.height)
        game_size = (frame.width * args.scale, frame.height * args.scale)
        panel_height = 88
        screen = pygame.display.set_mode((game_size[0], game_size[1] + panel_height))
        pygame.display.set_caption(
            f"gbharness multiplayer — {args.players} linked players — {args.rom.name}"
        )
        font = pygame.font.Font(None, 21)
        clock = pygame.time.Clock()
        keymaps = (P1_KEYS, P2_KEYS)
        # key -> (player, button, earliest release frame, requested release frame)
        # Keeping a tap alive for two emulated frames fixes fast keydown+keyup pairs that
        # otherwise arrive in the same Pygame event batch and never reach the cartridge.
        held_keys: dict[int, tuple[int, Button, int, int | None]] = {}
        running = True
        displayed = 0
        agent_decision = PlayerDecision(frozenset(), "disabled")
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        for player, keymap in enumerate(keymaps[: args.players]):
                            if autonomous is not None and player == autonomous.player:
                                continue
                            button = keymap.get(event.key)
                            if button is not None and event.key not in held_keys:
                                held_keys[event.key] = (
                                    player,
                                    button,
                                    displayed + 2,
                                    None,
                                )
                    elif event.type == pygame.KEYUP and event.key in held_keys:
                        player, button, earliest, _ = held_keys[event.key]
                        held_keys[event.key] = (
                            player,
                            button,
                            earliest,
                            max(displayed, earliest),
                        )
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        held_keys.clear()

                expired = [
                    key
                    for key, (_, _, _, release_at) in held_keys.items()
                    if release_at is not None and displayed >= release_at
                ]
                for key in expired:
                    held_keys.pop(key)

                inputs: dict[int, object] = {
                    player: tuple(
                        button
                        for held_player, button, _, _ in held_keys.values()
                        if held_player == player
                    )
                    for player in range(min(args.players, len(keymaps)))
                    if autonomous is None or player != autonomous.player
                }
                if autonomous is not None:
                    agent_decision = autonomous.update(latch.require(), linked.frame)
                    inputs[autonomous.player] = agent_decision.buttons

                if not running or linked.step(inputs, 1).emulation_ended:
                    break
                frame = latch.require()
                if (frame.width, frame.height) != native_size:
                    raise RuntimeError("live layout changed dimensions unexpectedly")
                rgb = to_rgb(frame)
                native = pygame.image.frombytes(rgb.tobytes(), native_size, "RGB")
                scaled = pygame.transform.scale(native, game_size)
                screen.fill((13, 15, 22))
                screen.blit(scaled, (0, 0))
                p1 = font.render(
                    "P1 YOU  arrows · hold N=accelerate · M=brake · Enter=Start",
                    True,
                    (255, 112, 112),
                )
                held_p1 = "+".join(
                    button.value.upper() for button in canon_sort(port.pressed_for(0))
                )
                input_status = font.render(
                    f"P1 INPUT  {held_p1 or 'none'}",
                    True,
                    (255, 190, 190),
                )
                if autonomous is not None:
                    agent_buttons = "+".join(
                        button.value.upper()
                        for button in canon_sort(agent_decision.buttons)
                    )
                    p2_label = (
                        f"P{autonomous.player + 1} AGENT  {agent_decision.scene} · "
                        f"{agent_buttons or 'none'}"
                    )
                else:
                    p2_label = (
                        "P2 HUMAN  WASD · hold F=accelerate · G=brake · Space=Start"
                    )
                p2 = font.render(
                    p2_label,
                    True,
                    (112, 174, 255),
                )
                screen.blit(p1, (12, game_size[1] + 8))
                screen.blit(input_status, (12, game_size[1] + 34))
                screen.blit(p2, (12, game_size[1] + 60))
                pygame.display.flip()
                displayed += 1
                if args.max_frames is not None and displayed >= args.max_frames:
                    running = False
                clock.tick(round(FRAME_RATE))
        finally:
            _save_screenshot(latch, args.screenshot)
            pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
