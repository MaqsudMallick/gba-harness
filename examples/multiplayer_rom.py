"""Run a real 2-4 player linked GBA session in one split-screen window.

Designed for Mario Kart: Super Circuit's multiplayer mode, but the backend is generic to
GBA games that use the link cable. ROM files remain local and gitignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from gbharness.agent.mario_kart import open_linked_lobby
from gbharness.backends import LibretroMultiplayerPort, find_multiplayer_core, validate_rom
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession
from gbharness.timing import FRAME_RATE
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_png, to_rgb

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
    if not args.rom.is_file():
        raise SystemExit(f"ROM does not exist: {args.rom}")

    rom = args.rom.read_bytes()
    validate_rom(rom)
    core = find_multiplayer_core(args.core)
    latch = FrameLatch()

    with LibretroMultiplayerPort(
        rom,
        players=args.players,
        core=core,
        frame_sink=latch.sink,
    ) as port:
        port.start_linked_session()
        if args.mario_kart_lobby:
            open_linked_lobby(LinkedSession(port, max_step_frames=1800))
        if args.headless:
            port.advance(args.max_frames or 600)
            _save_screenshot(latch, args.screenshot)
            print(
                f"linked session: {args.players} players, {port.frames_run} frames, "
                f"controller ports polled={sorted(port.queried_ports)}"
            )
            return 0

        pygame.init()
        frame = latch.require()
        native_size = (frame.width, frame.height)
        game_size = (frame.width * args.scale, frame.height * args.scale)
        panel_height = 62
        screen = pygame.display.set_mode((game_size[0], game_size[1] + panel_height))
        pygame.display.set_caption(
            f"gbharness multiplayer — {args.players} linked players — {args.rom.name}"
        )
        font = pygame.font.Font(None, 21)
        clock = pygame.time.Clock()
        keymaps = (P1_KEYS, P2_KEYS)
        held_keys: dict[int, tuple[int, Button]] = {}
        running = True
        displayed = 0
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        for player, keymap in enumerate(keymaps[: args.players]):
                            button = keymap.get(event.key)
                            if button is not None and event.key not in held_keys:
                                held_keys[event.key] = (player, button)
                                port.press_for(player, button)
                    elif event.type == pygame.KEYUP and event.key in held_keys:
                        player, button = held_keys.pop(event.key)
                        port.release_for(player, button)
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        for player, button in held_keys.values():
                            port.release_for(player, button)
                        held_keys.clear()

                if not running or not port.advance(1):
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
                    "P1  arrows · N=A · M=B · K/L=shoulders · Enter=Start",
                    True,
                    (255, 112, 112),
                )
                p2 = font.render(
                    "P2  WASD · F=A · G=B · Q/E=shoulders · Space=Start",
                    True,
                    (112, 174, 255),
                )
                screen.blit(p1, (12, game_size[1] + 8))
                screen.blit(p2, (12, game_size[1] + 34))
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
