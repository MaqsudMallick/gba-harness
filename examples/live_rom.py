"""Run the configured GBA ROM in a live, keyboard-controlled window.

This is a frontend for the same ``LibretroInputPort`` used by agents and tests, not a
separate emulator process. What appears in the window is therefore exactly what an agent
observes, and keyboard input travels through the same ten-button port contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from gbharness.backends import LibretroInputPort, find_core, validate_rom
from gbharness.buttons import Button
from gbharness.config import load_settings
from gbharness.timing import FRAME_RATE
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_rgb

KEY_BINDINGS: dict[int, Button] = {
    pygame.K_UP: Button.UP,
    pygame.K_DOWN: Button.DOWN,
    pygame.K_LEFT: Button.LEFT,
    pygame.K_RIGHT: Button.RIGHT,
    pygame.K_z: Button.A,
    pygame.K_x: Button.B,
    pygame.K_a: Button.L,
    pygame.K_s: Button.R,
    pygame.K_RETURN: Button.START,
    pygame.K_RSHIFT: Button.SELECT,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--rom",
        type=Path,
        help="path to a .gba file; defaults to GBHARNESS_ROM",
    )
    result.add_argument(
        "--scale",
        type=int,
        default=3,
        help="integer display scale (default: 3, producing a 720x480 game view)",
    )
    result.add_argument(
        "--max-frames",
        type=int,
        help="exit after this many displayed frames; useful for automated smoke tests",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.scale < 1:
        raise SystemExit("--scale must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")

    settings = load_settings()
    rom_path = args.rom or settings.rom_path
    if rom_path is None:
        raise SystemExit("no ROM configured: pass --rom PATH or set GBHARNESS_ROM")
    if not rom_path.is_file():
        raise SystemExit(f"ROM does not exist: {rom_path}")

    rom = rom_path.read_bytes()
    validate_rom(rom)
    core_path = find_core(settings.libretro_core)

    pygame.init()
    game_size = (240 * args.scale, 160 * args.scale)
    panel_height = 58
    screen = pygame.display.set_mode((game_size[0], game_size[1] + panel_height))
    pygame.display.set_caption(f"gbharness — {rom_path.name}")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    latch = FrameLatch()
    held_keys: set[int] = set()

    try:
        with LibretroInputPort(rom, core=core_path, frame_sink=latch.sink) as port:
            port.advance(6)  # discard undefined startup framebuffer frames
            running = True
            displayed_frames = 0
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key in KEY_BINDINGS:
                        if event.key not in held_keys:
                            held_keys.add(event.key)
                            port.press(KEY_BINDINGS[event.key])
                    elif event.type == pygame.KEYUP and event.key in KEY_BINDINGS:
                        held_keys.discard(event.key)
                        port.release(KEY_BINDINGS[event.key])
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        for key in held_keys:
                            port.release(KEY_BINDINGS[key])
                        held_keys.clear()

                if not running:
                    break
                if not port.advance(1):
                    break

                frame = latch.require()
                rgb = to_rgb(frame)
                native = pygame.image.frombytes(
                    rgb.tobytes(),
                    (frame.width, frame.height),
                    "RGB",
                )
                scaled = pygame.transform.scale(native, game_size)
                screen.fill((16, 18, 24))
                screen.blit(scaled, (0, 0))
                controls = font.render(
                    "Arrows: D-pad   Z: A   X: B   A: L   S: R",
                    True,
                    (232, 235, 242),
                )
                system = font.render(
                    "Enter: Start   Right Shift: Select   Esc: Quit",
                    True,
                    (168, 176, 192),
                )
                screen.blit(controls, (12, game_size[1] + 8))
                screen.blit(system, (12, game_size[1] + 31))
                pygame.display.flip()
                displayed_frames += 1
                if args.max_frames is not None and displayed_frames >= args.max_frames:
                    running = False
                clock.tick(round(FRAME_RATE))
    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
