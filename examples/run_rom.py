"""Boot a user-supplied GBA cartridge and capture a smoke-test screenshot.

ROM images are intentionally not fetched by this project. Supply a cartridge dump you are
legally allowed to use with ``--rom`` or ``GBHARNESS_ROM``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gbharness.actions import press, release, tap, wait
from gbharness.backends import LibretroInputPort, find_core, validate_rom
from gbharness.config import load_settings
from gbharness.session import InputSession
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_png


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--rom",
        type=Path,
        help="path to a legally obtained .gba file; defaults to GBHARNESS_ROM",
    )
    result.add_argument(
        "--boot-frames",
        type=int,
        default=180,
        help="frames to run before tapping Start (default: 180, about three seconds)",
    )
    result.add_argument(
        "--after-start-frames",
        type=int,
        default=120,
        help="frames to run after tapping Start (default: 120, about two seconds)",
    )
    result.add_argument(
        "--screenshot",
        type=Path,
        default=Path("artifacts/rom-smoke.png"),
        help="PNG output path (default: artifacts/rom-smoke.png)",
    )
    result.add_argument(
        "--move-right-frames",
        type=int,
        default=0,
        help="after boot, hold Right for this many frames before capturing",
    )
    result.add_argument(
        "--jump",
        action="store_true",
        help="tap A while moving right (requires --move-right-frames)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings()
    rom_path = args.rom or settings.rom_path
    if rom_path is None:
        raise SystemExit("no ROM configured: pass --rom PATH or set GBHARNESS_ROM")
    if not rom_path.is_file():
        raise SystemExit(f"ROM does not exist: {rom_path}")
    if args.boot_frames < 1 or args.after_start_frames < 0 or args.move_right_frames < 0:
        raise SystemExit("frame counts must be non-negative, and --boot-frames must be positive")
    if args.jump and args.move_right_frames == 0:
        raise SystemExit("--jump requires --move-right-frames")

    rom = rom_path.read_bytes()
    validate_rom(rom)
    core_path = find_core(settings.libretro_core)
    latch = FrameLatch()
    with (
        LibretroInputPort(
            rom,
            core=core_path,
            frame_sink=latch.sink,
        ) as port,
        InputSession(port) as session,
    ):
        session.submit(wait(args.boot_frames))
        session.submit(tap("start"))
        session.submit(wait(args.after_start_frames))
        if args.move_right_frames:
            session.submit(press("right"))
            consumed = 0
            if args.jump:
                jump = tap("a")
                session.submit(jump)
                consumed = jump.hold_frames + jump.gap_frames
            if args.move_right_frames > consumed:
                session.submit(wait(args.move_right_frames - consumed))
            session.submit(release("right"))
        frame = latch.require()

    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    args.screenshot.write_bytes(to_png(frame, scale=2))
    print(f"ROM         {rom_path}")
    print(f"core        {core_path}")
    print(f"frames      {port.frames_run}")
    print(f"framebuffer {frame.width}x{frame.height}, pitch {frame.pitch} bytes")
    print(f"screenshot  {args.screenshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
