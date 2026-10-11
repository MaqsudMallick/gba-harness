"""Run four Vertex visual agents against one linked Zelda: Four Swords session.

The user-owned ROM stays local under ``roms/``. Use ``--smoke`` to verify the four linked
consoles without spending model calls, then remove it for a bounded, Neatlogs-traced run.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pygame

from gbharness.agent import (
    FOUR_SWORDS_POLICY,
    FOUR_SWORDS_ROLES,
    FirstTrainingPuzzleResult,
    LinkedAgentRunner,
    LinkedEpisodeResult,
    LinkedRunnerConfig,
    LinkedTurnRecord,
    PlayerAssignment,
    VertexDecisionProvider,
    bootstrap_four_swords,
    solve_first_training_puzzle,
)
from gbharness.backends import LibretroMultiplayerPort, find_multiplayer_core, validate_rom
from gbharness.config import load_settings
from gbharness.multiplayer import LinkedSession
from gbharness.observability import NeatlogsSession
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_png, to_rgb

DEFAULT_ROM = Path("roms/Legend of Zelda, The - A Link to the Past & Four Swords (USA).gba")
DEFAULT_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--rom", type=Path, default=DEFAULT_ROM)
    result.add_argument("--core", type=Path, help="mGBA Splitscreen libretro core")
    result.add_argument("--max-turns", type=int, default=5)
    result.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=3,
        help="abort after this many consecutive failures for any one player",
    )
    result.add_argument(
        "--max-model-calls",
        type=int,
        default=20,
        help="hard team-wide ceiling; one complete turn costs four calls",
    )
    result.add_argument(
        "--models",
        default=os.environ.get("GBHARNESS_ZELDA_MODELS", ",".join(DEFAULT_MODELS)),
        help="exactly four comma-separated Vertex model IDs",
    )
    result.add_argument(
        "--smoke",
        action="store_true",
        help="boot four linked consoles and exit without calling Vertex or neatlogs",
    )
    result.add_argument("--boot-frames", type=int, default=900)
    result.add_argument(
        "--skip-setup",
        action="store_true",
        help="leave the consoles at file select instead of entering Four Swords",
    )
    result.add_argument(
        "--setup-only",
        action="store_true",
        help="enter verified four-player gameplay and exit without cloud calls",
    )
    result.add_argument(
        "--skip-first-puzzle",
        action="store_true",
        help="start Vertex agents in the lobby instead of solving the first switch puzzle",
    )
    result.add_argument("--screenshot", type=Path)
    result.add_argument("--log-turns", action="store_true")
    result.add_argument(
        "--live",
        action="store_true",
        help="show all four consoles and agent decisions in a live Pygame window",
    )
    result.add_argument("--scale", type=int, default=2, help="live window integer scale")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.setup_only and args.skip_setup:
        raise SystemExit("--setup-only and --skip-setup cannot be combined")
    if args.live and (args.smoke or args.setup_only):
        raise SystemExit("--live cannot be combined with --smoke or --setup-only")
    if args.scale < 1:
        raise SystemExit("--scale must be positive")
    if not args.rom.is_file():
        raise SystemExit(f"ROM does not exist: {args.rom}")
    if args.boot_frames < 1:
        raise SystemExit("--boot-frames must be positive")
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if len(models) != 4:
        raise SystemExit(f"--models needs exactly four Vertex model IDs, got {len(models)}")

    rom = args.rom.read_bytes()
    validate_rom(rom)
    latch = FrameLatch()
    telemetry: NeatlogsSession | None = None
    final_png: bytes | None = None

    try:
        with LibretroMultiplayerPort(
            rom,
            players=4,
            core=find_multiplayer_core(args.core),
            frame_sink=latch.sink,
            four_swords_assist=True,
        ) as port:
            port.start_linked_session()
            linked = LinkedSession(port, max_step_frames=1800)
            linked.step({}, args.boot_frames)
            if args.smoke:
                final_png = to_png(latch.require())
                print(
                    "zelda smoke passed: players=4, "
                    f"frames={linked.frame}, ports={sorted(port.queried_ports)}"
                )
                return 0

            if not args.skip_setup:
                setup = bootstrap_four_swords(linked, latch)
                print(
                    "zelda setup passed: four-player dungeon visible, "
                    f"frames={setup.frames_advanced}, "
                    f"handshake_confirms={setup.handshake_confirms}",
                    flush=True,
                )
                if args.setup_only:
                    final_png = to_png(latch.require())
                    return 0

            settings = load_settings()
            project = settings.require_vertex_project()
            neatlogs_key = settings.require_neatlogs_key()
            with NeatlogsSession(
                neatlogs_key,
                workflow_name="zelda-four-swords-multi-agent",
                endpoint=settings.neatlogs_endpoint,
            ) as telemetry:
                assignments: list[PlayerAssignment] = []
                for player, (model, role) in enumerate(zip(models, FOUR_SWORDS_ROLES, strict=True)):
                    vertex = VertexDecisionProvider(
                        project=project,
                        location=settings.google_cloud_location,
                        model=model,
                    )
                    provider = vertex.with_client(
                        telemetry.wrap(
                            vertex.client,
                            game=FOUR_SWORDS_POLICY.name,
                            player=player + 1,
                            role=role,
                        )
                    )
                    assignments.append(PlayerAssignment(player, role, provider))

                runner = LinkedAgentRunner(
                    session=linked,
                    frames=latch,
                    assignments=tuple(assignments),
                    policy=FOUR_SWORDS_POLICY,
                    config=LinkedRunnerConfig(
                        max_turns=args.max_turns,
                        max_model_calls=args.max_model_calls,
                        confidence_threshold=settings.confidence_threshold,
                        max_consecutive_errors=args.max_consecutive_errors,
                    ),
                    telemetry=telemetry,
                    on_turn=_print_turn if args.log_turns else None,
                )

                def puzzle_prelude(
                    on_phase: Callable[[str], None] | None = None,
                ) -> FirstTrainingPuzzleResult:
                    solved = solve_first_training_puzzle(
                        linked,
                        latch,
                        on_phase=on_phase,
                        telemetry=telemetry,
                    )
                    print(
                        "zelda puzzle passed: Basic Training 1 outer switches solved, "
                        f"frames={solved.frames_advanced}, "
                        f"warp_sweep_steps={solved.warp_sweep_steps}, "
                        f"reward_chests_visible={solved.reward_chests_visible}",
                        flush=True,
                    )
                    return solved

                prelude = None if args.skip_setup or args.skip_first_puzzle else puzzle_prelude
                if args.live:
                    result = _run_live(runner, latch, args.scale, prelude=prelude)
                else:
                    if prelude is not None:
                        prelude()
                    result = runner.run()
                final_png = to_png(latch.require())
                summary = (
                    f"zelda agents finished: status={result.status}, "
                    f"turns={len(result.turns)}, frames={result.total_frames}, "
                    f"calls={result.model_calls}, input_tokens={result.input_tokens}, "
                    f"output_tokens={result.output_tokens}, "
                    f"providers={','.join(result.providers)}"
                )
                if telemetry.trace_id is not None:
                    summary += f", neatlogs_trace_id={telemetry.trace_id}"
                print(summary)
                return 0 if result.status == "completed" else 1
    except Exception:
        if telemetry is not None and telemetry.trace_id is not None:
            print(f"zelda agents failed: neatlogs_trace_id={telemetry.trace_id}", file=sys.stderr)
        raise
    finally:
        if args.screenshot is not None and final_png is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            args.screenshot.write_bytes(final_png)


def _print_turn(record: LinkedTurnRecord) -> None:
    actions = " | ".join(
        f"P{player.player + 1} {player.provider}: {player.action} "
        f"({player.confidence:.0%}, {player.scene})"
        for player in record.players
    )
    print(
        f"turn={record.turn} frame={record.frame} +{record.frames_advanced} | {actions}",
        flush=True,
    )


def _run_live(
    runner: LinkedAgentRunner,
    latch: FrameLatch,
    scale: int,
    *,
    prelude: Callable[[Callable[[str], None] | None], FirstTrainingPuzzleResult] | None = None,
) -> LinkedEpisodeResult:
    """Keep a responsive four-console window open while agents run in a worker."""
    pygame.init()
    frame = latch.require()
    native_size = (frame.width, frame.height)
    game_size = (frame.width * scale, frame.height * scale)
    panel_height = 146
    screen = pygame.display.set_mode((game_size[0], game_size[1] + panel_height))
    pygame.display.set_caption("gbharness — ZELDA FOUR SWORDS — 4 VERTEX AGENTS")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    latest: list[LinkedTurnRecord | None] = [None]
    phase = [
        "FAST MOTOR SKILL — preparing the first cooperative puzzle"
        if prelude is not None
        else "4 VERTEX AGENTS — observing the current room"
    ]
    outcome: list[LinkedEpisodeResult] = []
    failure: list[BaseException] = []
    previous_on_turn = runner.on_turn

    def update(record: LinkedTurnRecord) -> None:
        latest[0] = record
        if previous_on_turn is not None:
            previous_on_turn(record)

    runner.on_turn = update

    def work() -> None:
        try:
            if prelude is not None:
                prelude(lambda message: phase.__setitem__(0, message))
                phase[0] = "PUZZLE 1 SOLVED — four Vertex agents are observing"
            outcome.append(runner.run())
        except BaseException as exc:
            failure.append(exc)

    worker = threading.Thread(target=work, name="zelda-four-agent", daemon=True)
    worker.start()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    runner.stop_event.set()
                    running = False

            latest_frame = latch.latest()
            if latest_frame is None:
                clock.tick(30)
                continue
            if (latest_frame.width, latest_frame.height) != native_size:
                raise RuntimeError("live Zelda layout changed dimensions unexpectedly")
            rgb = to_rgb(latest_frame)
            native = pygame.image.frombytes(rgb.tobytes(), native_size, "RGB")
            screen.fill((10, 12, 18))
            screen.blit(pygame.transform.scale(native, game_size), (0, 0))

            record = latest[0]
            complete = not worker.is_alive()
            header = "RUN FINISHED — press Esc to close" if complete else phase[0]
            lines = [header]
            if record is None:
                lines.append("All four players are observing the dungeon…")
            else:
                for player in record.players:
                    model = player.provider.removeprefix("vertex:")
                    lines.append(
                        f"P{player.player + 1} {model}: {player.action} ({player.confidence:.0%})"
                    )
            lines.append("Esc: stop agents and close")
            for index, line in enumerate(lines[:6]):
                color = (110, 238, 174) if index == 0 else (225, 228, 238)
                screen.blit(
                    font.render(line, True, color),
                    (12, game_size[1] + 8 + index * 22),
                )
            pygame.display.flip()
            clock.tick(30)
    finally:
        runner.stop_event.set()
        worker.join()
        pygame.quit()

    if failure:
        raise failure[0]
    return outcome[0]


if __name__ == "__main__":
    raise SystemExit(main())
