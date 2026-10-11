"""Run four Vertex visual agents against one linked Zelda: Four Swords session.

The user-owned ROM stays local under ``roms/``. Use ``--smoke`` to verify the four linked
consoles without spending model calls, then remove it for a bounded, Neatlogs-traced run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from gbharness.agent import (
    FOUR_SWORDS_POLICY,
    FOUR_SWORDS_ROLES,
    LinkedAgentRunner,
    LinkedRunnerConfig,
    LinkedTurnRecord,
    PlayerAssignment,
    VertexDecisionProvider,
    bootstrap_four_swords,
)
from gbharness.backends import LibretroMultiplayerPort, find_multiplayer_core, validate_rom
from gbharness.config import load_settings
from gbharness.multiplayer import LinkedSession
from gbharness.observability import NeatlogsSession
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_png

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
    result.add_argument("--screenshot", type=Path)
    result.add_argument("--log-turns", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.setup_only and args.skip_setup:
        raise SystemExit("--setup-only and --skip-setup cannot be combined")
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


if __name__ == "__main__":
    raise SystemExit(main())
