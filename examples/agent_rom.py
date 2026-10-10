"""Let a traced Vertex AI agent play the configured ROM while you watch."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import pygame

from gbharness.agent import (
    ACTION_DESCRIPTIONS,
    ACTIONS,
    AgentRunner,
    EpisodeResult,
    GamePolicy,
    RunnerConfig,
    TurnRecord,
    VertexDecisionProvider,
)
from gbharness.backends import LibretroInputPort, find_core, validate_rom
from gbharness.config import load_settings
from gbharness.observability import NeatlogsSession
from gbharness.safety import Limits
from gbharness.session import InputSession
from gbharness.vision.capture import FrameLatch
from gbharness.vision.encode import to_rgb


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--rom", type=Path, help=".gba file; defaults to GBHARNESS_ROM")
    result.add_argument("--max-turns", type=int, default=500)
    result.add_argument("--scale", type=int, default=3)
    result.add_argument("--headless", action="store_true", help="run without a Pygame window")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings()
    if settings.agent_provider != "vertex":
        raise SystemExit(
            f"agent_rom currently supports provider 'vertex', got {settings.agent_provider!r}"
        )
    rom_path = args.rom or settings.rom_path
    if rom_path is None or not rom_path.is_file():
        raise SystemExit("configure an existing ROM with --rom or GBHARNESS_ROM")
    rom = rom_path.read_bytes()
    validate_rom(rom)
    project = settings.require_vertex_project()
    neatlogs_key = settings.require_neatlogs_key()
    policy = GamePolicy(
        name="super-mario-bros",
        goal=(
            "Start the game if needed, then autonomously complete the current level by "
            "moving toward its goal and surviving obstacles."
        ),
        actions=ACTIONS,
        descriptions=ACTION_DESCRIPTIONS,
    )
    latch = FrameLatch()

    with (
        NeatlogsSession(neatlogs_key, endpoint=settings.neatlogs_endpoint) as telemetry,
        LibretroInputPort(
            rom,
            core=find_core(settings.libretro_core),
            frame_sink=latch.sink,
        ) as port,
        InputSession(
            port,
            Limits(episode_frame_budget=settings.max_episode_frames),
        ) as session,
    ):
        session.submit(ACTIONS["wait"])  # render a stable first observation
        vertex = VertexDecisionProvider(
            project=project,
            location=settings.google_cloud_location,
            model=settings.vertex_model,
        )
        provider = vertex.with_client(
            telemetry.wrap(
                vertex.client,
                game=policy.name,
                mode="autonomous",
            )
        )
        runner = AgentRunner(
            session=session,
            frames=latch,
            provider=provider,
            policy=policy,
            config=RunnerConfig(
                max_turns=args.max_turns,
                confidence_threshold=settings.confidence_threshold,
            ),
            telemetry=telemetry,
        )
        result = runner.run() if args.headless else _run_live(runner, latch, args.scale)

    print(
        f"agent finished: status={result.status}, turns={len(result.turns)}, "
        f"frames={result.total_frames}, provider={result.provider}"
    )
    return 0 if result.status == "completed" else 1


def _run_live(runner: AgentRunner, latch: FrameLatch, scale: int) -> EpisodeResult:
    if scale < 1:
        raise SystemExit("--scale must be at least 1")
    pygame.init()
    game_size = (240 * scale, 160 * scale)
    panel_height = 86
    screen = pygame.display.set_mode((game_size[0], game_size[1] + panel_height))
    pygame.display.set_caption("gbharness — AGENT CONTROL")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    latest: list[TurnRecord | None] = [None]
    outcome: list[EpisodeResult] = []
    failure: list[BaseException] = []

    runner.on_turn = lambda record: latest.__setitem__(0, record)

    def work() -> None:
        try:
            outcome.append(runner.run())
        except BaseException as exc:
            failure.append(exc)

    worker = threading.Thread(target=work, name="game-agent", daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    runner.stop_event.set()

            frame = latch.require()
            rgb = to_rgb(frame)
            native = pygame.image.frombytes(rgb.tobytes(), (frame.width, frame.height), "RGB")
            screen.fill((12, 14, 20))
            screen.blit(pygame.transform.scale(native, game_size), (0, 0))
            record = latest[0]
            lines = ["AGENT CONTROL — keyboard input disabled"]
            if record is None:
                lines.append("Observing screen and choosing the first action…")
                lines.append("Esc: stop")
            else:
                lines.append(
                    f"Turn {record.turn}  Action: {record.action}  "
                    f"Confidence: {record.confidence:.0%}"
                )
                lines.append(
                    f"Scene: {record.scene}  Progress: "
                    f"{record.progress:.0%}" if record.progress is not None else
                    f"Scene: {record.scene}  Progress: unknown"
                )
            for index, line in enumerate(lines):
                color = (112, 232, 168) if index == 0 else (220, 224, 234)
                screen.blit(font.render(line, True, color), (12, game_size[1] + 8 + 25 * index))
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
