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
    FourSwordsVisualTargetNavigator,
    HumanActionSource,
    HumanDecisionProvider,
    LinkedAgentRunner,
    LinkedEpisodeResult,
    LinkedRunnerConfig,
    LinkedTurnRecord,
    PlayerAssignment,
    VertexDecisionProvider,
    VertexTeamDecisionProvider,
    bootstrap_four_swords,
    four_swords_dialogue_visible,
    four_swords_probe_target_reached,
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
    result.add_argument("--max-turns", type=int, default=12)
    result.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=3,
        help="abort after this many consecutive failures for any one player",
    )
    result.add_argument(
        "--max-model-calls",
        type=int,
        default=12,
        help="hard ceiling; coordinated human mode costs one rotating-captain call per turn",
    )
    result.add_argument(
        "--models",
        default=os.environ.get("GBHARNESS_ZELDA_MODELS", ",".join(DEFAULT_MODELS)),
        help="four comma-separated Vertex model IDs; a human player's entry is unused",
    )
    result.add_argument(
        "--human-player",
        type=int,
        choices=(0, 1, 2, 3, 4),
        help="one-based keyboard player; 0 means all agents (default: P1 in live mode)",
    )
    result.add_argument(
        "--npc-autopilot",
        action="store_true",
        help="let the team planner move P1 and complete visible NPC dialogue without combat",
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
    human_player = args.human_player if args.human_player is not None else (1 if args.live else 0)
    if human_player and not args.live:
        raise SystemExit("--human-player requires --live so keyboard input can be captured")

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
                human_actions = HumanActionSource() if human_player else None
                assignments: list[PlayerAssignment] = []
                ai_providers: list[VertexDecisionProvider] = []
                for player, (model, role) in enumerate(zip(models, FOUR_SWORDS_ROLES, strict=True)):
                    if human_actions is not None and player == human_player - 1:
                        assignments.append(
                            PlayerAssignment(
                                player,
                                role,
                                HumanDecisionProvider(human_actions),
                                participant_type="human",
                            )
                        )
                        continue
                    if human_player:
                        role = {
                            1: (
                                "Red Link AI route captain: identify the shared obstacle, "
                                "select a team mechanic, and give P1 one explicit instruction."
                            ),
                            2: (
                                "Blue Link AI mechanics specialist: test attack/interact when "
                                "movement is blocked and tell P1 how to complement it."
                            ),
                            3: (
                                "Purple Link AI switch specialist: seek a distinct pressure "
                                "plate or flank route and tell P1 which position to hold."
                            ),
                        }.get(player, role)
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
                    ai_providers.append(provider)
                    assignments.append(PlayerAssignment(player, role, provider))

                target_navigator = (
                    FourSwordsVisualTargetNavigator() if args.npc_autopilot else None
                )
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
                        autoplay_human_dialogue=args.npc_autopilot,
                    ),
                    team_provider=(
                        VertexTeamDecisionProvider(
                            tuple(ai_providers),
                            navigation_only=args.npc_autopilot,
                        )
                        if human_player
                        else None
                    ),
                    telemetry=telemetry,
                    on_turn=_print_turn if args.log_turns else None,
                    dialogue_detector=(
                        four_swords_dialogue_visible if args.npc_autopilot else None
                    ),
                    visual_plan_override=target_navigator,
                    probe_target_detector=(
                        four_swords_probe_target_reached if args.npc_autopilot else None
                    ),
                    team_target_detector=(
                        target_navigator.all_assigned_players_reached
                        if target_navigator is not None
                        else None
                    ),
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

                def run_attempt(
                    on_phase: Callable[[str], None] | None = None,
                    after_episode: Callable[[LinkedEpisodeResult], None] | None = None,
                ) -> LinkedEpisodeResult:
                    with telemetry.span(
                        "four_swords_puzzle_attempt",
                        "WORKFLOW",
                        input_value={
                            "game": FOUR_SWORDS_POLICY.name,
                            "objective": FOUR_SWORDS_POLICY.goal,
                            "participants": (
                                "one human, three Vertex agents"
                                if human_player
                                else "four Vertex agents"
                            ),
                            "human_player": f"P{human_player}" if human_player else None,
                            "models": models,
                        },
                        attributes={
                            "game": FOUR_SWORDS_POLICY.name,
                            "objective": FOUR_SWORDS_POLICY.goal,
                            "participant.count": 4,
                            "participant.mode": (
                                "one_human_three_vertex_agents"
                                if human_player
                                else "four_vertex_agents"
                            ),
                            "human.player": human_player,
                            "human.npc_autopilot": args.npc_autopilot,
                            "runtime.walkthrough_access": False,
                            "models": ",".join(models),
                            "puzzle.prelude_enabled": prelude is not None,
                        },
                    ) as root_span:
                        telemetry.log(
                            "Four Swords team attempt started with four private visual "
                            "observations and one shared emulator clock"
                        )
                        if prelude is not None:
                            prelude(on_phase)
                            if on_phase is not None:
                                on_phase(
                                    "PUZZLE 1 SOLVED — human and agents choosing next actions"
                                    if human_player
                                    else "PUZZLE 1 SOLVED — four agents choosing next actions"
                                )
                        episode = runner.run()
                        if after_episode is not None:
                            after_episode(episode)
                        telemetry.annotate(
                            root_span,
                            {
                                "run.status": episode.status,
                                "run.turns": len(episode.turns),
                                "run.frames": episode.total_frames,
                                "run.model_calls": episode.model_calls,
                                "run.input_tokens": episode.input_tokens,
                                "run.output_tokens": episode.output_tokens,
                            },
                        )
                        telemetry.record_output(
                            root_span,
                            "WORKFLOW",
                            {
                                "status": episode.status,
                                "turns": len(episode.turns),
                                "frames": episode.total_frames,
                                "model_calls": episode.model_calls,
                                "tokens": {
                                    "input": episode.input_tokens,
                                    "output": episode.output_tokens,
                                },
                                "last_instruction": (
                                    episode.turns[-1].team_instruction
                                    if episode.turns
                                    else ""
                                ),
                            },
                        )
                        return episode

                if args.live:
                    result = _run_live(
                        runner,
                        latch,
                        args.scale,
                        run_attempt=run_attempt,
                        includes_prelude=prelude is not None,
                        human_actions=human_actions,
                        human_player=human_player,
                    )
                else:
                    result = run_attempt()
                final_png = to_png(latch.require())
                if len(telemetry.trace_ids) != 1:
                    raise RuntimeError(
                        "expected one Neatlogs trace for the complete team run, got "
                        f"{len(telemetry.trace_ids)}: {','.join(telemetry.trace_ids)}"
                    )
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
        f"turn={record.turn} stage={record.team_stage} frame={record.frame} "
        f"+{record.frames_advanced} | {actions}",
        flush=True,
    )


def _run_live(
    runner: LinkedAgentRunner,
    latch: FrameLatch,
    scale: int,
    *,
    run_attempt: Callable[
        [Callable[[str], None] | None, Callable[[LinkedEpisodeResult], None] | None],
        LinkedEpisodeResult,
    ],
    includes_prelude: bool,
    human_actions: HumanActionSource | None,
    human_player: int,
) -> LinkedEpisodeResult:
    """Keep a responsive four-console window open while agents run in a worker."""
    pygame.init()
    pygame.key.set_repeat(180, 120)
    frame = latch.require()
    native_size = (frame.width, frame.height)
    game_size = (frame.width * scale, frame.height * scale)
    panel_height = 174
    screen = pygame.display.set_mode((game_size[0], game_size[1] + panel_height))
    pygame.display.set_caption(
        "gbharness — ZELDA FOUR SWORDS — "
        + ("1 HUMAN + 3 VERTEX AGENTS" if human_player else "4 VERTEX AGENTS")
    )
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    latest: list[LinkedTurnRecord | None] = [None]
    phase = [
        "FAST MOTOR SKILL — preparing the first cooperative puzzle"
        if includes_prelude
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

    def keep_human_active(_episode: LinkedEpisodeResult) -> None:
        if human_actions is None or not human_player:
            return
        phase[0] = f"AGENTS FINISHED — P{human_player} FREE PLAY ACTIVE"
        runner.telemetry.log(
            "Agent turn budget finished; P{player} human free play remains active",
            player=human_player,
        )
        action_count = 0
        while not runner.stop_event.is_set():
            action = human_actions.poll_action(timeout=0.05)
            if runner.stop_event.is_set():
                break
            if action is None or action not in runner.policy.actions:
                continue
            with runner.telemetry.span(
                f"player_{human_player}_free_play_action",
                "TOOL",
                input_value={"action": action, "phase": "free play after agent episode"},
                attributes={
                    "neatlogs.tool.name": "human_keyboard_controller",
                    "player.number": human_player,
                    "free_play.action_index": action_count + 1,
                    "frame.before": runner.session.frame,
                },
            ) as action_span:
                result = runner.session.submit({human_player - 1: runner.policy.resolve(action)})
                action_count += 1
                runner.telemetry.annotate(
                    action_span,
                    {
                        "frame.after": result.total_frames,
                        "frames.advanced": result.frames_advanced,
                        "emulation.ended": result.emulation_ended,
                    },
                )
                runner.telemetry.record_output(
                    action_span,
                    "TOOL",
                    {
                        "frames_advanced": result.frames_advanced,
                        "emulation_ended": result.emulation_ended,
                    },
                )
            phase[0] = f"P{human_player} FREE PLAY — {action}"
            if result.emulation_ended:
                runner.stop_event.set()
                break

    def work() -> None:
        try:
            outcome.append(
                run_attempt(
                    lambda message: phase.__setitem__(0, message),
                    keep_human_active,
                )
            )
            phase[0] = "PUZZLE 1 SOLVED — run trace finalized"
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
                    if human_actions is not None:
                        human_actions.submit("wait")
                    running = False
                elif human_actions is not None and event.type == pygame.KEYDOWN:
                    action = _human_action_for_key(event.key)
                    if action is not None:
                        human_actions.submit(action)
                        phase[0] = f"P{human_player} HUMAN CONTROL — {action}"

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
                lines.append("The team is preparing its first synchronized decision…")
            else:
                for player in record.players:
                    model = player.provider.removeprefix("vertex:")
                    lines.append(
                        f"P{player.player + 1} {model}: {player.action} ({player.confidence:.0%})"
                    )
                advice = record.team_instruction or next(
                    (
                        player.note
                        for player in record.players
                        if not player.provider.startswith("human:") and player.note
                    ),
                    "",
                )
                if advice:
                    lines.append(f"TEAM ADVICE: {advice[:105]}")
            if human_player:
                lines.append("Human: arrows/WASD move · N/F interact · M/G sword · Space wait")
            lines.append("Esc: stop agents and close")
            for index, line in enumerate(lines[:8]):
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


def _human_action_for_key(key: int) -> str | None:
    return {
        pygame.K_UP: "move_up",
        pygame.K_DOWN: "move_down",
        pygame.K_LEFT: "move_left",
        pygame.K_RIGHT: "move_right",
        pygame.K_n: "interact",
        pygame.K_m: "use_sword",
        pygame.K_RETURN: "start",
        pygame.K_RSHIFT: "select",
        pygame.K_SPACE: "wait",
        pygame.K_w: "move_up",
        pygame.K_s: "move_down",
        pygame.K_a: "move_left",
        pygame.K_d: "move_right",
        pygame.K_f: "interact",
        pygame.K_g: "use_sword",
    }.get(key)


if __name__ == "__main__":
    raise SystemExit(main())
