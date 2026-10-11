# neatHack build log

## Project

**Working name:** gbharness arena

A provider-neutral game-agent runtime where AI agents observe emulator frames, choose
bounded controller actions, and compete with humans or other agents. Super Mario Bros. is
the first integration test, not the final game or product.

**Problem:** game-playing agent demos are often opaque, tied to one game, or reduced to a
single scripted bot. This project aims to make agent decisions observable, replayable, and
portable across games and multiplayer formats.

## Team

- Abhinav Ohri

## Stack

- Python 3.13 and `uv`
- mGBA through the Libretro ABI
- Pygame live viewer
- Vertex AI / Google Gen AI SDK (default cloud decision provider)
- OpenAI Decisions API (optional comparison adapter)
- neatlogs for agent-run traces
- Entire for code-session checkpoints
- cfo.ai for pricing, cost, revenue, and runway planning

## Build log

### 2026-10-10

- Completed: cloned `MaqsudMallick/gba-harness` and created branch
  `feat/mario-rom-integration`.
- Completed: added macOS support for downloading and discovering the official mGBA
  Libretro core.
- Completed: loaded the locally supplied GBA cartridge and verified World 1-1 renders.
- Completed: added ROM smoke-test and live keyboard-controlled emulator entry points.
- Completed verification: 294 tests passed after the emulator integration; the generated
  screenshots showed the title/level flow and a scripted move-and-jump action.
- Finding: the live window is currently human-controlled. The prior move-and-jump check
  was scripted and must not be presented as an autonomous agent run.
- Completed: installed Entire CLI 0.11.4 from the official Homebrew cask.
- Completed: authenticated Entire, enabled the repository, installed Git/Codex/Claude
  hooks, and verified the installation with `entire doctor`. Codex reports that its seven
  repository hooks still require explicit approval through `/hooks`.
- Completed: recovered the missed Codex hook capture with Entire's supported session-attach
  flow. Checkpoint `01M4K4DYFVKB5TF5R77KZMJD5J` is attached to commit
  `c47ae93c9583137756b38c9a89c1b5cba36f04c7` and is locally verified by both
  `entire checkpoint list` and `entire checkpoint explain HEAD`. Expected web URL:
  <https://entire.io/gh/MaqsudMallick/gba-harness/commit/c47ae93c9583137756b38c9a89c1b5cba36f04c7>.
- Completed: installed neatlogs 1.4.26 with its Vertex AI integration and ran the
  network-free Doctor v2 check. It passed with four spans, zero dropped spans, and a valid
  local envelope.
- Completed: ran the authenticated Neatlogs Doctor v2 hosted probe against the required US
  ingest endpoint. Trace `7dc923559d27cee58ecd0e0fdab5e77c` finalized successfully; all
  four spans were visible with valid hierarchy, attributes, input/output, SDK metadata, and
  numeric token fields. This proves the synthetic telemetry path, not the application run.
- Completed: implemented a provider-neutral decision contract, Vertex AI structured-output
  adapter, game-policy boundary, long-horizon runner, low-confidence fallback, decision
  error recovery, terminal-state confirmation, neatlogs workflow/agent/tool spans, and a
  live **AGENT CONTROL** viewer.
- Completed verification: `ruff` passed, strict `mypy` passed for all 23 source files, and
  all 302 tests passed. The code tests are offline and do not prove a real Vertex decision;
  the separate Doctor probe proves hosted trace ingestion and readback.
- Revalidated the authenticated Neatlogs hosted probe. Trace
  `c7f5c274f3d8b34c1bc3fa5eaa754d80` finalized with four spans, valid hierarchy,
  preserved inputs/outputs and metadata, numeric token fields, and no dropped spans.
- Used a real traced Vertex episode to reproduce intermittent truncated decision JSON. With
  the original 160-token output ceiling, trace `9aac73ec297bd530759980c3d0ca4ca6`
  recorded three completed turns and two bounded error-recovery attempts in a five-turn run.
- Fixed the failure by raising the structured-decision output ceiling to 320 tokens. Also
  made the runner print its Neatlogs trace ID on both successful and failed process exits so
  every application run can be found without guessing in the dashboard.
- Verified the fix with trace `a467fa9f0aaeb9bd08a00d29fa09e4d8`: all five requested
  turns produced decisions and actions, with no recovery turn consumed. The short run ended
  at its configured turn limit; it does not claim level completion.
- Completed post-fix verification: `ruff` passed, strict `mypy` passed for all 23 source
  files, and all 303 tests passed. A focused regression test checks the 320-token budget and
  a new observability test checks stable trace-ID capture.
- Resolved on 2026-10-11: refreshed the Entire login and confirmed the GitHub mirror is
  ready. The recorded checkpoint ID and target URL remain the submission evidence.
- Pending external verification: open the recorded application traces in the dashboard and
  save sanitized shareable links or screenshots. The local API key can ingest and Doctor can
  verify its own probe, but public trace readback requires a separate read token/project ID.
- Pending external action: build the business plan in cfo.ai, covering pricing, costs,
  revenue, and runway; `cfo-ai-brief.md` contains prepared inputs, but only the resulting
  cfo.ai plan counts as submission evidence.

### 2026-10-11

- Completed: added a local closed-loop Mario World 1-1 visual skill after the high-frequency
  Vertex baseline exhausted its request quota. The controller uses only framebuffer cues and
  bounded controller actions; it does not inspect game RAM or restore save states.
- Completed fresh-boot emulator verification: the agent cleared World 1-1, touched the flag,
  entered the castle, and stopped only after the World 1-2 card appeared. The passing run
  completed in 126 control turns and 2,928 emulated frames.
- Completed Neatlogs application evidence for the level-clear run: trace
  `857e03574e75b220897fd4c0c42dc5b1` finished with status `completed` using provider
  `local-vision`. A local verification screenshot is written to
  `artifacts/agent-level-clear.png` (the artifacts directory remains uncommitted).
- Completed: the Entire GitHub mirror is ready at `/gh/MaqsudMallick/gba-harness`, and the
  recorded checkpoint remains `01M4K4DYFVKB5TF5R77KZMJD5J` at commit `c47ae93...`.
- Completed: created `feat/mario-kart-multiplayer` from the updated remote `main` and added
  real 2-4 player link-cable support using the open-source mGBA Splitscreen libretro core.
  The harness now routes independent controller ports, applies multi-agent actions on one
  shared clock, and extracts a native 240x160 observation for each player.
- Completed real-ROM verification with the locally supplied Mario Kart: Super Circuit
  cartridge: both linked screens reached the game's shared Mario GP / VS / Battle menu and
  displayed `2P`. Evidence is captured locally at
  `artifacts/mario-kart-linked-lobby.png`; neither the ROM nor artifacts are committed.
- Completed multiplayer verification: all 322 tests pass, Ruff is clean, strict mypy is
  clean across 27 source files, and the real core reports controller ports 0 and 1 polled.
- Completed: built a four-agent Zelda: Four Swords integration without changing the Mario
  Kart adapter. Four private screen observations are sent concurrently to four Google models
  through Vertex AI, then their finite controller macros execute atomically on one linked
  clock. Calls, confidence, recovery, tokens, roles, and actions are Neatlogs-instrumented.
- Completed real-ROM verification from a fresh boot: deterministic setup created four local
  files, entered Four Swords, completed the four-system handshake, and visually confirmed all
  four Links in the same dungeon before spending a model call. Local evidence is written to
  `artifacts/zelda-four-swords-4p-setup.png` and remains uncommitted.
- Completed live four-model verification with a four-call ceiling. Gemini 2.5 Flash Lite,
  Gemini 2.5 Flash, Gemini 2.5 Pro, and Gemini 3 Flash Preview each observed its own dungeon
  view and chose one action; the team advanced in lockstep. Trace
  `651dabc1d005e5c15346b5899a107a47` records 4,106 input and 372 output tokens. The run ended
  at its configured one-turn limit and does not claim room completion.
- Completed: restructured Four Swords observability around one
  `four_swords_puzzle_attempt` WORKFLOW trace. The verified local puzzle motor skill is now a
  child CHAIN with phase TOOL spans and a framebuffer GUARDRAIL; the subsequent team loop is
  another child CHAIN with distinct P1-P4 decision/input spans, nested wrapped Vertex calls,
  synchronized controller parameters, consensus checks, retries, token totals, and final run
  outcome. This preserves the important distinction that the first switch puzzle is solved
  by the local visual motor skill rather than claiming it was solved by the LLMs.
- Completed: added a live human-plus-agent mode. Live Zelda defaults to P1 keyboard input and
  three Vertex players, records the human lane as a TOOL rather than an LLM AGENT, and does
  not count human actions against the model-call budget. Headless mode and
  `--human-player 0` preserve the four-agent evaluation.
- Completed focused verification for the trace hierarchy and human input adapter: Ruff and
  targeted strict mypy passed, and 9 focused tests passed. Full regression verification also
  passed all 348 tests.
- Completed a fresh hosted single-trace evaluation after the instrumentation change. Trace
  `2d9b1ce01e94f88fc7622be13093d61b` includes the verified puzzle prelude and one four-model
  synchronized decision turn with 4,110 input and 325 output tokens. The process observed
  one trace ID for the run and saved local visual evidence at
  `artifacts/zelda-single-trace-verification.png`; dashboard-tree inspection or a sanitized
  share link remains required before public submission.
- Failure reproduced: the first live human-plus-agent implementation queued one human macro
  behind the slowest model call, so P1 often timed out to `wait` and did not feel human
  controlled. Trace `e56cc3206808bca57ce0adff5d535b27` records that failed behavior.
- Fixed: the hybrid runner now executes P1 keyboard macros on the emulator immediately while
  P2-P4 are reasoning, supports arrows/WASD and held-key repeat, keeps those actions inside
  the same root trace, and excludes them from both the model-call budget and the agents'
  synchronized action batch. A regression test blocks all three agents until the human move
  is observed on the controller port.
- Verified the fix in hosted trace `1f7240d2a33dd1e334174e91025b1aab`: P1 executed
  `move_left` as `human_live_control` while three Vertex decisions were in progress. The run
  completed two turns and six model calls before the window was closed.
- Failure reproduced: after the configured agent-turn budget, the viewer displayed
  `RUN FINISHED` but continued accepting UI events into an unconsumed queue, making every
  keyboard control appear stuck. The no-prelude diagnostic trace
  `aa8ea6976a50f8af566a1e18b7d77545` records P1 correctly waiting through five turns before
  this frozen results state.
- Fixed: a live human-plus-agent run now enters traced `P1 FREE PLAY` after its agent budget.
  The worker continues consuming and executing keyboard actions until Escape, so the viewer
  no longer presents a non-interactive final frame as though it were playable.
- Failure reproduced: extending the three independent Vertex players to twelve turns spent
  more calls but did not solve the room. Their scene labels contradicted each other and their
  actions lacked a shared objective, proving that additional independent button guesses are
  not an adequate cooperative planner.
- Completed: replaced independent live decisions with a rotating team-captain provider.
  Gemini 2.5 Flash, Gemini 2.5 Pro, and Gemini 3 Flash Preview now alternate as captain; one
  composite visual call returns the joint P2-P4 actions, shared objective, and a concrete P1
  instruction. Neatlogs records the captain plan, nested wrapped call, each assigned player
  action, human live inputs, and synchronized execution under the same root trace. This cuts
  the live planning cost from three calls per turn to one.
- Evaluation finding: the first six-plan captain run produced consistent shared scene labels
  and identified the Chambers of Insight pressure-plate puzzle, but it did not yet prove a
  room clear and often assigned the same action to all AI players. Multi-step spatial plans
  and a verified progress evaluator remain necessary before claiming autonomous completion.

## Demo

Current runnable demo and smoke tests:

```bash
uv run python examples/run_rom.py --rom roms/game.gba --screenshot artifacts/boot.png
uv run python examples/live_rom.py
uv run python examples/agent_rom.py --controller local --max-turns 150
uv run python examples/multiplayer_rom.py --rom roms/game.gba --players 2
uv run python examples/zelda_four_swords.py --setup-only
```

ROMs and credentials are local-only and must not be committed or shared.

## Known limitations and next steps

- World 1-1 has a verified local visual skill; unfamiliar levels still require a general
  provider or another game-specific skill.
- The reusable game-policy boundary now has Mario and Zelda: Four Swords adapters. Zelda has
  proven four-agent action execution; completing a full cooperative room remains an evaluation
  milestone.
- The representative completed application trace has been recorded; a sanitized dashboard
  link or screenshot still has to be prepared for the public submission.
- Linked agent-vs-human and agent-vs-agent stepping is implemented. Mario Kart has a verified
  local race completion; Four Swords still needs a longer bounded evaluation that completes
  and verifies a cooperative room.
- Ranked matches need a policy for save states; emulator rollback should normally be
  disabled because it gives agents an unfair advantage.

## Submission

Confirmed from the published hackathon page on 2026-10-10:

- Use neatlogs, Entire, and cfo.ai with a clear role for each.
- Provide a public repository and a demo video of three minutes or less.
- The demo must explicitly cover architecture, implementation, runtime, reasoning, and
  results for neatlogs, Entire, and cfo.ai.
- Preserve evidence: before/after neatlogs traces, an Entire checkpoint or graph output,
  the cfo.ai business plan, a recovery-from-failure run, and public build links tagged
  `#neatHack`.
- Submission is a public quote post during the announced window; the final cutoff must be
  confirmed through the official Discord announcement.

Outstanding:

- Shareable neatlogs trace links or sanitized screenshots.
- cfo.ai business plan.
- Public repository, demo video, and public build posts.
