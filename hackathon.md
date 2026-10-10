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
- Vertex AI / Google Gen AI SDK (planned default decision provider)
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
- Pending external verification: first Entire checkpoint and a shareable checkpoint URL/ID.
- Pending external verification: representative traced agent run and a sanitized shareable
  trace link or screenshot.
- Pending external action: build the business plan in cfo.ai, covering pricing, costs,
  revenue, and runway; `cfo-ai-brief.md` contains prepared inputs, but only the resulting
  cfo.ai plan counts as submission evidence.

## Demo

Pending. Current local smoke tests:

```bash
uv run python examples/run_rom.py --rom roms/game.gba --screenshot artifacts/boot.png
uv run python examples/live_rom.py
```

ROMs and credentials are local-only and must not be committed or shared.

## Known limitations and next steps

- The autonomous loop and live viewer are implemented but have not yet been exercised with
  a real Vertex decision.
- The reusable game-policy boundary exists, but only the Mario starter policy is implemented.
- No representative application trace has been verified yet. Per the supplied Neatlogs
  instructions, the builder must run the instrumented entry point and provide its trace ID.
- Agent-vs-human and agent-vs-agent match orchestration are planned, not implemented.
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
- Accessible Entire checkpoint link and ID.
- cfo.ai business plan.
- Public repository, demo video, and public build posts.
