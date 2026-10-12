# neatHack submission pack

## Project

**gbharness Arena — Observable Visual Game Agents**

gbharness Arena is an open-source runtime for visual game agents on real emulators. An agent reads emulator frames, selects bounded controller actions, sends frame-exact input, and verifies results from the game screen.

## Demo result

The local visual Mario controller completed Super Mario Bros. World 1-1 in 126 control turns and 2,928 emulated frames. The run verified flag touch, castle entry, and the World 1-2 card.

Live recording: `artifacts/submission/mario-live-demo.mp4`

Neatlogs trace: `7d385d54ecf7e20640a3906edc5fcea4`

## Architecture

1. mGBA runs the game cartridge.
2. Vision converts each emulator frame to an observation.
3. A game policy limits the action set.
4. The controller sends frame-exact input.
5. The runner verifies game progress and completion.
6. neatlogs records the decision loop and recovery data.

## Hackathon tools

- **neatlogs:** Records workflow, agent, tool, LLM, and guardrail spans. Each trace stores observations, actions, inputs, outputs, latency, status, and recovery data.
- **Entire:** Preserves an auditable development checkpoint. Checkpoint ID: `01M4K4DYFVKB5TF5R77KZMJD5J`. Checkpoint URL: https://entire.io/gh/MaqsudMallick/gba-harness/commit/c47ae93c9583137756b38c9a89c1b5cba36f04c7
- **cfo.ai:** Models pricing, cost, revenue, and runway for hosted game-agent matches. Add the published cfo.ai plan URL to the submission form.

## Links

- Repository: https://github.com/MaqsudMallick/gba-harness
- Official launch post: https://x.com/bettersayAJ/status/2109639214723932495
