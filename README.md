# gbharness

A modular visual-game agent harness: model providers observe emulator frames, choose from a
game-defined action space, and produce frame-exact input on a real GBA emulator.

The emulator is mGBA, driven as a [libretro](https://github.com/libretro/libretro-common)
core over `ctypes` — so there is no build step, no Python version constraint, and
`retro_run()` is exactly one frame.

The layers deliberately depend inward, so changing the game or model does not rewrite the
emulator:

| Layer | Responsibility | Status |
|---|---|---|
| **Input** | bounded actions → frame-exact controller state | built |
| **Vision** | latest emulator framebuffer → PNG/RGB observation | built |
| **Game policy** | goal, allowed macros, and descriptions | Mario starter policy built |
| **Decision provider** | screenshot + state + action set → one decision | Vertex AI built; provider protocol is portable |
| **Runner** | observe → decide → act → recover → verify | built |
| **Multiplayer** | 2-4 linked consoles, synchronized agents, split observations | built |

Dependencies mirror that split: the input layer imports with nothing but the standard
library, and `vision` and `agent` are optional extras.

## Setup

```bash
uv sync --all-extras
./scripts/fetch-core.sh      # mGBA libretro core into vendor/, no root needed
cp .env.example .env         # then configure neatlogs and Vertex for agent runs
uv run pytest
```

For autonomous play, fill in `NEATLOGS_API_KEY`, `GOOGLE_CLOUD_PROJECT`, and optionally
`GOOGLE_CLOUD_LOCATION` in the gitignored `.env`, then configure Google Application Default
Credentials. Human-controlled emulation and tests do not need cloud credentials. A real
environment variable always beats `.env`.

The emulator core is a **system artifact, not a Python dependency**, so it is fetched once
rather than installed by `uv sync`. Any of these work, and discovery finds all three:

| How | Where it lands |
|---|---|
| `./scripts/fetch-core.sh` | `vendor/libretro/` — no root needed |
| `sudo apt install libretro-mgba` | a system core directory on Debian/Ubuntu |
| `export GBHARNESS_LIBRETRO_CORE=...` | wherever you point it |

`fetch-core.sh` supports Apple Silicon, Intel macOS, and Debian/Ubuntu Linux. On macOS it
downloads the official Libretro nightly core; on Linux it extracts the distribution package.

Linked multiplayer uses the separate
[mGBA Splitscreen](https://docs.libretro.com/library/mgba_splitscreen/) core. Build its
pinned open-source revision locally (CMake is required) with:

```bash
./scripts/fetch-multiplayer-core.sh
```

Integration tests skip cleanly if no core is found; `examples/demo.py` prints the three
options and exits.

## The ten inputs

`up` `down` `left` `right` `a` `b` `l` `r` `start` `select`

That is the whole vocabulary. Keyboard names (WASD, ZXCV) are an *emulator binding*, not
hardware, so they are never accepted by default — `l`/`r` are the **shoulder buttons**, while
the directions are `left`/`right`. Opt into a keyboard dialect explicitly if a human needs it:

```python
parse("a")                  # Button.A     the A face button
parse("a", dialect="wasd")  # Button.LEFT  the D-pad direction
parse("a", dialect="mgba")  # Button.L     the left shoulder
```

## Usage

```python
from gbharness.actions import hold, press, release, repeat, tap, wait
from gbharness.backends import LibretroInputPort
from gbharness.session import InputSession

with (
    LibretroInputPort("roms/game.gba") as port,
    InputSession(port) as session,
):
    session.submit(wait(120))                  # let it boot
    session.submit(tap("start"))               # title screen
    session.submit(repeat(tap("a"), 3))        # through some dialogue
    session.submit(press("r"))                 # hold R to dash...
    session.submit(hold("right", 90))          # ...while running right
    session.submit(release())                  # let everything up

    result = session.submit(tap(["l", "r"]))   # an L+R chord
    print(result.frames_advanced, result.warnings)
```

Time is always in **frames** (59.7275 fps, ~16.74 ms each), never milliseconds. Games read
the joypad once per frame, so a sub-frame press can be missed entirely — the minimum safe tap
is 2 frames, and `tap()` adds a gap after it so menu debouncing doesn't merge consecutive taps.

## See it work

```bash
uv run python examples/demo.py
```

Drives the full stack against the real emulator and prints, frame by frame, **what the game
saw** — read back out of the emulator's framebuffer rather than from this package's own
bookkeeping. The cartridge is `examples/keyinput_rom.py`: a hand-assembled 18-instruction
GBA ROM that paints one indicator pixel per button every frame.

That ROM is also the test suite's oracle. libretro doesn't expose KEYINPUT for read-back, so
a cartridge that reports what it saw is what makes `tests/test_end_to_end.py` able to verify
the whole chain — session → safety → scheduler → libretro callback → ARM7 → VRAM.

To smoke-test a cartridge you own, put its `.gba` file under the gitignored `roms/`
directory and run:

```bash
uv run python examples/run_rom.py --rom roms/game.gba --screenshot artifacts/boot.png
```

The runner validates the cartridge before giving it to the emulator, boots it, taps Start,
advances the requested number of frames, and writes a screenshot. ROM images are deliberately
never downloaded or committed by this project.

To include a simple gameplay-input smoke test, hold Right and jump before capturing:

```bash
uv run python examples/run_rom.py --rom roms/game.gba \
  --move-right-frames 90 --jump --screenshot artifacts/gameplay.png
```

## Live emulator window

The same Libretro core can be driven interactively in a live window—there is no separate
emulator process and no drift between what the viewer sees and what the harness controls:

```bash
uv run python examples/live_rom.py
```

Controls: arrow keys for the D-pad, `Z`/`X` for A/B, `A`/`S` for L/R, Enter for Start,
Right Shift for Select, and Escape to quit.

## Linked multiplayer

`LibretroMultiplayerPort` runs two to four GBA instances inside one core and connects them
through mGBA's lockstep link-cable driver. Controller ports are independent, while
`LinkedSession.step()` applies every agent's inputs before advancing the one shared clock.
The composite framebuffer can be converted back into native 240x160 observations with
`split_player_frames()`, so each agent sees only its own console.

The supplied Mario Kart launcher can boot directly into the game's confirmed `2P` linked
lobby. Use only a cartridge image you own; ROMs remain in the gitignored `roms/` directory.

```bash
./scripts/fetch-multiplayer-core.sh
uv run python examples/multiplayer_rom.py \
  --rom "roms/Mario Kart - Super Circuit (USA).gba" \
  --players 2 --mario-kart-lobby
```

P1 uses arrows, N/M for A/B, K/L for the shoulders, and Enter for Start. P2 uses
WASD, F/G for A/B, Q/E for the shoulders, and Space for Start. The backend itself supports
four controllers; the initial keyboard frontend maps two because the next milestone is to
assign P3/P4 to gamepads or autonomous policies.

To race against the local visual driver, let the launcher choose the default racers and
Mushroom Cup, then assign Player 2 to the Mario Kart adapter:

```bash
uv run python examples/multiplayer_rom.py \
  --rom "roms/Mario Kart - Super Circuit (USA).gba" \
  --players 2 --quick-race --agent mario-kart --agent-player 2
```

Player 1 is **you**: hold `N` to accelerate and use the arrow keys to steer. Player 2 is
the agent; its keyboard mapping is disabled while autonomous control is active. The status
panel shows the exact P1 input reaching the cartridge and the agent's current scene and
buttons. The Mario Kart adapter reads only P2's 240x160 view, estimates near/far road
geometry, perspective, grass coverage, and the game's wrong-way warning. A temporal
minimap navigator identifies P2 from its red/white blinking marker, estimates motion along
an injectable course profile, and looks ahead around bends. Minimap guidance anticipates
turns but is gated when it conflicts with a strong near-field road observation. The driver
also preserves turn direction when the road leaves view, realigns after crossing back onto
asphalt, and uses a bounded reverse/forward escape when motion stalls.
`AutonomousPlayer` owns cropping and decision cadence, so another game can provide a
`VisualPlayerAgent` without changing the emulator or frontend.

The deterministic headless evaluation completed all three laps on Peach Circuit in
`3:55.98` (lap times `1:19.58`, `1:10.70`, and `1:25.70`) using visual input only. This is
`1:50.77` faster than the road-view-only controller's `5:46.75` benchmark.

For a fast evidence capture without a window:

```bash
uv run python examples/multiplayer_rom.py \
  --rom "roms/Mario Kart - Super Circuit (USA).gba" \
  --players 2 --mario-kart-lobby --headless --max-frames 1 \
  --screenshot artifacts/mario-kart-linked-lobby.png
```

## Autonomous agent

The default Mario controller is a local closed-loop visual skill. It reads emulator frames,
reacts to enemies, pipes, pits, and staircases, and confirms completion only after the flag,
castle, and next-level card appear. It does not read game RAM or restore emulator state.
This keeps high-frequency platforming control local while Neatlogs records the episode.

Vertex AI remains available as the general visual decision provider. Its structured output
must select one key from the finite action vocabulary. `DecisionProvider` is vendor-neutral,
and `GamePolicy` is game-neutral, so another game can provide a different goal and action
table without changing the runner.

```bash
# Local Mario skill: only NEATLOGS_API_KEY is required for tracing.
uv run python -m neatlogs doctor --probe --json
uv run python examples/agent_rom.py --controller local --max-turns 150

# General cloud vision agent: also requires GOOGLE_CLOUD_PROJECT and ADC.
uv run python examples/agent_rom.py --controller vertex --max-turns 100
```

The Pygame window says **AGENT CONTROL**, disables keyboard gameplay input, and displays the
latest action, confidence, scene, and progress estimate. Use `--headless` for evaluation
runs. Each run is grouped in neatlogs as `visual-game-agent`, including agent decisions,
controller actions, state changes, errors, recovery attempts, and Vertex calls when used.
The final console summary includes `neatlogs_trace_id`, including on failed runs, so the
exact dashboard trace can be attached to a bug report or hackathon submission.

The Mario action table in `src/gbharness/agent/policy.py` is only a starter adapter. Add a
new `GamePolicy` for another game; no provider or emulator changes are required.

## Hackathon evidence

The project follows the official neatHack build-log workflow in [`hackathon.md`](hackathon.md).
That file distinguishes completed checks from credential- or account-dependent evidence.
ROMs, API keys, private traces, and account information are never committed.

## Checks

```bash
uv run pytest
uv run ruff check src/ tests/ examples/
uv run mypy
```
