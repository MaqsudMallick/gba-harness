# gbharness

Input layer for an agentic **Game Boy Advance** harness: an agent-facing tool surface that
turns high-level button requests into frame-exact input on a real GBA emulator.

The emulator is mGBA, driven as a [libretro](https://github.com/libretro/libretro-common)
core over `ctypes` — so there is no build step, no Python version constraint, and
`retro_run()` is exactly one frame.

Layers, each with its own plan:

| Layer | Status | Plan |
|---|---|---|
| **Input** — button requests → frame-exact presses | built, 266 tests | [`PLAN-input-layer.md`](PLAN-input-layer.md) |
| **Vision** — emulator frames → observations | planned | [`PLAN-vision-layer.md`](PLAN-vision-layer.md) |
| **Agent** — the OpenAI Decisions loop | config done, rest planned | [`PLAN-agent-layer.md`](PLAN-agent-layer.md) |

Dependencies mirror that split: the input layer imports with nothing but the standard
library, and `vision` and `agent` are optional extras.

## Setup

```bash
uv sync --all-extras
./scripts/fetch-core.sh      # mgba_libretro.so into vendor/, no root needed
cp .env.example .env         # then add your OPENAI_API_KEY
uv run pytest
```

`.env` is gitignored and only the agent layer needs a key — the input and vision layers run
without one. A real environment variable always beats `.env`.

The emulator core is a **system artifact, not a Python dependency**, so it is fetched once
rather than installed by `uv sync`. Any of these work, and discovery finds all three:

| How | Where it lands |
|---|---|
| `./scripts/fetch-core.sh` | `vendor/libretro/` — no root needed |
| `sudo apt install libretro-mgba` | a system core directory |
| `export GBHARNESS_LIBRETRO_CORE=...` | wherever you point it |

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

## Checks

```bash
uv run pytest        # 271 tests
uv run ruff check src/ tests/
uv run mypy
```
