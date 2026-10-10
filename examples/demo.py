"""Drive the input harness against a real GBA emulator and show what the game sees.

Run with::

    uv run python examples/demo.py

The cartridge is `keyinput_rom.py` -- a hand-assembled ROM that writes one indicator pixel
per button every frame. So everything printed below is read back out of the emulator's
framebuffer: it is the game's own view of the controller, not this package's bookkeeping.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from keyinput_rom import KEYINPUT_BITS, PROGRAM, build_rom  # noqa: E402

from gbharness.actions import hold, press, release, repeat, tap, wait  # noqa: E402
from gbharness.backends import LibretroInputPort, find_core  # noqa: E402
from gbharness.buttons import CANON_ORDER, Button  # noqa: E402
from gbharness.safety import BudgetExhausted, CallTooLarge, Limits  # noqa: E402
from gbharness.session import InputSession  # noqa: E402
from gbharness.timing import FRAME_RATE  # noqa: E402

GLYPH = {
    "up": "^", "down": "v", "left": "<", "right": ">",
    "a": "A", "b": "B", "l": "L", "r": "R", "start": "S", "select": "s",
}
DIM, GREEN, BOLD, CYAN, YELLOW, RED, OFF = (
    "\033[90m", "\033[92m", "\033[1m", "\033[96m", "\033[93m", "\033[91m", "\033[0m",
)
WHITE_PIXEL = 0x4000  # indicator pixels are 0xFFFF or 0x0000; any midpoint works


class GameView:
    """Decodes the cartridge's indicator pixels into the button set it observed."""

    def __init__(self) -> None:
        self.per_frame: list[frozenset[Button]] = []

    def sink(self, ptr: int, width: int, height: int, pitch: int) -> None:
        row = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16))[0:10]
        self.per_frame.append(
            frozenset(
                Button(KEYINPUT_BITS[i]) for i, v in enumerate(row) if v > WHITE_PIXEL
            )
        )

    def reset(self) -> None:
        self.per_frame.clear()

    def timeline(self, limit: int = 40) -> str:
        """One character per frame: the button seen, a digit for a combo, dot for none."""
        out = []
        for seen in self.per_frame[:limit]:
            if not seen:
                out.append(f"{DIM}.{OFF}")
            elif len(seen) == 1:
                out.append(f"{GREEN}{GLYPH[next(iter(seen)).value]}{OFF}")
            else:
                out.append(f"{GREEN}{len(seen)}{OFF}")
        if len(self.per_frame) > limit:
            out.append(f"{DIM} … {len(self.per_frame)} frames{OFF}")
        return "".join(out)

    def peak(self) -> frozenset[Button]:
        return max(self.per_frame, key=len, default=frozenset())


def controller(seen: frozenset[Button]) -> str:
    cells = []
    for button in CANON_ORDER:
        glyph = GLYPH[button.value]
        if button in seen:
            cells.append(f"\033[42;30m {glyph} {OFF}")
        else:
            cells.append(f"\033[100;90m {glyph} {OFF}")
    return "".join(cells)


def rule(title: str = "") -> None:
    print(f"\n{CYAN}{'─' * 4} {title} {'─' * max(0, 66 - len(title))}{OFF}")


def main() -> int:
    try:
        core = find_core()
    except Exception as exc:
        print(f"{RED}{exc}{OFF}")
        return 1

    view = GameView()
    print(f"{BOLD}gbharness — Game Boy Advance input layer{OFF}")
    print(f"  emulator  {core}")
    print(f"  cartridge hand-assembled KEYINPUT reporter, {len(PROGRAM)} instructions")
    print(f"  timing    {FRAME_RATE:.4f} fps, {1000 / FRAME_RATE:.3f} ms per frame")
    print("\n  Everything below is read back out of the emulator's framebuffer:")
    print(f"  it is {BOLD}what the game saw{OFF}, not what we think we sent.")

    with (
        LibretroInputPort(build_rom(), core=core, frame_sink=view.sink) as port,
        InputSession(port, Limits(episode_frame_budget=1200)) as session,
    ):

        def step(label: str, action, note: str = "") -> None:
            view.reset()
            start = session.frame
            result = session.submit(action)
            print(f"  {DIM}f{start:<5}{OFF} {label:<26} {view.timeline()}")
            print(f"         {' ' * 26} {controller(view.peak())}  {DIM}{note}{OFF}")
            for warning in result.warnings:
                print(f"         {' ' * 26} {YELLOW}! {warning}{OFF}")

        # The framebuffer is undefined until the cartridge paints it for the first time,
        # so the first frames read as garbage (all indicators white). Burn them before
        # believing anything the screen says -- an observation layer will need the same
        # guard.
        session.submit(wait(6))
        view.reset()

        rule("a tap is two frames held, then a gap")
        step("wait(6)", wait(6), "idle: nothing held")
        step("tap('start')", tap("start"), "2 frames pressed, 6 frames gap")
        step("tap('a', 5, 3)", tap("a", 5, 3), "hold 5, gap 3 — count the characters")

        rule("repeats are spaced, not merged")
        step("repeat(tap('a'), 3)", repeat(tap("a"), 3), "three distinct presses")
        step("repeat(tap('b',1,0),3)", repeat(tap("b", 1, 0), 3),
             "gap=0: three taps the game cannot tell from one hold")

        rule("combos are simultaneous, not staggered")
        step("hold(['l','r'], 4)", hold(["l", "r"], 4), "an L+R chord: 2 buttons per frame")
        step("hold(all ten, 3)", hold([b.value for b in CANON_ORDER], 3),
             "every input the GBA has")

        rule("hold B to run, tap A to jump — B must survive")
        step("press('b')", press("b"), "open-ended hold")
        step("tap('a', 3, 2)", tap("a", 3, 2), "A taps; B stays down the whole time")
        step("release()", release(), "let everything up")

        rule("guards report back instead of failing silently")
        step("tap('a', hold_frames=0)", tap("a", hold_frames=0),
             "0 frames is unobservable — clamped to 2")
        step("hold(['left','right'],3)", hold(["left", "right"], 3),
             "opposing D-pad: hardware allows, games glitch")

        step("wait(5000)", wait(5000), "one huge value: clamped, and it says so")

        rule("a shape too big to clamp is refused, not truncated")
        print(f"  {DIM}f{session.frame:<5}{OFF} repeat(wait(60), 1000)")
        try:
            session.submit(repeat(wait(60), 1000))
        except CallTooLarge as exc:
            print(f"         {RED}refused  {OFF}{exc}")
        print(f"         {DIM}every value is in range; only the total is not, "
              f"so clamping cannot help{OFF}")

        rule("the episode budget stops a runaway loop")
        calls = 0
        while True:
            try:
                session.submit(wait(300))
                calls += 1
            except BudgetExhausted as exc:
                print(f"  {DIM}f{session.frame:<5}{OFF} after {calls} more wait(300) calls")
                print(f"         {RED}refused  {OFF}{exc}")
                break

        rule("totals")
        print(f"  session advanced   {session.frame} frames "
              f"({session.frame / FRAME_RATE:.2f}s of game time)")
        print(f"  core actually ran  {port.frames_run} frames")
        print(f"  core input polls   {port.polls}")
        print(f"  buttons still held {sorted(b.value for b in session.pressed()) or 'none'}")
        print(f"  budget remaining   {session.budget.remaining}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
