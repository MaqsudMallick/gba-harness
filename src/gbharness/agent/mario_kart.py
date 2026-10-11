"""Setup automation and a local visual driver for Mario Kart: Super Circuit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from gbharness.agent.multiplayer import PlayerDecision
from gbharness.buttons import Button
from gbharness.multiplayer import LinkedSession

KART_X = 120
ROAD_AIM_X = 140
ROAD_SCAN_ROWS = (90, 102, 114, 126)
ROAD_X_MIN = 36
ROAD_X_MAX = 188


def open_linked_lobby(session: LinkedSession) -> None:
    """Navigate every cartridge to the verified Multi-Pak 2-4 player lobby.

    This is setup automation, not a racing policy. It reproduces the game's documented
    multiplayer flow so a live demo starts at the shared Mario GP / VS / Battle menu.
    """
    everyone = range(session.port.players)

    def tap_all(button: Button, settle: int) -> None:
        session.step({player: (button,) for player in everyone}, 3)
        session.step({player: () for player in everyone}, settle)

    session.step({}, 900)
    tap_all(Button.DOWN, 12)  # SINGLE PLAYER -> MULTIPLAYER
    tap_all(Button.A, 180)
    tap_all(Button.A, 300)  # MULTI-PAK LINK
    tap_all(Button.UP, 12)  # cable confirmation: NO -> YES
    tap_all(Button.A, 1200)


def start_quick_race(
    session: LinkedSession,
    player_one_view: Callable[[], np.ndarray] | None = None,
) -> None:
    """Choose default racers and Mushroom Cup from the linked game lobby.

    The sequence deliberately ends before driving begins. It exists to make autonomous
    evaluation repeatable; the visual agent below controls only the race itself.
    """

    def tap(player: int, button: Button, settle: int) -> None:
        session.step({player: (button,)}, 3)
        session.step({player: ()}, settle)

    tap(0, Button.A, 240)  # Mario GP -> character select
    tap(0, Button.A, 30)  # Player 1 chooses the default/random racer
    tap(0, Button.A, 30)  # Player 1 confirms
    tap(1, Button.A, 360)  # Player 2 resolves the random/default racer
    tap(1, Button.A, 600)  # Player 2 confirms -> cup select transition

    # Link timing varies slightly across host/core builds. Pulse the host's confirm key
    # until the visual race detector sees the starting grid instead of assuming the cup
    # menu became interactive on one exact frame.
    for _ in range(12):
        tap(0, Button.A, 90)
        if player_one_view is not None and observe_mario_kart(player_one_view()).racing:
            return


@dataclass(frozen=True, slots=True)
class MarioKartObservation:
    """Track geometry visible to one racer, without reading emulator memory."""

    racing: bool
    road_center: float | None
    far_center: float | None
    heading: float | None
    road_width: int
    confidence: float
    on_road: bool
    grass_fraction: float
    reverse: bool


def _road_span(mask: np.ndarray, *, max_gap: int = 18) -> tuple[int, int] | None:
    """Merge lane-marking and sprite gaps into one plausible road span."""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return None

    candidates: list[tuple[int, int, int]] = []
    group_start = int(starts[0])
    group_end = int(ends[0])
    pixels = group_end - group_start
    for start_raw, end_raw in zip(starts[1:], ends[1:], strict=True):
        start, end = int(start_raw), int(end_raw)
        if start - group_end <= max_gap:
            group_end = end
            pixels += end - start
        else:
            candidates.append((group_start, group_end, pixels))
            group_start, group_end, pixels = start, end, end - start
    candidates.append((group_start, group_end, pixels))

    plausible = [
        item
        for item in candidates
        if item[1] - item[0] >= 18
        and item[2] / (item[1] - item[0]) >= 0.55
    ]
    if not plausible:
        return None
    start, end, _ = max(
        plausible,
        key=lambda item: item[2] - abs((item[0] + item[1]) / 2 - KART_X) * 0.08,
    )
    return start, end


def _surface_cues(pixels: np.ndarray) -> tuple[float, bool]:
    """Return near-wheel grass coverage and the game's REVERSE warning state."""
    near_surface = np.concatenate(
        (pixels[132:154, 70:105], pixels[132:154, 135:170]), axis=1
    )
    grass = (
        (near_surface[:, :, 1] > near_surface[:, :, 0] * 1.15)
        & (near_surface[:, :, 1] > near_surface[:, :, 2] * 1.08)
        & (near_surface[:, :, 1] > 110)
    )
    reverse_region = pixels[35:85, 80:175]
    reverse_red = (
        (reverse_region[:, :, 0] > 180)
        & (reverse_region[:, :, 1] < 110)
        & (reverse_region[:, :, 2] < 110)
    )
    reverse_white = np.all(reverse_region > 220, axis=2)
    reverse = int(reverse_red.sum()) > 197 and int(reverse_white.sum()) > 300
    return float(grass.mean()), reverse


def observe_mario_kart(rgb: np.ndarray) -> MarioKartObservation:
    """Estimate the road's near-field center from the native 240x160 view.

    Super Circuit renders the drivable surface as a broad, low-chroma Mode-7 region.
    Sampling several rows makes the cue tolerate lane markings, sprites, and items. This
    deliberately uses only pixels available to a human player.
    """
    if rgb.shape != (160, 240, 3):
        raise ValueError(f"expected a 160x240 RGB frame, got {rgb.shape!r}")

    pixels = rgb.astype(np.int16)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    light = pixels.mean(axis=2)
    road = (chroma <= 34) & (light >= 52) & (light <= 224)
    grass_fraction, reverse = _surface_cues(pixels)

    samples: list[tuple[int, float, int, int, int]] = []
    for row in ROAD_SCAN_ROWS:
        span = _road_span(road[row, ROAD_X_MIN:ROAD_X_MAX])
        if span is None:
            continue
        start, end = span
        start += ROAD_X_MIN
        end += ROAD_X_MIN
        width = end - start
        samples.append((row, (start + end) / 2, width, start, end))

    if not samples:
        return MarioKartObservation(
            False,
            None,
            None,
            None,
            0,
            0.0,
            False,
            grass_fraction,
            reverse,
        )

    near = samples[-2:]
    far = samples[:2]
    near_center = float(np.average([sample[1] for sample in near]))
    far_center = float(np.average([sample[1] for sample in far]))
    heading = far_center - near_center
    width = max(sample[2] for sample in samples)
    confidence = min(1.0, len(samples) / len(ROAD_SCAN_ROWS) * width / 95)
    on_road_votes = sum(start <= KART_X <= end for _, _, _, start, end in near)
    return MarioKartObservation(
        len(samples) >= 2,
        near_center,
        far_center,
        heading,
        width,
        confidence,
        on_road_votes >= 1 and grass_fraction < 0.28,
        grass_fraction,
        reverse,
    )


@dataclass(slots=True)
class MarioKartVisionAgent:
    """Closed-loop Player 2 driver: accelerate and steer toward visible road."""

    decision_interval: int = 1
    dead_zone: float = 14.0
    smoothed_center: float = ROAD_AIM_X
    smoothed_heading: float = 0.0
    previous_error: float = 0.0
    last_steer: Button | None = None
    recovery_steer: Button | None = None
    lost_frames: int = 0
    name: str = field(default="mario-kart-local-vision", init=False)

    def reset(self) -> None:
        self.smoothed_center = ROAD_AIM_X
        self.smoothed_heading = 0.0
        self.previous_error = 0.0
        self.last_steer = None
        self.recovery_steer = None
        self.lost_frames = 0

    def decide(self, rgb: np.ndarray, frame: int) -> PlayerDecision:
        observation = observe_mario_kart(rgb)
        buttons = {Button.A}
        if (
            observation.racing
            and observation.road_center is not None
            and observation.heading is not None
        ):
            self.lost_frames = 0
            self.recovery_steer = None
            self.smoothed_center = (
                self.smoothed_center * 0.72 + observation.road_center * 0.28
            )
            self.smoothed_heading = (
                self.smoothed_heading * 0.7 + observation.heading * 0.3
            )
            lateral_error = self.smoothed_center - ROAD_AIM_X
            derivative = lateral_error - self.previous_error
            self.previous_error = lateral_error
            control = lateral_error + self.smoothed_heading + derivative * 0.35

            if observation.reverse:
                recovery_target = observation.far_center or observation.road_center
                control = recovery_target - ROAD_AIM_X
                if abs(control) < self.dead_zone:
                    control = -30 if self.last_steer == Button.LEFT else 30
                scene = "reverse-recovery"
                duty = 8
            elif not observation.on_road:
                recovery_target = observation.far_center or observation.road_center
                control = recovery_target - ROAD_AIM_X
                scene = "off-road-recovery"
                duty = 8
            else:
                scene = "race"
                magnitude = abs(control)
                duty = (
                    0
                    if magnitude < self.dead_zone
                    else 1
                    if magnitude < 25
                    else 2
                    if magnitude < 38
                    else 3
                    if magnitude < 55
                    else 4
                )
                if abs(self.smoothed_heading) >= 30:
                    duty = max(duty, 8)
                elif abs(self.smoothed_heading) >= 18:
                    duty = max(duty, 5)

            steer = Button.RIGHT if control < 0 else Button.LEFT
            phase = (frame // self.decision_interval) % 12
            # Digital steering at full throttle is extremely twitchy in Super Circuit.
            # Lift for half of each control cycle on a visible bend, like a human easing
            # off the accelerator before turning in.
            if (
                observation.on_road
                and abs(observation.heading) >= 16
                and phase >= (4 if abs(observation.heading) >= 28 else 7)
            ):
                buttons.discard(Button.A)
            if duty and phase < duty:
                self.last_steer = steer
                buttons.add(steer)
            return PlayerDecision(
                frozenset(buttons),
                scene,
                note=(
                    f"near={observation.road_center:.1f}, "
                    f"heading={observation.heading:+.1f}, control={control:+.1f}, "
                    f"duty={duty}/12, grass={observation.grass_fraction:.2f}, "
                    f"confidence={observation.confidence:.2f}"
                ),
            )

        if self.lost_frames == 0:
            if self.last_steer == Button.LEFT:
                self.recovery_steer = Button.RIGHT
            elif self.last_steer == Button.RIGHT:
                self.recovery_steer = Button.LEFT
        self.lost_frames += self.decision_interval
        if self.recovery_steer is None:
            phase = (frame // 90) % 2
            self.recovery_steer = Button.LEFT if phase == 0 else Button.RIGHT
        elif self.lost_frames % 120 == 0:
            self.recovery_steer = (
                Button.RIGHT if self.recovery_steer == Button.LEFT else Button.LEFT
            )
        recovery_duty = 9 if self.lost_frames <= 120 else 6
        if (frame // self.decision_interval) % 12 < recovery_duty:
            buttons.add(self.recovery_steer)
        return PlayerDecision(
            frozenset(buttons),
            "recovering",
            note="Road not visible; counter-steer/search until it re-enters the view.",
        )
