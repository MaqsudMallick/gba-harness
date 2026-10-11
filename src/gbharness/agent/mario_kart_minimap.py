"""Temporal minimap navigation for Mario Kart: Super Circuit.

The game identifies the local racer by blinking its marker between red and white.
Other racers remain red (or cyan for the linked human player), so a single frame is
ambiguous but a short sequence identifies the controlled kart without emulator memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import atan2, degrees, hypot

import numpy as np

MINIMAP_X = 152
MINIMAP_Y = 78
MINIMAP_WIDTH = 88
MINIMAP_HEIGHT = 82

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class TrackProfile:
    """An ordered minimap centreline, independent of the driving controller."""

    name: str
    points: tuple[Point, ...]
    lookahead: float = 9.0


# Coordinates are relative to the minimap crop. Profiles make another course or game a
# data addition: marker detection and the steering controller do not need to change.
PEACH_CIRCUIT = TrackProfile(
    name="peach-circuit",
    points=(
        (51.0, 42.0),
        (51.0, 27.0),
        (50.0, 19.0),
        (46.0, 18.0),
        (41.0, 25.0),
        (37.0, 37.0),
        (34.0, 53.0),
        (36.0, 63.0),
        (44.0, 73.0),
        (48.0, 73.0),
        (53.0, 65.0),
        (66.0, 65.0),
        (75.0, 62.0),
        (77.0, 54.0),
        (74.0, 52.0),
        (55.0, 52.0),
        (51.0, 48.0),
    ),
)


@dataclass(frozen=True, slots=True)
class MinimapObservation:
    """Position and route-relative steering inferred from visible HUD pixels."""

    position: Point | None
    velocity: Point | None
    steering_error: float | None
    progress: float | None
    moving_forward: bool | None
    confidence: float


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return four-connected components from a tiny binary minimap mask."""
    seen = np.zeros(mask.shape, dtype=np.bool_)
    components: list[list[tuple[int, int]]] = []
    for y_raw, x_raw in zip(*np.nonzero(mask), strict=True):
        y, x = int(y_raw), int(x_raw)
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        component: list[tuple[int, int]] = []
        while stack:
            current_y, current_x = stack.pop()
            component.append((current_y, current_x))
            for delta_y, delta_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_y = current_y + delta_y
                next_x = current_x + delta_x
                if (
                    0 <= next_y < mask.shape[0]
                    and 0 <= next_x < mask.shape[1]
                    and mask[next_y, next_x]
                    and not seen[next_y, next_x]
                ):
                    seen[next_y, next_x] = True
                    stack.append((next_y, next_x))
        components.append(component)
    return components


def _marker_candidates(minimap: np.ndarray) -> tuple[list[Point], list[Point]]:
    """Find red and white 2x2 racer fills surrounded by their dark outline."""
    red = (
        (minimap[:, :, 0] > 145)
        & (minimap[:, :, 0] > minimap[:, :, 1] * 1.45)
        & (minimap[:, :, 0] > minimap[:, :, 2] * 1.30)
        & (minimap[:, :, 1] < 145)
    )
    white = (minimap.min(axis=2) > 185) & (
        minimap.max(axis=2) - minimap.min(axis=2) < 30
    )

    def candidates(mask: np.ndarray) -> list[Point]:
        result: list[Point] = []
        for component in _components(mask):
            if len(component) != 4:
                continue
            ys = [point[0] for point in component]
            xs = [point[1] for point in component]
            if max(xs) - min(xs) != 1 or max(ys) - min(ys) != 1:
                continue
            y_start, y_end = max(0, min(ys) - 2), min(minimap.shape[0], max(ys) + 3)
            x_start, x_end = max(0, min(xs) - 2), min(minimap.shape[1], max(xs) + 3)
            dark_outline = minimap[y_start:y_end, x_start:x_end].max(axis=2) < 70
            if int(dark_outline.sum()) < 5:
                continue
            result.append((sum(xs) / 4, sum(ys) / 4))
        return result

    return candidates(red), candidates(white)


def _distance(first: Point, second: Point) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


@dataclass(slots=True)
class MarioKartMinimapNavigator:
    """Track the blinking local marker and aim along an ordered course profile."""

    profile: TrackProfile = PEACH_CIRCUIT
    position: Point | None = None
    missing_frames: int = 0
    completed_laps: int = 0
    _last_progress: float | None = field(default=None, init=False, repr=False)
    _pending_white: deque[tuple[int, Point]] = field(
        default_factory=lambda: deque(maxlen=32), init=False, repr=False
    )
    _history: deque[tuple[int, Point]] = field(
        default_factory=lambda: deque(maxlen=90), init=False, repr=False
    )

    def reset(self) -> None:
        self.position = None
        self.missing_frames = 0
        self.completed_laps = 0
        self._last_progress = None
        self._pending_white.clear()
        self._history.clear()

    def _acquire_or_track(
        self, red: list[Point], white: list[Point], frame: int
    ) -> Point | None:
        candidates = red + white
        if self.position is not None and candidates:
            previous = self.position
            nearest = min(candidates, key=lambda point: _distance(point, previous))
            if _distance(nearest, previous) <= 3.5:
                self.missing_frames = 0
                return nearest

        if self.position is not None:
            self.missing_frames += 1
            if self.missing_frames <= 30:
                return self.position
            self.position = None
            self._history.clear()

        # A white candidate followed by red at the same location is the local racer's
        # blink. Static white route pixels never satisfy this colour transition.
        for candidate in red:
            matches = [
                pending
                for pending_frame, pending in self._pending_white
                if frame - pending_frame <= 24 and _distance(candidate, pending) <= 3.5
            ]
            if matches:
                self.missing_frames = 0
                return candidate

        for candidate in white:
            self._pending_white.append((frame, candidate))
        while self._pending_white and frame - self._pending_white[0][0] > 24:
            self._pending_white.popleft()
        return None

    def _route_projection(self, point: Point) -> tuple[float, Point, float]:
        points = self.profile.points
        lengths = [
            _distance(points[index], points[(index + 1) % len(points)])
            for index in range(len(points))
        ]
        total = sum(lengths)
        best_distance = float("inf")
        best_progress = 0.0
        best_projection = points[0]
        travelled = 0.0
        for index, length in enumerate(lengths):
            start = points[index]
            end = points[(index + 1) % len(points)]
            delta_x, delta_y = end[0] - start[0], end[1] - start[1]
            denominator = delta_x * delta_x + delta_y * delta_y
            fraction = 0.0
            if denominator:
                fraction = max(
                    0.0,
                    min(
                        1.0,
                        ((point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y)
                        / denominator,
                    ),
                )
            projection = (start[0] + fraction * delta_x, start[1] + fraction * delta_y)
            distance = _distance(point, projection)
            if distance < best_distance:
                best_distance = distance
                best_progress = (travelled + fraction * length) / total
                best_projection = projection
            travelled += length
        return best_progress, best_projection, total

    def _point_at(self, progress: float, total: float) -> Point:
        points = self.profile.points
        target = (progress % 1.0) * total
        travelled = 0.0
        for index in range(len(points)):
            start = points[index]
            end = points[(index + 1) % len(points)]
            length = _distance(start, end)
            if travelled + length >= target:
                fraction = (target - travelled) / max(length, 0.001)
                return (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            travelled += length
        return points[0]

    def update(self, rgb: np.ndarray, frame: int) -> MinimapObservation:
        if rgb.shape != (160, 240, 3):
            raise ValueError(f"expected a 160x240 RGB frame, got {rgb.shape!r}")
        minimap = rgb[
            MINIMAP_Y : MINIMAP_Y + MINIMAP_HEIGHT,
            MINIMAP_X : MINIMAP_X + MINIMAP_WIDTH,
        ].astype(np.int16)
        red, white = _marker_candidates(minimap)
        tracked = self._acquire_or_track(red, white, frame)
        if tracked is None:
            return MinimapObservation(None, None, None, None, None, 0.0)

        if self.position is None or _distance(tracked, self.position) > 0.1:
            self._history.append((frame, tracked))
        self.position = tracked
        progress, _, total = self._route_projection(tracked)
        if (
            self._last_progress is not None
            and self._last_progress > 0.82
            and progress < 0.18
        ):
            self.completed_laps += 1
        self._last_progress = progress

        recent = [item for item in self._history if frame - item[0] <= 60]
        if len(recent) < 2:
            return MinimapObservation(tracked, None, None, progress, None, 0.35)
        origin = recent[0][1]
        velocity = (tracked[0] - origin[0], tracked[1] - origin[1])
        speed = hypot(*velocity)
        if speed < 1.5:
            return MinimapObservation(tracked, None, None, progress, None, 0.45)

        target = self._point_at(progress + self.profile.lookahead / total, total)
        desired = (target[0] - tracked[0], target[1] - tracked[1])
        cross = velocity[0] * desired[1] - velocity[1] * desired[0]
        dot = velocity[0] * desired[0] + velocity[1] * desired[1]
        steering_error = degrees(atan2(cross, dot))
        moving_forward = dot > 0
        confidence = min(1.0, 0.55 + speed / 12)
        return MinimapObservation(
            tracked,
            velocity,
            steering_error,
            progress,
            moving_forward,
            confidence,
        )
