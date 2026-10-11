from __future__ import annotations

import pytest

from gbharness.vision.capture import RawFrame
from gbharness.vision.multiplayer import crop_frame, split_player_frames


def _frame(width: int, height: int) -> RawFrame:
    pitch = width * 2 + 4
    rows = []
    for y in range(height):
        visible = b"".join(
            ((y * width + x) & 0xFFFF).to_bytes(2, "little") for x in range(width)
        )
        rows.append(visible + b"PAD!")
    return RawFrame(b"".join(rows), width, height, pitch, 7)


def test_crop_frame_respects_source_pitch() -> None:
    cropped = crop_frame(_frame(4, 3), 1, 1, 2, 2)
    values = [
        int.from_bytes(cropped.data[i : i + 2], "little")
        for i in range(0, len(cropped.data), 2)
    ]
    assert values == [5, 6, 9, 10]
    assert cropped.pitch == 4
    assert cropped.index == 7


@pytest.mark.parametrize(
    ("players", "size"),
    [(2, (480, 160)), (3, (480, 320)), (4, (480, 320))],
)
def test_split_auto_layout_returns_native_player_frames(
    players: int, size: tuple[int, int]
) -> None:
    result = split_player_frames(_frame(*size), players)
    assert len(result) == players
    assert {(frame.width, frame.height) for frame in result} == {(240, 160)}


def test_split_rejects_non_auto_layout_dimensions() -> None:
    with pytest.raises(ValueError, match="auto layout"):
        split_player_frames(_frame(480, 320), 2)
