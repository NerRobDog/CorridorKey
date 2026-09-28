"""Post-processing of Vision masks into CorridorKey hints (no macOS needed)."""

import numpy as np

from CorridorKeyModule.vision_hint import shape_hint


def test_hint_is_eroded_and_feathered():
    mask = np.zeros((200, 200), np.float32)
    mask[50:150, 50:150] = 1.0
    hint = shape_hint(mask, erode_px=6, feather_px=8)

    assert hint.dtype == np.float32
    assert 0.0 <= hint.min() and hint.max() <= 1.0
    assert hint[100, 100] > 0.99  # solid core kept
    assert hint[100, 51] < 0.5  # edge pulled in by the erosion
    assert 0.0 < hint[100, 56] < 1.0  # soft transition, not a hard step


def test_empty_mask_gives_empty_hint():
    assert shape_hint(np.zeros((50, 50), np.float32)).max() == 0.0
