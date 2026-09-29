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


def _plate(screen=(0.15, 0.55, 0.15), size=(270, 480)):
    img = np.empty((*size, 3), np.float32)
    img[:] = screen[::-1]  # BGR, as OpenCV reads it
    return img


def test_screen_mask_keeps_subject_drops_screen():
    from CorridorKeyModule.vision_hint import screen_mask

    img = _plate()
    img[80:200, 150:300] = (0.1, 0.1, 0.6)  # red subject (BGR)
    m = screen_mask(img, "green")
    assert m[140, 225] == 1.0
    assert m[20, 20] == 0.0 and m[140, 400] == 0.0


def test_screen_mask_ignores_letterbox_bars():
    from CorridorKeyModule.vision_hint import screen_mask

    img = _plate()
    img[:30] = 0.0
    img[-30:] = 0.0
    img[100:200, 150:300] = (0.1, 0.1, 0.6)
    m = screen_mask(img, "green")
    assert m[:30].max() == 0.0 and m[-30:].max() == 0.0
    assert m[150, 225] == 1.0


def test_screen_mask_drops_small_specks():
    from CorridorKeyModule.vision_hint import screen_mask

    img = _plate()
    img[50:54, 50:54] = 0.9  # tracking marker
    img[100:200, 150:300] = (0.1, 0.1, 0.6)
    m = screen_mask(img, "green")
    assert m[52, 52] == 0.0
    assert m[150, 225] == 1.0


def test_screen_mask_blue_screen():
    from CorridorKeyModule.vision_hint import screen_mask

    img = _plate(screen=(0.1, 0.2, 0.6))
    img[100:200, 150:300] = (0.1, 0.1, 0.6)  # red subject
    m = screen_mask(img, "blue")
    assert m[150, 225] == 1.0 and m[20, 20] == 0.0


def test_screen_mask_accepts_uint16():
    from CorridorKeyModule.vision_hint import screen_mask

    img = _plate()
    img[100:200, 150:300] = (0.1, 0.1, 0.6)
    m = screen_mask((img * 65535).astype(np.uint16), "green")
    assert m[150, 225] == 1.0 and m[20, 20] == 0.0


def test_screen_mask_fills_small_holes_but_not_open_screen():
    from CorridorKeyModule.vision_hint import screen_mask

    img = _plate()
    img[60:220, 100:380] = (0.1, 0.1, 0.6)
    img[120:126, 200:206] = (0.15, 0.55, 0.15)  # green reflection inside the subject
    img[100:180, 250:330] = (0.15, 0.55, 0.15)  # big enclosed screen gap (between arms)
    m = screen_mask(img, "green")
    assert m[123, 203] == 1.0
    assert m[140, 290] == 0.0


def test_detect_screen_colour():
    from CorridorKeyModule.vision_hint import detect_screen

    assert detect_screen(_plate()) == "green"
    assert detect_screen(_plate(screen=(0.1, 0.2, 0.6))) == "blue"


def test_generate_hints_screen_mode_needs_no_vision(tmp_path):
    import cv2

    from CorridorKeyModule.vision_hint import generate_hints

    img = _plate()
    img[80:200, 150:300] = (0.1, 0.1, 0.6)
    frame = tmp_path / "plate_0001.tif"
    cv2.imwrite(str(frame), (img * 65535).astype(np.uint16))
    assert generate_hints([frame], tmp_path / "hints", mode="screen") == 1
    hint = cv2.imread(str(tmp_path / "hints" / "plate_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert hint[140, 225] == 255 and hint[20, 20] == 0
