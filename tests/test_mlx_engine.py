"""Tests for the float MLX engine — run anywhere via an injected fake model."""

from unittest import mock

import numpy as np
import pytest

from CorridorKeyModule.core import color_utils as cu
from CorridorKeyModule.mlx_engine import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MLXFloatEngine,
    blend_ramp,
    finalize_outputs,
    run_tiled,
    tile_coords,
)


def passthrough_model(tile: int):
    """Fake model: alpha = hint, fg = de-normalized RGB. Asserts the fixed tile shape."""

    def model_fn(x):
        assert x.shape == (1, tile, tile, 4), x.shape
        assert x.dtype == np.float32
        rgb = x[0, :, :, :3] * IMAGENET_STD + IMAGENET_MEAN
        return {"alpha": x[0, :, :, 3:4].copy(), "fg": rgb}

    return model_fn


class TestTileCoords:
    @pytest.mark.parametrize("size", [512, 513, 1000, 1080, 1920, 3840])
    def test_full_coverage_and_fixed_length(self, size):
        spans = tile_coords(size, 512, 64)
        assert spans[0][0] == 0
        assert spans[-1][1] == size
        assert all(e - s == 512 for s, e in spans)
        for (_, prev_end), (start, _) in zip(spans, spans[1:], strict=False):
            assert start < prev_end, "adjacent tiles must overlap"

    def test_smaller_than_tile(self):
        assert tile_coords(300, 512, 64) == [(0, 300)]

    def test_overlap_must_be_smaller_than_tile(self):
        with pytest.raises(ValueError):
            tile_coords(1000, 64, 64)


def test_engine_rejects_overlap_not_smaller_than_tile():
    with pytest.raises(ValueError, match="overlap"):
        MLXFloatEngine(passthrough_model(64), model_size=64, tiled=True, overlap=64)


def test_blend_ramp_is_strictly_positive():
    w = blend_ramp(512, 64, True, True)
    assert w.min() > 0.0
    assert w[64:-64].min() == 1.0


class TestFloatPrecision:
    def test_tiled_passthrough_is_lossless(self):
        """Blending identical overlapping predictions must reproduce them exactly (no 8-bit step)."""
        rng = np.random.default_rng(0)
        h, w = 700, 1100
        image = rng.random((h, w, 3), dtype=np.float32)
        image[0, 0] = [1.7, 0.25, 3.0]  # HDR survives: the old adapter clipped to [0, 1]
        mask = rng.random((h, w), dtype=np.float32)

        engine = MLXFloatEngine(passthrough_model(512), model_size=512, tiled=True, overlap=64)
        pred = engine.predict(image, mask)

        np.testing.assert_allclose(pred["fg"], image, atol=1e-5)
        np.testing.assert_allclose(pred["alpha"][:, :, 0], mask, atol=1e-6)
        # far more than the 256 levels a uint8 path can express
        assert np.unique(pred["alpha"]).size > 10_000

    def test_linear_input_is_encoded_to_srgb(self):
        """input_is_linear was a no-op upstream; the model must see sRGB."""
        image = np.full((64, 64, 3), 0.18, dtype=np.float32)  # linear mid-grey
        mask = np.ones((64, 64), dtype=np.float32)
        engine = MLXFloatEngine(passthrough_model(128), model_size=128, tiled=True)

        pred = engine.predict(image, mask, input_is_linear=True)

        np.testing.assert_allclose(pred["fg"], cu.linear_to_srgb(image), atol=1e-5)
        assert pred["fg"].mean() > 0.4  # ~0.46 in sRGB, not 0.18

    def test_uint8_and_uint16_inputs_are_scaled(self):
        engine = MLXFloatEngine(passthrough_model(64), model_size=64, tiled=True, overlap=16)
        img8 = np.full((32, 32, 3), 255, dtype=np.uint8)
        mask16 = np.full((32, 32), 65535, dtype=np.uint16)
        pred = engine.predict(img8, mask16)
        np.testing.assert_allclose(pred["fg"], 1.0, atol=1e-5)
        np.testing.assert_allclose(pred["alpha"], 1.0, atol=1e-6)

    def test_full_frame_resizes_to_model_and_back(self):
        h, w = 270, 480
        image = np.full((h, w, 3), 0.3, dtype=np.float32)
        mask = np.full((h, w), 0.6, dtype=np.float32)
        engine = MLXFloatEngine(passthrough_model(256), model_size=256, tiled=False)

        pred = engine.predict(image, mask)

        assert pred["alpha"].shape == (h, w, 1)
        assert pred["fg"].shape == (h, w, 3)
        np.testing.assert_allclose(pred["alpha"], 0.6, atol=1e-4)
        np.testing.assert_allclose(pred["fg"], 0.3, atol=1e-4)


def test_run_tiled_single_tile_matches_model():
    x = np.random.default_rng(1).random((1, 64, 64, 4), dtype=np.float32)
    out = run_tiled(passthrough_model(64), x, tile=64, overlap=16)
    np.testing.assert_allclose(out["alpha"], x[0, :, :, 3:4], atol=1e-6)


class TestProcessFrameContract:
    def test_keys_shapes_dtypes(self):
        h, w = 96, 160
        engine = MLXFloatEngine(passthrough_model(64), model_size=64, tiled=True, overlap=16)
        image = np.random.default_rng(2).random((h, w, 3), dtype=np.float32)
        result = engine.process_frame(image, np.ones((h, w, 1), dtype=np.float32))

        assert result["alpha"].shape == (h, w, 1)
        assert result["fg"].shape == (h, w, 3)
        assert result["comp"].shape == (h, w, 3)
        assert result["processed"].shape == (h, w, 4)
        for key in ("alpha", "fg", "comp", "processed"):
            assert result[key].dtype == np.float32, key

    def test_generate_comp_false(self):
        engine = MLXFloatEngine(passthrough_model(32), model_size=32, tiled=True, overlap=8)
        result = engine.process_frame(
            np.zeros((32, 32, 3), np.float32), np.zeros((32, 32), np.float32), generate_comp=False
        )
        assert result["comp"] is None

    def test_blue_screen_rejected(self):
        engine = MLXFloatEngine(passthrough_model(32), model_size=32, tiled=True, overlap=8)
        with pytest.raises(NotImplementedError, match="screen_channel"):
            engine.process_frame(np.zeros((32, 32, 3), np.float32), np.zeros((32, 32), np.float32), screen_channel=2)


def test_finalize_matches_torch_postprocess_math():
    rng = np.random.default_rng(3)
    alpha = rng.random((40, 40, 1), dtype=np.float32)
    fg = rng.random((40, 40, 3), dtype=np.float32)
    out = finalize_outputs(alpha, fg, despill_strength=0.0, auto_despeckle=False)
    expected_rgb = cu.premultiply(cu.srgb_to_linear(fg), alpha)
    np.testing.assert_allclose(out["processed"][:, :, :3], expected_rgb, atol=1e-6)
    np.testing.assert_allclose(out["processed"][:, :, 3:], alpha)


def test_create_engine_mlx_uses_float_engine(tmp_path):
    from CorridorKeyModule import backend

    ckpt = tmp_path / "CorridorKey_v1.0.safetensors"
    ckpt.write_bytes(b"x")
    sentinel = object()
    with (
        mock.patch.object(backend, "CHECKPOINT_DIR", str(tmp_path)),
        mock.patch.object(backend, "resolve_backend", return_value="mlx"),
        mock.patch.object(MLXFloatEngine, "from_checkpoint", return_value=sentinel) as factory,
    ):
        assert backend.create_engine(backend="mlx", tile_size=768, overlap=32) is sentinel
    factory.assert_called_once_with(str(ckpt), img_size=2048, tile_size=768, overlap=32)


def test_opaque_core_comes_from_plate_not_from_tiles():
    """Per-tile colour drift must not reach the opaque core (the 6K grid artefact)."""
    calls = {"n": 0}

    def drifting_model(x):
        calls["n"] += 1
        rgb = x[0, :, :, :3] * IMAGENET_STD + IMAGENET_MEAN
        return {"alpha": x[0, :, :, 3:4].copy(), "fg": rgb * (1.0 + 0.1 * calls["n"])}  # each tile brighter

    h, w = 200, 300
    image = np.full((h, w, 3), 0.4, np.float32)
    mask = np.ones((h, w), np.float32)
    mask[:, :50] = 0.5  # semi-transparent strip keeps the model's colour
    engine = MLXFloatEngine(drifting_model, model_size=128, tiled=True, overlap=16)
    pred = engine.predict(image, mask)

    np.testing.assert_allclose(pred["fg"][:, 60:], 0.4, atol=1e-6)
    assert not np.allclose(pred["fg"][:, :40], 0.4)
