"""Float-precision MLX engine for Apple Silicon.

Replaces the upstream ``corridorkey_mlx.CorridorKeyMLXEngine`` wrapper, whose
``process_frame`` contract is uint8 end to end: inputs are quantized to 8 bits
before inference, alpha/fg come back as uint8 (256 alpha levels), and
``input_is_linear`` is a no-op. This engine reuses only the MLX *model* from
``corridorkey_mlx`` and keeps everything else in float32:

* input stays float (HDR values above 1.0 survive, like the Torch engine);
* linear input is converted to sRGB before the model, matching Torch;
* tiling and blending are done with float32 accumulators;
* post-processing (despill, despeckle, premultiply, comp) is shared with the
  Torch engine's OpenCV path via :func:`finalize_outputs`.

Everything except the model forward pass is plain NumPy/OpenCV, so tiling and
post-processing are unit-testable on machines without MLX (see
``tests/test_mlx_engine.py``, which injects a fake ``model_fn``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import numpy as np

from CorridorKeyModule.core import color_utils as cu

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model callable: (1, H, W, 4) float32 NHWC -> {"alpha": (H, W, 1), "fg": (H, W, 3)} float32.
ModelFn = Callable[[np.ndarray], dict[str, np.ndarray]]


def tile_coords(size: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) spans covering ``[0, size)`` with ``tile``-sized windows.

    Every span is exactly ``tile`` long when ``size >= tile`` (the last one is
    shifted back instead of shrinking), so a compiled fixed-shape model never
    sees a different input shape.
    """
    if overlap >= tile:
        raise ValueError(f"overlap ({overlap}) must be less than tile size ({tile})")
    if size <= tile:
        return [(0, size)]
    stride = tile - overlap
    spans = []
    start = 0
    while True:
        end = min(start + tile, size)
        spans.append((end - tile, end))
        if end == size:
            return spans
        start += stride


def blend_ramp(length: int, overlap: int, ramp_start: bool, ramp_end: bool) -> np.ndarray:
    """1D blend weights: linear ramps over ``overlap`` px at shared edges.

    Ramps run over (0, 1] rather than [0, 1] so a pixel covered by a single
    tile edge never ends up with zero total weight.
    """
    w = np.ones(length, dtype=np.float32)
    n = min(overlap, length)
    if n > 0:
        ramp = np.arange(1, n + 1, dtype=np.float32) / n
        if ramp_start:
            w[:n] *= ramp
        if ramp_end:
            w[-n:] *= ramp[::-1]
    return w


def build_model_input(rgb_srgb: np.ndarray, hint: np.ndarray) -> np.ndarray:
    """ImageNet-normalize sRGB and append the hint: (1, H, W, 4) float32."""
    normalized = (rgb_srgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.concatenate([normalized, hint], axis=-1)[np.newaxis].astype(np.float32, copy=False)


def run_tiled(model_fn: ModelFn, x: np.ndarray, tile: int, overlap: int) -> dict[str, np.ndarray]:
    """Run ``model_fn`` on overlapping tiles of ``x`` and blend in float32."""
    _, h, w, _ = x.shape
    ys = tile_coords(h, tile, overlap)
    xs = tile_coords(w, tile, overlap)

    alpha_acc = np.zeros((h, w, 1), dtype=np.float32)
    fg_acc = np.zeros((h, w, 3), dtype=np.float32)
    weight_acc = np.zeros((h, w, 1), dtype=np.float32)

    for yi, (y0, y1) in enumerate(ys):
        wy = blend_ramp(y1 - y0, overlap, yi > 0, yi < len(ys) - 1)
        for xi, (x0, x1) in enumerate(xs):
            wx = blend_ramp(x1 - x0, overlap, xi > 0, xi < len(xs) - 1)
            patch = x[:, y0:y1, x0:x1, :]
            ph, pw = tile - (y1 - y0), tile - (x1 - x0)
            if ph > 0 or pw > 0:
                # Only happens when the frame is smaller than a tile on this axis.
                patch = np.pad(patch, ((0, 0), (0, max(ph, 0)), (0, max(pw, 0)), (0, 0)), mode="edge")
            out = model_fn(patch)
            weight = (wy[:, None] * wx[None, :])[:, :, None]
            alpha_acc[y0:y1, x0:x1] += out["alpha"][: y1 - y0, : x1 - x0] * weight
            fg_acc[y0:y1, x0:x1] += out["fg"][: y1 - y0, : x1 - x0] * weight
            weight_acc[y0:y1, x0:x1] += weight

    return {"alpha": alpha_acc / weight_acc, "fg": fg_acc / weight_acc}


CORE_ALPHA_LO = 0.90
CORE_ALPHA_HI = 0.98


def core_from_plate(fg: np.ndarray, alpha: np.ndarray, plate_srgb: np.ndarray) -> np.ndarray:
    """Use the plate's own colour where the subject is opaque.

    Where alpha is ~1 the screen does not show through, so the true foreground
    colour *is* the plate pixel; the model is only needed to unmix
    semi-transparent pixels. Taking the core from the plate removes the
    per-tile brightness drift of tiled inference (visible as a grid on 6K
    plates) and keeps full camera sharpness. Blends smoothly between
    CORE_ALPHA_LO and CORE_ALPHA_HI. Despill still runs afterwards.
    """
    w = np.clip((alpha - CORE_ALPHA_LO) / (CORE_ALPHA_HI - CORE_ALPHA_LO), 0.0, 1.0)
    return (fg * (1.0 - w) + plate_srgb[..., :3] * w).astype(np.float32, copy=False)


def finalize_outputs(
    alpha: np.ndarray,
    fg: np.ndarray,
    *,
    fg_is_straight: bool = True,
    despill_strength: float = 1.0,
    auto_despeckle: bool = True,
    despeckle_size: int = 400,
    generate_comp: bool = True,
    screen_channel: int = 1,
) -> dict[str, np.ndarray | None]:
    """Torch-engine output contract from full-resolution float predictions.

    Mirrors ``CorridorKeyEngine._postprocess_opencv`` after its resize step.
    """
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]
    if auto_despeckle:
        processed_alpha = cu.clean_matte_opencv(alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
    else:
        processed_alpha = alpha

    fg_despilled = cu.despill_opencv(fg, limit_mode="average", strength=despill_strength, screen_channel=screen_channel)
    fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
    processed_rgba = np.concatenate([cu.premultiply(fg_despilled_lin, processed_alpha), processed_alpha], axis=-1)

    comp_srgb = None
    if generate_comp:
        h, w = fg.shape[:2]
        bg_lin = cu.srgb_to_linear(cu.create_checkerboard(w, h, checker_size=128, color1=0.15, color2=0.55))
        if fg_is_straight:
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        else:
            comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)
        comp_srgb = cu.linear_to_srgb(comp_lin)

    return {"alpha": alpha, "fg": fg, "comp": comp_srgb, "processed": processed_rgba}


def _to_float(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32, copy=False)


class MLXFloatEngine:
    """``process_frame``-compatible engine that keeps the whole path in float32.

    Args:
        model_fn: Forward pass, see :data:`ModelFn`. Use :meth:`from_checkpoint`
            for the real MLX model.
        model_size: Square resolution the model was built for (tile size when
            tiling, otherwise the full-frame inference size).
        tiled: If True, run on native-resolution tiles of ``model_size``
            (aspect ratio preserved, no resize). If False, resize the whole
            frame to ``model_size`` and back, like the Torch engine.
        overlap: Tile overlap in pixels.
    """

    def __init__(self, model_fn: ModelFn, model_size: int, tiled: bool, overlap: int = 64) -> None:
        if tiled and overlap >= model_size:
            raise ValueError(f"overlap ({overlap}) must be less than tile size ({model_size})")
        self._model_fn = model_fn
        self.model_size = model_size
        self.tiled = tiled
        self.overlap = overlap
        self._refiner_warned = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        img_size: int = 2048,
        tile_size: int | None = 512,
        overlap: int = 64,
        compile: bool | None = None,
    ) -> MLXFloatEngine:
        """Load the ``corridorkey_mlx`` GreenFormer and wrap it.

        ``compile`` defaults to True: tiles are always exactly ``tile_size``
        square, so a fixed-shape compile is safe, and on an M1 Pro it cut
        tiled frame time by ~22% (bench_mac.py).
        """
        import mlx.core as mx  # type: ignore[import-not-found]
        from corridorkey_mlx.inference.pipeline import load_model  # type: ignore[import-not-found]

        tiled = bool(tile_size)
        model_size = int(tile_size) if tiled else img_size
        if compile is None:
            compile = True
        model = load_model(checkpoint_path, img_size=model_size, compile=compile, slim=True)

        def model_fn(x: np.ndarray) -> dict[str, np.ndarray]:
            out = model(mx.array(x))
            mx.eval(out["alpha_final"], out["fg_final"])  # noqa: S307 — MLX materialization, not Python eval
            return {
                "alpha": np.array(out["alpha_final"][0], dtype=np.float32),
                "fg": np.array(out["fg_final"][0], dtype=np.float32),
            }

        mode = f"tiled {model_size}/{overlap}" if tiled else f"full-frame {model_size}"
        logger.info("MLX float engine loaded: %s [%s, compile=%s]", checkpoint_path, mode, compile)
        return cls(model_fn, model_size=model_size, tiled=tiled, overlap=overlap)

    def predict(self, image: np.ndarray, mask: np.ndarray, input_is_linear: bool = False) -> dict[str, np.ndarray]:
        """Raw full-resolution float predictions: alpha (H, W, 1), fg (H, W, 3) sRGB."""
        image = _to_float(image)
        mask = _to_float(mask)
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        h, w = image.shape[:2]

        if self.tiled:
            rgb = cu.linear_to_srgb(np.maximum(image, 0.0)) if input_is_linear else image
            out = run_tiled(self._model_fn, build_model_input(rgb, mask), self.model_size, self.overlap)
            return {"alpha": out["alpha"], "fg": core_from_plate(out["fg"], out["alpha"], rgb)}

        # Full frame: resize (in linear light when the input is linear), then encode to sRGB.
        size = (self.model_size, self.model_size)
        shrink = self.model_size < max(h, w)
        interp = cv2.INTER_AREA if shrink else cv2.INTER_CUBIC
        rgb = cv2.resize(image, size, interpolation=interp)
        if input_is_linear:
            rgb = cu.linear_to_srgb(np.maximum(rgb, 0.0))
        hint = cv2.resize(mask, size, interpolation=interp)[:, :, np.newaxis]
        out = self._model_fn(build_model_input(rgb, hint))
        alpha = cv2.resize(out["alpha"], (w, h), interpolation=cv2.INTER_LANCZOS4)
        fg = cv2.resize(out["fg"], (w, h), interpolation=cv2.INTER_LANCZOS4)
        alpha = alpha.reshape(h, w, 1)
        plate = cu.linear_to_srgb(np.maximum(image, 0.0)) if input_is_linear else image
        return {"alpha": alpha, "fg": core_from_plate(fg, alpha, plate)}

    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
        generate_comp: bool = True,
        screen_channel: int = 1,
        **_kwargs,
    ) -> dict[str, np.ndarray | None]:
        """Same contract as ``CorridorKeyEngine.process_frame`` for a single frame."""
        if screen_channel != 1:
            raise NotImplementedError(
                f"MLX backend does not support screen_channel={screen_channel}: there is no blue MLX "
                "checkpoint yet. Use --backend torch with --screen-color blue."
            )
        if refiner_scale != 1.0 and not self._refiner_warned:
            logger.warning("refiner_scale=%s is not supported on the MLX backend; using 1.0", refiner_scale)
            self._refiner_warned = True

        pred = self.predict(image, mask_linear, input_is_linear=input_is_linear)
        return finalize_outputs(
            pred["alpha"],
            pred["fg"],
            fg_is_straight=fg_is_straight,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
            generate_comp=generate_comp,
            screen_channel=screen_channel,
        )
