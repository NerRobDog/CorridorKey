"""Alpha hints from Apple's Vision framework (macOS, runs on the Neural Engine).

Two modes:

* ``person``  — VNGeneratePersonSegmentationRequest: soft mask of all people.
* ``objects`` — VNGenerateForegroundInstanceMaskRequest (macOS 14+): every
  salient foreground object, the same "lift subject" technology as Photos.

CorridorKey is trained on coarse, slightly eroded hints and is better at
adding detail than removing it, so masks are thresholded, eroded a little and
feathered before being written.

Only the functions that touch Vision need macOS; the post-processing is plain
NumPy/OpenCV and is unit-tested everywhere.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODES = ("person", "objects")


def shape_hint(mask: np.ndarray, erode_px: int = 6, feather_px: int = 8) -> np.ndarray:
    """Binary-ish Vision mask -> CorridorKey-style coarse hint in [0, 1] float32."""
    m = (mask.astype(np.float32) > 0.5).astype(np.uint8)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, k)
    out = m.astype(np.float32)
    if feather_px > 0:
        out = cv2.GaussianBlur(out, (0, 0), feather_px / 2.0)
    return np.clip(out, 0.0, 1.0)


def _pixel_buffer_to_array(pb) -> np.ndarray:
    """CVPixelBuffer (one component, 8-bit or 32-bit float) -> float32 array in [0, 1]."""
    import Quartz  # type: ignore[import-not-found]

    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        w = Quartz.CVPixelBufferGetWidth(pb)
        h = Quartz.CVPixelBufferGetHeight(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        fmt = Quartz.CVPixelBufferGetPixelFormatType(pb)
        raw = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h)
        if fmt == Quartz.kCVPixelFormatType_OneComponent32Float:
            arr = np.frombuffer(raw, dtype=np.float32).reshape(h, bpr // 4)[:, :w].copy()
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, bpr)[:, :w].astype(np.float32) / 255.0
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)
    return arr


def vision_mask(image_path: str | Path, mode: str = "person") -> np.ndarray:
    """Raw Vision mask for one image file, resized to the image, float32 [0, 1]."""
    import Vision  # type: ignore[import-not-found]
    from Foundation import NSURL  # type: ignore[import-not-found]

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    url = NSURL.fileURLWithPath_(str(image_path))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})

    if mode == "person":
        request = Vision.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
        request.setQualityLevel_(Vision.VNGeneratePersonSegmentationRequestQualityLevelAccurate)
    else:
        request = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().initWithCompletionHandler_(None)

    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision failed on {image_path}: {err}")
    results = request.results() or []
    if not results:
        return np.zeros((1, 1), np.float32)  # nothing found; caller resizes to frame

    if mode == "person":
        pb = results[0].pixelBuffer()
    else:
        obs = results[0]
        pb, err = obs.generateScaledMaskForImageForInstances_fromRequestHandler_error_(
            obs.allInstances(), handler, None
        )
        if pb is None:
            raise RuntimeError(f"Vision could not build the foreground mask for {image_path}: {err}")
    return _pixel_buffer_to_array(pb)


def generate_hints(
    frames: list[Path],
    out_dir: Path,
    mode: str = "person",
    erode_px: int = 6,
    feather_px: int = 8,
    on_frame=None,
) -> int:
    """Write one 8-bit PNG hint per frame (same stem) into ``out_dir``; skip existing ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, frame in enumerate(frames):
        dst = out_dir / f"{frame.stem}.png"
        if not dst.exists():
            size = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED).shape[:2]
            mask = cv2.resize(vision_mask(frame, mode), (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            hint = shape_hint(mask, erode_px=erode_px, feather_px=feather_px)
            cv2.imwrite(str(dst), (hint * 255.0 + 0.5).astype(np.uint8))
            written += 1
        if on_frame:
            on_frame(i, len(frames))
    return written
