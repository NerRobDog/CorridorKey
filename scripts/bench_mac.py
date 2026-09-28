"""Benchmark CorridorKey configurations on one frame (built for Apple Silicon).

Measures, per configuration: load time, per-frame latency (model and
post-processing split for MLX), peak memory, and how far its alpha deviates
from a reference configuration. Prints a table and writes JSON you can send
back for analysis.

Configurations are ``backend:key=value:...`` strings:

    mlx:tile=512            MLX float engine, 512 px native-resolution tiles
    mlx:tile=768:compile    same with mx.compile
    mlx:full=2048           MLX full frame (needs ~27 GB — not on 16 GB machines)
    torch:img=2048          Torch on MPS (or --device), fp16 weights
    torch:img=1536:fp32     Torch with fp32 weights

Examples:

    uv run python scripts/bench_mac.py                          # synthetic 1080p frame
    uv run python scripts/bench_mac.py --size 3840x2160 --runs 3
    uv run python scripts/bench_mac.py --frame plate.exr --hint hint.png --linear \\
        --configs torch:img=2048 mlx:tile=512 mlx:tile=768 --reference torch:img=2048
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_CONFIGS = ["torch:img=2048", "mlx:tile=512", "mlx:tile=768", "mlx:tile=768:compile"]


def parse_config(spec: str) -> dict:
    backend, *parts = spec.split(":")
    cfg = {"name": spec, "backend": backend}
    for part in parts:
        key, _, value = part.partition("=")
        cfg[key] = int(value) if value.isdigit() else (value or True)
    return cfg


def system_info() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "model": sysctl("hw.model"),
        "chip": sysctl("machdep.cpu.brand_string"),
        "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2**30, 1) or None,
    }


def synthetic_frame(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Green screen with a soft-edged, semi-transparent subject and sensor noise."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / (0.22 * w), (yy - h / 2) / (0.4 * h))
    alpha = np.clip((1.0 - r) * 8.0, 0.0, 1.0)
    subject = np.stack([0.8 * np.ones_like(r), 0.55 - 0.2 * r, 0.45 * np.ones_like(r)], -1)
    screen = np.array([0.12, 0.62, 0.2], np.float32)
    img = subject * alpha[..., None] + screen * (1 - alpha[..., None])
    img = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1).astype(np.float32)
    hint = cv2.GaussianBlur((alpha > 0.5).astype(np.float32), (0, 0), max(w, h) / 200)
    return img, hint


def load_frame(args) -> tuple[np.ndarray, np.ndarray]:
    if not args.frame:
        w, h = (int(v) for v in args.size.lower().split("x"))
        return synthetic_frame(w, h)
    from backend.frame_io import read_image_frame

    img = read_image_frame(args.frame)
    if img is None:
        sys.exit(f"cannot read {args.frame}")
    hint = cv2.imread(args.hint, cv2.IMREAD_UNCHANGED)
    if hint is None:
        sys.exit(f"cannot read {args.hint}")
    if hint.ndim == 3:
        hint = hint[..., 0]
    scale = {np.uint8: 255.0, np.uint16: 65535.0}.get(hint.dtype.type, 1.0)
    hint = hint.astype(np.float32) / scale
    if hint.shape[:2] != img.shape[:2]:
        hint = cv2.resize(hint, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return img, hint


def build_engine(cfg: dict, checkpoint_dir: str, device: str):
    from CorridorKeyModule import backend as ck

    if checkpoint_dir:
        ck.CHECKPOINT_DIR = checkpoint_dir
    if cfg["backend"] == "mlx":
        from CorridorKeyModule.mlx_engine import MLXFloatEngine

        ckpt = ck._discover_checkpoint(ck.MLX_EXT)
        return MLXFloatEngine.from_checkpoint(
            str(ckpt),
            img_size=cfg.get("full", 2048),
            tile_size=None if "full" in cfg else cfg.get("tile", 512),
            overlap=cfg.get("overlap", 64),
            compile=bool(cfg.get("compile", False)),
        )
    import torch

    from CorridorKeyModule.inference_engine import CorridorKeyEngine

    ckpt = ck._discover_checkpoint(ck.TORCH_EXT)
    return CorridorKeyEngine(
        checkpoint_path=str(ckpt),
        device=device,
        img_size=cfg.get("img", 2048),
        model_precision=torch.float32 if cfg.get("fp32") else torch.float16,
    )


class Memory:
    """Best-effort peak memory probes for MLX, MPS and the whole process."""

    def __init__(self, backend: str):
        self.backend = backend
        self.mps_peak = 0

    def reset(self):
        if self.backend == "mlx":
            import mlx.core as mx

            mx.reset_peak_memory()

    def sample(self):
        if self.backend == "torch":
            import torch

            if torch.backends.mps.is_available():
                self.mps_peak = max(self.mps_peak, torch.mps.driver_allocated_memory())

    def report(self) -> dict:
        out = {}
        if self.backend == "mlx":
            import mlx.core as mx

            out["mlx_peak_mb"] = round(mx.get_peak_memory() / 2**20)
        if self.mps_peak:
            out["mps_driver_peak_mb"] = round(self.mps_peak / 2**20)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["process_max_rss_mb"] = round(rss / (2**20 if sys.platform == "darwin" else 2**10))
        return out


def release(engine, backend: str):
    del engine
    gc.collect()
    if backend == "mlx":
        import mlx.core as mx

        mx.clear_cache()
    else:
        from device_utils import clear_device_cache

        for dev in ("mps", "cuda"):
            try:
                clear_device_cache(dev)
            except Exception:  # noqa: BLE001 — cache clearing is best effort
                pass


def bench_config(cfg, img, hint, args) -> tuple[dict, np.ndarray | None]:
    result = {"config": cfg["name"]}
    mem = Memory(cfg["backend"])
    t0 = time.perf_counter()
    try:
        mem.reset()
        engine = build_engine(cfg, args.checkpoint_dir, args.device)
    except Exception as exc:  # noqa: BLE001 — report and move on to the next config
        result["error"] = f"load: {type(exc).__name__}: {exc}"
        return result, None
    result["load_s"] = round(time.perf_counter() - t0, 2)

    kwargs = {"input_is_linear": args.linear, "despill_strength": 1.0, "auto_despeckle": True}
    totals, model_times, post_times = [], [], []
    alpha = None
    try:
        for i in range(args.warmup + args.runs):
            t0 = time.perf_counter()
            if cfg["backend"] == "mlx":
                from CorridorKeyModule.mlx_engine import finalize_outputs

                pred = engine.predict(img, hint, input_is_linear=args.linear)
                t1 = time.perf_counter()
                out = finalize_outputs(pred["alpha"], pred["fg"], despill_strength=1.0, auto_despeckle=True)
                t2 = time.perf_counter()
                if i >= args.warmup:
                    model_times.append(t1 - t0)
                    post_times.append(t2 - t1)
            else:
                out = engine.process_frame(img, hint, **kwargs)
                t2 = time.perf_counter()
            mem.sample()
            if i >= args.warmup:
                totals.append(t2 - t0)
            alpha = out["alpha"][..., 0].astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"run: {type(exc).__name__}: {exc}"
    finally:
        result.update(mem.report())
        release(engine, cfg["backend"])

    if totals:
        result["frame_s_median"] = round(float(np.median(totals)), 3)
        result["frame_s_min"] = round(float(np.min(totals)), 3)
    if model_times:
        result["model_s_median"] = round(float(np.median(model_times)), 3)
        result["post_s_median"] = round(float(np.median(post_times)), 3)
    if alpha is not None:
        result["alpha_levels"] = int(np.unique(alpha).size)
    return result, alpha


def compare(alpha: np.ndarray, ref: np.ndarray) -> dict:
    diff = np.abs(alpha - ref)
    edge = (ref > 0.01) & (ref < 0.99)
    return {
        "alpha_mae_vs_ref": float(diff.mean()),
        "alpha_edge_mae_vs_ref": float(diff[edge].mean()) if edge.any() else None,
        "alpha_max_err_vs_ref": float(diff.max()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--reference", default=None, help="config used as quality reference (default: first)")
    p.add_argument("--frame", help="input frame (EXR/PNG/...); synthetic if omitted")
    p.add_argument("--hint", help="alpha hint for --frame")
    p.add_argument("--size", default="1920x1080", help="synthetic frame size WxH")
    p.add_argument("--linear", action="store_true", help="frame is linear (EXR)")
    p.add_argument("--device", default="mps", help="torch device")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", default="bench_results.json")
    args = p.parse_args()
    if args.frame and not args.hint:
        p.error("--frame requires --hint")

    img, hint = load_frame(args)
    print(f"frame {img.shape[1]}x{img.shape[0]}, {args.warmup} warmup + {args.runs} runs per config", flush=True)

    results, alphas = [], {}
    for spec in args.configs:
        cfg = parse_config(spec)
        print(f"-> {spec}", flush=True)
        res, alpha = bench_config(cfg, img, hint, args)
        results.append(res)
        if alpha is not None:
            alphas[spec] = alpha
        print("   ", json.dumps(res, ensure_ascii=False), flush=True)

    ref_name = args.reference or next(iter(alphas), None)
    if ref_name in alphas:
        for res in results:
            if res["config"] in alphas and res["config"] != ref_name:
                res.update(compare(alphas[res["config"]], alphas[ref_name]))

    report = {
        "system": system_info(),
        "frame": {"source": args.frame or f"synthetic {args.size}", "linear": args.linear},
        "reference": ref_name,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'config':28} {'frame s':>8} {'model s':>8} {'peak MB':>8} {'levels':>8} {'edge MAE':>9}")
    for r in results:
        peak = r.get("mlx_peak_mb") or r.get("mps_driver_peak_mb") or r.get("process_max_rss_mb")
        edge = r.get("alpha_edge_mae_vs_ref")
        print(
            f"{r['config']:28} {r.get('frame_s_median', '-')!s:>8} {r.get('model_s_median', '-')!s:>8} "
            f"{peak!s:>8} {r.get('alpha_levels', '-')!s:>8} {'' if edge is None else f'{edge:.5f}':>9}"
            + (f"  ERROR {r['error']}" if "error" in r else "")
        )
    print(f"\nreference: {ref_name}\nsaved {args.out}")


if __name__ == "__main__":
    main()
