"""Key a folder of green-screen frames: drag it into the terminal after `ck`.

    ck /path/to/plate_frames                  # hint folder found automatically
    ck /path/to/plate_frames /path/to/hint     # or pass the hint folder explicitly

The hint (white subject on black) is looked up, in order:
  * a subfolder of the plate folder named AlphaHint / Hint / Mask / Matte;
  * a sibling folder named <plate>_hint, <plate>_mask, AlphaHint or Hint.

Nothing is moved. Results go next to the plate, in <plate>_CorridorKey/Output/
(Processed = linear premultiplied RGBA EXR, FG, Matte, Comp preview).
Re-running resumes: frames already rendered are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

IMAGE_EXTS = (".tif", ".tiff", ".png", ".exr", ".dpx", ".jpg", ".jpeg", ".bmp")
HINT_NAMES = ("alphahint", "hint", "mask", "matte")


def clean(p: str) -> Path:
    """Paths dragged into a terminal can carry quotes, escaped spaces or a trailing slash."""
    path = Path(p.strip().strip("'\"").replace("\\ ", " ")).expanduser()
    if not path.is_absolute():  # relative to where the user typed `ck`, not the repo
        path = Path(os.environ.get("CK_CALLER_CWD", os.getcwd())) / path
    return path.resolve()


def is_frame(f: Path) -> bool:
    """Image files only; "._name" AppleDouble files that macOS writes on exFAT drives are not frames."""
    return f.suffix.lower() in IMAGE_EXTS and not f.name.startswith(".")


def has_frames(folder: Path) -> bool:
    return folder.is_dir() and any(is_frame(f) for f in folder.iterdir())


def frames_below(folder: Path, skip_hints: bool) -> Path | None:
    """The single folder at or below ``folder`` that holds frames (Resolve nests renders in subfolders)."""
    if has_frames(folder):
        return folder
    found = [
        d
        for d in folder.rglob("*")
        if d.is_dir()
        and has_frames(d)
        and not (skip_hints and any(part.lower() in HINT_NAMES for part in d.relative_to(folder).parts))
    ]
    return found[0] if len(found) == 1 else None


def plate_frames_dir(folder: Path) -> Path:
    """The folder with the plate frames: the folder itself or the one subfolder holding frames."""
    plate = frames_below(folder, skip_hints=True)
    if plate is None:
        sys.exit(f"Expected exactly one folder of plate frames in {folder}")
    return plate


def find_hint(folder: Path) -> Path | None:
    candidates = [d for d in folder.iterdir() if d.is_dir() and d.name.lower() in HINT_NAMES]
    candidates += [folder.parent / n for n in (f"{folder.name}_hint", f"{folder.name}_mask", "AlphaHint", "Hint")]
    for cand in candidates:
        if cand.is_dir():
            frames = frames_below(cand, skip_hints=False)
            if frames:
                return frames
    return None


def check_readable(folder: Path, label: str) -> None:
    """Fail early on truncated or unreadable frames instead of 'finishing' with no output."""
    import cv2

    frames = sorted(f for f in folder.iterdir() if is_frame(f))
    for f in (frames[0], frames[-1]):
        if cv2.imread(str(f), cv2.IMREAD_UNCHANGED) is None:
            size_mb = f.stat().st_size / 1e6
            sys.exit(
                f"Cannot read {label} frame {f.name} ({size_mb:.2f} MB). The file looks truncated or in an "
                "unsupported TIFF variant: re-render it (check free disk space), or render 16-bit TIFF "
                "without compression, PNG 16-bit or EXR."
            )


def auto_hint(plate: Path, out_dir: Path, mode: str) -> Path:
    """Generate hints for every frame (cheap next to keying): Apple Vision (macOS) or a chroma key."""
    if mode != "screen" and sys.platform != "darwin":
        sys.exit("No alpha hint found, and automatic hints need macOS (Apple Vision).")
    if out_dir.is_symlink():
        out_dir.unlink()  # an earlier run linked a user hint here; replace it with generated frames
    try:
        from CorridorKeyModule.vision_hint import generate_hints
    except ImportError as exc:
        sys.exit(f"Apple Vision bindings missing ({exc}). Use the ck launcher, which adds them on macOS.")
    frames = sorted(f for f in plate.iterdir() if is_frame(f))
    print(f"No hint folder: generating '{mode}' hints -> {out_dir}", flush=True)
    t0 = time.monotonic()
    written = generate_hints(
        frames, out_dir, mode=mode, on_frame=lambda i, n: print(f"\r  hint {i + 1}/{n}", end="", flush=True)
    )
    print(f"\n  {written} hints in {time.monotonic() - t0:.0f} s", flush=True)
    return out_dir


def recipe(hint: Path, args) -> dict:
    """What the results in Output/ depend on besides the plate."""
    frames = sorted(f for f in hint.iterdir() if is_frame(f))
    return {
        "hint": str(hint.resolve()),
        "hint_frames": len(frames),
        "hint_mtime": max((int(f.stat().st_mtime) for f in frames), default=0),
        "despill": args.despill,
        "despeckle": not args.no_despeckle,
        "despeckle_size": args.despeckle_size,
        "linear": args.linear,
    }


def guard_stale_output(work: Path, hint: Path, args) -> None:
    """Resume only results made with the same hint and settings; set older ones aside."""
    import json

    manifest = work / "ck_recipe.json"
    current = recipe(hint, args)
    output = work / "Output"
    if output.exists() and manifest.exists():
        try:
            previous = json.loads(manifest.read_text())
        except (OSError, ValueError):
            previous = None
        if previous != current:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            aside = work / f"Output_prev_{stamp}"
            output.rename(aside)
            print(f"Hint or settings changed since the last run: previous results moved to {aside.name}", flush=True)
    manifest.write_text(json.dumps(current, indent=2))


def link(target: Path, link_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.resolve() == target.resolve():
            return
        sys.exit(f"{link_path} already exists and points elsewhere; remove it or pick another folder.")
    link_path.symlink_to(target, target_is_directory=True)


def main() -> None:
    p = argparse.ArgumentParser(prog="ck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plate", help="folder with plate frames (or a shot folder with Input/)")
    p.add_argument("hint", nargs="?", help="folder with alpha-hint frames (optional)")
    p.add_argument("--despill", type=float, default=5, help="0-10 (default 5)")
    p.add_argument("--despeckle-size", type=int, default=400)
    p.add_argument("--no-despeckle", action="store_true")
    p.add_argument("--linear", action="store_true", help="plate is linear (default: auto, EXR = linear)")
    p.add_argument("--test", action="store_true", help="only the first 10 frames, to check the key")
    p.add_argument(
        "--auto-hint",
        choices=("screen", "person", "objects"),
        help="generate the hint with a rough chroma key (screen) or Apple Vision (person, objects); "
        "default when no hint folder is found: screen",
    )
    args = p.parse_args()

    shot = clean(args.plate)
    plate = plate_frames_dir(shot)
    work = shot.parent / f"{shot.name}_CorridorKey"
    work.mkdir(exist_ok=True)
    link(plate, work / "Input")

    hint = None
    if args.hint:
        hint = frames_below(clean(args.hint), skip_hints=False)
    elif not args.auto_hint:
        hint = find_hint(shot)
    if hint is None:
        hint = auto_hint(plate, work / "AlphaHint", args.auto_hint or "screen")
    else:
        link(hint, work / "AlphaHint")

    n_plate = sum(1 for f in plate.iterdir() if is_frame(f))
    n_hint = sum(1 for f in hint.iterdir() if is_frame(f))
    if n_plate != n_hint:
        sys.exit(f"Plate has {n_plate} frames but the hint has {n_hint}; they must match.")
    linear = args.linear or any(f.suffix.lower() == ".exr" for f in plate.iterdir())
    check_readable(plate, "plate")
    check_readable(hint, "hint")

    guard_stale_output(work, hint, args)

    from clip_manager import ClipEntry, InferenceSettings, run_inference

    clip = ClipEntry(work.name, str(work))
    clip.find_assets()
    clip.validate_pair()
    settings = InferenceSettings(
        input_is_linear=linear,
        despill_strength=max(0.0, min(10.0, args.despill)) / 10.0,
        auto_despeckle=not args.no_despeckle,
        despeckle_size=args.despeckle_size,
        refiner_scale=1.0,
        generate_comp=True,
        gpu_post_processing=False,
        tiled_inference=True,
    )
    print(f"Keying {n_plate} frames from {plate}\n  hint:   {hint}\n  output: {work / 'Output'}", flush=True)
    t0 = time.monotonic()
    run_inference(
        [clip],
        backend="mlx",
        max_frames=10 if args.test else None,
        skip_existing=True,
        settings=settings,
        on_frame_complete=lambda i, n: print(f"\r  frame {i + 1}/{n}", end="", flush=True),
    )
    print(f"\nDone in {time.monotonic() - t0:.0f} s -> {work / 'Output'}")


if __name__ == "__main__":
    main()
