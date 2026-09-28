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
    return Path(p.strip().strip("'\"").replace("\\ ", " ")).expanduser().resolve()


def has_frames(folder: Path) -> bool:
    return folder.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in folder.iterdir())


def plate_frames_dir(folder: Path) -> Path:
    """The folder with the plate frames: the folder itself, or its Input/ subfolder."""
    if has_frames(folder):
        return folder
    for name in ("Input", "input"):
        if has_frames(folder / name):
            return folder / name
    sys.exit(f"No image frames found in {folder}")


def find_hint(folder: Path) -> Path | None:
    for sub in folder.iterdir():
        if sub.is_dir() and sub.name.lower() in HINT_NAMES and has_frames(sub):
            return sub
    for name in (f"{folder.name}_hint", f"{folder.name}_mask", "AlphaHint", "Hint"):
        cand = folder.parent / name
        if has_frames(cand):
            return cand
    return None


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
    args = p.parse_args()

    shot = clean(args.plate)
    plate = plate_frames_dir(shot)
    hint = clean(args.hint) if args.hint else (find_hint(shot) or (find_hint(plate) if plate != shot else None))
    if hint is None or not has_frames(hint):
        sys.exit(
            "No alpha hint found. Put the white-on-black matte frames in a subfolder named AlphaHint "
            f"inside {shot.name}, or pass the hint folder as the second argument."
        )

    work = shot.parent / f"{shot.name}_CorridorKey"
    work.mkdir(exist_ok=True)
    link(plate, work / "Input")
    link(hint, work / "AlphaHint")

    n_plate = sum(1 for f in plate.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    n_hint = sum(1 for f in hint.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if n_plate != n_hint:
        sys.exit(f"Plate has {n_plate} frames but the hint has {n_hint}; they must match.")
    linear = args.linear or any(f.suffix.lower() == ".exr" for f in plate.iterdir())

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
