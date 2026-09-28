"""Key the clip under the playhead in DaVinci Resolve Studio with CorridorKey and put the result back.

Timeline convention (all on the same frame range):

    V1  the green-screen plate
    V2  (skip with --auto-hint) the alpha hint: a copy of the plate graded to a white-on-black matte
        (e.g. a Magic Mask node, white inside / black outside)
    V3  the keyed result is placed here (created if missing)

Run it from a terminal while Resolve Studio is open (external scripting needs Studio):

    uv run --extra mlx python scripts/resolve_bridge.py

or, with the hint made by Apple Vision instead of V2:

    uv run --extra mlx --with pyobjc-framework-Vision --with pyobjc-framework-Quartz \\
        python scripts/resolve_bridge.py --auto-hint

What it does, for the V1 clip under the playhead:
  1. renders the plate range with only V1 enabled  -> ClipsForInference/<shot>/Input/
  2. renders the same range with only V2 enabled   -> ClipsForInference/<shot>/AlphaHint/
  3. runs CorridorKey (MLX, tiled)                  -> ClipsForInference/<shot>/Processed/ ...
  4. imports the Processed EXR sequence and lays it on V3 at the plate's position

Track enable states and render settings are restored afterwards.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIPS_DIR = REPO / "ClipsForInference"
IMAGE_EXTS = (".tif", ".tiff", ".png", ".exr", ".dpx", ".jpg")

MAC_MODULES = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"


def connect():
    """Return the Resolve app object (inside Resolve's script menu or from a terminal)."""
    g = globals()
    if "resolve" in g and g["resolve"]:
        return g["resolve"]
    if "bmd" in g:
        return g["bmd"].scriptapp("Resolve")
    if os.path.isdir(MAC_MODULES) and MAC_MODULES not in sys.path:
        sys.path.append(MAC_MODULES)
    try:
        import DaVinciResolveScript as dvr  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("DaVinciResolveScript not found. Is DaVinci Resolve Studio installed?")
    app = dvr.scriptapp("Resolve")
    if app is None:
        sys.exit(
            "Could not connect to Resolve. Open Resolve Studio and enable Preferences > General > External scripting."
        )
    return app


def timecode_to_frame(tc: str, fps: float) -> int:
    h, m, s, f = (int(x) for x in re.split(r"[:;]", tc))
    base = round(fps)
    return ((h * 60 + m) * 60 + s) * base + f


def item_at(timeline, track: int, frame: int):
    for item in timeline.GetItemListInTrack("video", track) or []:
        if item.GetStart() <= frame < item.GetEnd():
            return item
    return None


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(name)[0]).strip("_") or "shot"


def flatten_images(folder: Path) -> int:
    """Move rendered frames out of any subfolders Resolve created; return frame count."""
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.parent != folder:
            shutil.move(str(path), folder / path.name)
    for sub in sorted((p for p in folder.rglob("*") if p.is_dir()), reverse=True):
        if not any(sub.iterdir()):
            sub.rmdir()
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def pick_tiff_codec(project) -> str:
    codecs = project.GetRenderCodecs("tif") or {}
    for desc, codec in codecs.items():
        if "16" in desc:
            return codec
    if codecs:
        return next(iter(codecs.values()))
    sys.exit("Resolve reports no TIFF codecs for rendering.")


def render_range(project, timeline, only_track: int, start: int, end: int, target: Path, name: str) -> int:
    """Render timeline frames [start, end) with only ``only_track`` enabled."""
    target.mkdir(parents=True, exist_ok=True)
    tracks = range(1, timeline.GetTrackCount("video") + 1)
    saved = {t: timeline.GetIsTrackEnabled("video", t) for t in tracks}
    try:
        for t in tracks:
            timeline.SetTrackEnable("video", t, t == only_track)
        project.SetCurrentRenderFormatAndCodec("tif", pick_tiff_codec(project))
        ok = project.SetRenderSettings(
            {
                "SelectAllFrames": False,
                "MarkIn": start,
                "MarkOut": end - 1,
                "TargetDir": str(target),
                "CustomName": name,
                "ExportVideo": True,
                "ExportAudio": False,
            }
        )
        if not ok:
            sys.exit("Resolve rejected the render settings.")
        job = project.AddRenderJob()
        if not job:
            sys.exit("Resolve could not add a render job.")
        project.StartRendering([job])
        while project.IsRenderingInProgress():
            time.sleep(1)
        status = project.GetRenderJobStatus(job) or {}
        project.DeleteRenderJob(job)
        if status.get("JobStatus") not in (None, "Complete"):
            sys.exit(f"Render failed: {status}")
    finally:
        for t, enabled in saved.items():
            timeline.SetTrackEnable("video", t, enabled)
    return flatten_images(target)


def check_hint(folder: Path) -> None:
    """A hint must be a white-on-black matte, not a copy of the plate (e.g. a Magic Mask sent to alpha)."""
    import cv2
    import numpy as np

    frames = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("."))
    img = cv2.imread(str(frames[len(frames) // 2]), cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Cannot read hint frame {frames[len(frames) // 2]}")
    img = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0 if img.dtype == np.uint8 else 1.0)
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb = img[:, :, :3]
        colourful = float(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean()) > 0.02
        grey = rgb.mean(axis=2)
    else:
        colourful, grey = False, img.reshape(img.shape[0], img.shape[1])
    mid = float(((grey > 0.1) & (grey < 0.9)).mean())
    if colourful or mid > 0.25:
        sys.exit(
            f"The V2 render in {folder} does not look like a matte ({mid:.0%} mid-grey pixels"
            f"{', colour' if colourful else ''}). Resolve rendered the picture, not the mask: in the V2 grade "
            "turn the mask into RGB (white inside, black outside) instead of sending it to the alpha output, "
            "or run with --auto-hint to let Apple Vision make the hint."
        )


def vision_hints(plate_dir: Path, hint_dir: Path, mode: str) -> int:
    """Make the hint from the rendered plate with Apple Vision; returns the hint frame count."""
    sys.path.insert(0, str(REPO))
    try:
        from CorridorKeyModule.vision_hint import generate_hints
    except ImportError as exc:
        sys.exit(
            f"Apple Vision bindings missing ({exc}). Run with: uv run --extra mlx "
            "--with pyobjc-framework-Vision --with pyobjc-framework-Quartz python scripts/resolve_bridge.py --auto-hint"
        )
    if hint_dir.exists():
        shutil.rmtree(hint_dir)  # a stale V2 render must not mix with generated hints
    frames = sorted(f for f in plate_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("."))
    t0 = time.monotonic()
    generate_hints(
        frames, hint_dir, mode=mode, on_frame=lambda i, n: print(f"\r  hint {i + 1}/{n}", end="", flush=True)
    )
    print(f"\n  '{mode}' hints from Apple Vision in {time.monotonic() - t0:.0f} s", flush=True)
    return len(frames)


def run_corridorkey(args) -> None:
    cmd = [
        "uv", "run", "--extra", "mlx", "python", "corridorkey_cli.py", "run-inference",
        "--backend", "mlx", "--tile", "--srgb",
        "--despill", str(args.despill), "--despeckle", "--despeckle-size", str(args.despeckle_size),
        "--refiner", "1", "--no-comp", "--cpu-post", "--skip-existing",
    ]  # fmt: skip
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def import_sequence(media_pool, folder: Path):
    files = sorted(glob.glob(str(folder / "*.exr")))
    if not files:
        sys.exit(f"No EXR frames in {folder}")
    m = re.match(r"^(.*?)(\d+)(\.exr)$", os.path.basename(files[0]))
    last = re.match(r"^(.*?)(\d+)(\.exr)$", os.path.basename(files[-1]))
    if m and last:
        prefix, digits, ext = m.groups()
        pattern = str(folder / f"{prefix}%0{len(digits)}d{ext}")
        items = media_pool.ImportMedia(
            [{"FilePath": pattern, "StartIndex": int(digits), "EndIndex": int(last.group(2))}]
        )
    else:
        items = media_pool.ImportMedia([str(folder)])
    if not items:
        sys.exit(f"Resolve could not import {folder}")
    return items[0], len(files)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plate-track", type=int, default=1)
    p.add_argument("--hint-track", type=int, default=2)
    p.add_argument("--out-track", type=int, default=3)
    p.add_argument("--despill", type=float, default=5)
    p.add_argument("--despeckle-size", type=int, default=400)
    p.add_argument("--name", help="shot folder name (default: plate clip name)")
    p.add_argument(
        "--import-only",
        action="store_true",
        help="skip rendering and keying; just import an existing Output/Processed onto the out track",
    )
    p.add_argument(
        "--auto-hint",
        nargs="?",
        const="person",
        choices=("person", "objects"),
        help="no V2 needed: make the hint with Apple Vision (person, default, or objects)",
    )
    args = p.parse_args()

    resolve = connect()
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if not timeline:
        sys.exit("Open a project with a timeline first.")

    fps = float(timeline.GetSetting("timelineFrameRate"))
    playhead = timecode_to_frame(timeline.GetCurrentTimecode(), fps)
    plate = item_at(timeline, args.plate_track, playhead)
    if plate is None:
        sys.exit(f"No clip on V{args.plate_track} under the playhead.")
    start, end = plate.GetStart(), plate.GetEnd()
    hint = None if args.auto_hint or args.import_only else item_at(timeline, args.hint_track, start)
    if not (args.auto_hint or args.import_only) and (hint is None or hint.GetEnd() < end):
        sys.exit(f"V{args.hint_track} must hold the alpha-hint clip covering the whole plate range.")

    shot = args.name or safe_name(plate.GetName())
    shot_dir = CLIPS_DIR / shot
    print(f"Shot '{shot}': timeline frames {start}-{end - 1} ({end - start} frames)", flush=True)

    if args.import_only:
        place_result(project, timeline, shot_dir, start, args.out_track)
        return

    n_in = render_range(project, timeline, args.plate_track, start, end, shot_dir / "Input", shot)
    if args.auto_hint:
        n_hint = vision_hints(shot_dir / "Input", shot_dir / "AlphaHint", args.auto_hint)
    else:
        n_hint = render_range(project, timeline, args.hint_track, start, end, shot_dir / "AlphaHint", shot)
    if n_in != n_hint:
        sys.exit(f"Frame count mismatch: {n_in} plate frames vs {n_hint} hint frames.")
    print(f"Have {n_in} plate and hint frames.", flush=True)
    if not args.auto_hint:
        check_hint(shot_dir / "AlphaHint")

    output = shot_dir / "Output"
    if output.exists():  # fresh plate and hint renders: never resume a key made from older ones
        aside = shot_dir / f"Output_prev_{time.strftime('%Y%m%d-%H%M%S')}"
        output.rename(aside)
        print(f"Previous results moved to {aside.name}", flush=True)

    t0 = time.monotonic()
    run_corridorkey(args)
    print(f"CorridorKey finished in {time.monotonic() - t0:.0f} s.", flush=True)

    place_result(project, timeline, shot_dir, start, args.out_track)


def place_result(project, timeline, shot_dir: Path, start: int, out_track: int) -> None:
    media_pool = project.GetMediaPool()
    clip, n_out = import_sequence(media_pool, shot_dir / "Output" / "Processed")
    for key, value in (("Alpha mode", "Premultiplied"), ("Input Color Space", "Rec.709 Linear")):
        clip.SetClipProperty(key, value)  # best effort: names vary between Resolve versions

    while timeline.GetTrackCount("video") < out_track:
        timeline.AddTrack("video")
    placed = media_pool.AppendToTimeline(
        [
            {
                "mediaPoolItem": clip,
                "startFrame": 0,
                "endFrame": n_out - 1,
                "trackIndex": out_track,
                "recordFrame": start,
            }
        ]
    )
    if not placed:
        sys.exit("Imported the key but could not place it on the timeline; drag it from the media pool.")
    print(f"Placed {n_out} keyed frames on V{out_track}.", flush=True)


if __name__ == "__main__":
    main()
