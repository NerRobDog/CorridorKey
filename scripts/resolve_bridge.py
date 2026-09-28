"""Key the clip under the playhead in DaVinci Resolve Studio with CorridorKey and put the result back.

Timeline convention (all on the same frame range):

    V1  the green-screen plate
    V2  the alpha hint: a copy of the plate graded to a white-on-black matte
        (e.g. a Magic Mask node, white inside / black outside)
    V3  the keyed result is placed here (created if missing)

Run it from a terminal while Resolve Studio is open (external scripting needs Studio):

    uv run --extra mlx python scripts/resolve_bridge.py

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
    hint = item_at(timeline, args.hint_track, start)
    if hint is None or hint.GetEnd() < end:
        sys.exit(f"V{args.hint_track} must hold the alpha-hint clip covering the whole plate range.")

    shot = args.name or safe_name(plate.GetName())
    shot_dir = CLIPS_DIR / shot
    print(f"Shot '{shot}': timeline frames {start}-{end - 1} ({end - start} frames)", flush=True)

    n_in = render_range(project, timeline, args.plate_track, start, end, shot_dir / "Input", shot)
    n_hint = render_range(project, timeline, args.hint_track, start, end, shot_dir / "AlphaHint", shot)
    if n_in != n_hint:
        sys.exit(f"Frame count mismatch: {n_in} plate frames vs {n_hint} hint frames.")
    print(f"Rendered {n_in} plate and hint frames.", flush=True)

    t0 = time.monotonic()
    run_corridorkey(args)
    print(f"CorridorKey finished in {time.monotonic() - t0:.0f} s.", flush=True)

    media_pool = project.GetMediaPool()
    clip, n_out = import_sequence(media_pool, shot_dir / "Processed")
    for key, value in (("Alpha mode", "Premultiplied"), ("Input Color Space", "Rec.709 Linear")):
        clip.SetClipProperty(key, value)  # best effort: names vary between Resolve versions

    while timeline.GetTrackCount("video") < args.out_track:
        timeline.AddTrack("video")
    placed = media_pool.AppendToTimeline(
        [
            {
                "mediaPoolItem": clip,
                "startFrame": 0,
                "endFrame": n_out - 1,
                "trackIndex": args.out_track,
                "recordFrame": start,
            }
        ]
    )
    if not placed:
        sys.exit("Imported the key but could not place it on the timeline; drag it from the media pool.")
    print(f"Placed {n_out} keyed frames on V{args.out_track}.", flush=True)


if __name__ == "__main__":
    main()
