# Fork status (Apple Silicon + DaVinci Resolve) — handoff

Branch: `claude/modest-cray-e5c1pv`. Always run with `uv run --extra mlx ...` (arm64 uv).

## Done and confirmed on hardware
- **MLX float engine** (`CorridorKeyModule/mlx_engine.py`): float32 end to end, honours `input_is_linear`,
  tiled 768/64 + `mx.compile`, opaque core colour from the plate (no tile grid on 6K),
  empty-tile skipping (96 px margin). MLX weights auto-converted from the Torch safetensors.
- **Speed (M1 Pro 16 GB):** 1080p ≈ 5.3–5.7 s/frame; 6K ≈ 51 s/frame, ≈ 39 s with tile skipping.
  Torch MPS was 324 s/frame. Vision hints: ≈ 0.17 s/frame 1080p, ≈ 1.1 s/frame 6K.
- **Vision hints** (`CorridorKeyModule/vision_hint.py`): person / objects, eroded + feathered PNGs.
- **`ck` wrapper** (`scripts/ck.py` + `ck`): drag a plate folder, finds or generates the hint,
  `--test` (10 frames), stale-output guard (`ck_recipe.json`), ignores `._` files, checks truncated frames.
- **Resolve bridge** (`scripts/resolve_bridge.py`): V1 plate under the playhead → render → key → V3.
  `--auto-hint [person|objects]` (no V2 needed), `--import-only`, check that a V2 render is a matte,
  old `Output` moved to `Output_prev_<stamp>` before keying.
- **Scripts menu buttons**: `uv run python scripts/install_resolve_menu.py` (Workspace > Scripts).

## Open / to verify
- ~~Gamma on import~~ **confirmed visually** (Resolve Studio 21.1, DWG colour-managed v2 project,
  2026-09-29): the gamma-2.4 copy (`Output/Processed_g24`) on V3 matches the plate on V1.
  `Input Gamma` is read-only through `SetClipProperty` for EXR clips — every value is refused
  (Linear, Linear Light, Gamma 2.4, sRGB, …) whatever `Input Color Space` is set to; `Input Color
  Space` itself accepts `Rec.709 (Scene)` and `Linear`. Values above 1.0 are less exact in the copy.
- Vision hints drop limbs and bodies in motion-blurred frames: on the test shot (`A_4-6/1`, 170
  frames) `person` loses > 15 % of the foreground on 36 frames, `objects` on 34 (different ones).
  New `--auto-hint screen` (rough chroma key, `vision_hint.screen_mask`) fixed all of them in a
  re-key of 11 of the worst frames: whole bodies, motion-blurred hands semi-transparent, no debris.
  It keys everyone in front of the screen (extras, Spider-Man). The user found a Resolve 3D Keyer
  V2 hint best in their own tests; automating it via `Graph.ApplyGradeFromDRX` is open.
- Default `--auto-hint` mode is still `person`; switching it to `screen` is open.
- Fixed: `--skip-existing` ignored finished frames under `--no-comp`, so every bridge run re-keyed
  every shot in `ClipsForInference`.
- Keying speed with Resolve open: 11.4 s/frame (full run), ≈ 45 s/frame on an 11-frame re-key,
  vs ≈ 5 s in the synthetic benchmark. Resolve sharing the GPU is the suspect — not measured.
- Menu buttons: not yet tried inside Resolve (macOS will ask to let Resolve control Terminal).
- Repeat the full 6K measurement.
- Colour notes: `Processed` = linear Rec.709 premultiplied RGBA EXR; `FG` = sRGB straight, not despilled.
  Plates: Rec.709/sRGB 16-bit TIFF. Hints: white on black, no alpha channel.

## Next (tracker)
5 LAN render farm (coordinator — local session) · 3 MLX attention (head_dim 56) · 4 fp16 on CPU ·
6 MCP part of the bridge · 9 subject crop · 10 TTA · 11 temporal stabilisation · 13 hint propagation ·
14 self-check / clean plate · 15 screen spill-light pass · 16 depth pass · 17 auto-tuning ·
18 LoRA · 19 shared farm with credits · 7 iPad worker (optional).
