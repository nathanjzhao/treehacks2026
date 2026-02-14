# Depth Anything V2

Monocular depth estimation on video using [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (ViT-Large) running on Modal A100 GPUs.

## Setup

```bash
pip install modal
modal setup  # one-time auth
```

## Run depth estimation

Single video:
```bash
modal run app.py --video-path ~/Downloads/video.MOV
```

Multiple videos in parallel (one A100 each):
```bash
modal run app.py --video-path "~/Downloads/VID1.MOV,~/Downloads/VID2.MOV,~/Downloads/VID3.mp4"
```

Options:
- `--fps 15` — frames per second to extract (default 15)
- `--input-size 518` — inference resolution (default 518)
- `--grayscale` — output grayscale instead of INFERNO colormap
- `--outdir examples/` — output directory (default `examples/`)

Results are saved as `{name}_depth.mp4` in the output directory.

## Viewer

Side-by-side playback of original + depth videos with frame-level controls.

```bash
python viewer.py examples/ --source-dir ~/Downloads
```

Open http://localhost:8080. Frames are pre-cached as ImageBitmaps for smooth playback.

Controls:
- `Space` — play/pause
- `←` `→` — step frame
- `[` `]` — skip 5 frames
- `Home` / `End` — first/last frame
- Dropdown to switch between videos
- FPS control for playback speed
