# Grounded SAM 2 on Modal

Open-set object detection and tracking in video using [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) + [SAM 2.1](https://github.com/facebookresearch/sam2) on A100 GPUs.

Give it a video and a text prompt, get back an annotated video with tracked bounding boxes/masks + per-frame detection JSON.

## Usage

```bash
modal run groundedsam2/app.py --video-path data/IMG_4723.MOV --text-prompt "book. door. painting."
```

Outputs `_tracked.mp4` and `_detections.json` next to the input file.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--text-prompt` | `"person."` | Objects to detect. Period-separated (e.g. `"car. person. dog."`) |
| `--prompt-type` | `mask` | How to prompt SAM 2 tracker: `point`, `box`, or `mask` |
| `--box-threshold` | `0.4` | Grounding DINO box confidence. Lower = more detections |
| `--text-threshold` | `0.3` | Grounding DINO text similarity threshold |
| `--ann-frame-idx` | `0` | Frame to run initial detection on |

```bash
# More detections (lower thresholds)
modal run groundedsam2/app.py --video-path vid.mov --text-prompt "chair. lamp. door." --box-threshold 0.25

# Detect from a later frame
modal run groundedsam2/app.py --video-path vid.mov --text-prompt "person." --ann-frame-idx 30
```

### Batch (parallel)

Process all `data/*.MOV` files in parallel across separate A100s:

```bash
modal run groundedsam2/run_batch.py --text-prompt "book. door. painting. chair."
```

Results go to `data/groundedsam2/`.

## Object-Aware Depth

Combines Grounded SAM 2 tracking with [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) to get depth maps masked to only the segmented objects.

```bash
modal run groundedsam2/depth_app.py --video-path data/IMG_4723.MOV --text-prompt "painting. chair. lamp. door."
```

Outputs to `data/groundedsam2/`:
- `{stem}_masked_depth.mp4` — depth colormap only where objects are, black elsewhere
- `{stem}_composite.mp4` — full depth dimmed to 20%, objects at full brightness with colored outlines + labels
- `{stem}_seg_depth.json` — per-frame detection metadata

### Viewer

Three-panel playback (Original | Object Depth | Composite):

```bash
python groundedsam2/depth_viewer.py data/groundedsam2/ --source-dir data/
# Open http://localhost:8080
```

Keyboard: `Space` play/pause, `←→` step, `[]` ±5 frames, `Home`/`End` first/last.

## Pipeline

1. Extract all video frames as JPEGs on Modal A100
2. Run Grounding DINO (HuggingFace) on the annotation frame to detect objects from text prompt
3. SAM 2.1 image predictor generates masks per detected object
4. Register objects with SAM 2 video predictor (point/box/mask prompt)
5. Propagate tracking across all frames
6. Annotate with bounding boxes + masks + labels via `supervision`
7. Stitch annotated frames back into mp4
