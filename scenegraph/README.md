# OpenFunGraph

Video → 3D functional scene graph. Takes a video walkthrough and produces a structured graph of objects, interactive parts, and functional relationships (e.g. "pulling handle opens door").

## Usage

```bash
# Run pipeline (requires Modal account + OPENAI_API_KEY)
modal run scenegraph/app.py --video-path data/IMG_4717.MOV

# Visualize output
pip install viser numpy jinja2 fastapi uvicorn
python scenegraph/viewer.py data/IMG_4717.json
```

### Options

```
modal run scenegraph/app.py --video-path <path> [--fps 5] [--confidence 50] [--model gpt-4o]
```

- `--fps` — frame extraction rate (default 5). Lower = fewer frames, faster but less coverage.
- `--confidence` — minimum confidence threshold for detections (default 50).
- `--model` — OpenAI model for graph construction (default gpt-4o).

## Pipeline

Runs on Modal (A100-80GB GPU). 5 stages:

1. **MapAnything** — extract frames from video, estimate per-frame depth maps, camera poses (cam-to-world 4x4), and intrinsics. Uses Facebook's [Map Anything](https://github.com/facebookresearch/map-anything).

2. **Object Detection** — RAM generates open-vocabulary tags per frame, GroundingDINO localizes them as bounding boxes, SAM produces instance masks. CLIP encodes each detection.

3. **3D Fusion** — backproject 2D masks into 3D using depth + camera poses. Match detections across frames by point cloud overlap. DBSCAN outlier removal + merge pass. Filter by minimum detection count.

4. **Part Detection** — for each fused object, detect interactive parts (handles, knobs, buttons, levers) using GroundingDINO + SAM. Same 3D fusion as stage 3.

5. **Graph Construction** — GPT infers functional relationships between objects and parts. Produces edges like `("pulling", "handle", "opens", "door", 0.95)`.

## Output Format

JSON with:

```json
{
  "objects": [
    {
      "class_name": "door",
      "pcd_np": [[x, y, z], ...],
      "pcd_color_np": [[r, g, b], ...],
      "clip_ft": [768-dim vector],
      "n_detections": 15
    }
  ],
  "parts": [/* same structure */],
  "edges": [
    [obj_idx, part_idx, target_idx, "description", confidence]
  ],
  "camera_poses": [/* N x 4x4 cam-to-world matrices */],
  "camera_intrinsics": [/* N x 3x3 */],
  "n_frames": 262
}
```

## Viewer

The viser-based viewer (`viewer.py`) shows:
- Colored point clouds for each object and part
- Labels at object centroids
- Edge lines between related objects/parts
- Camera frustums at each frame position (yellow = frame 0, blue = rest)
- **Camera scrubber** — slider to snap to any original camera viewpoint

```bash
python scenegraph/viewer.py data/            # browse all JSONs in directory
python scenegraph/viewer.py data/scene.json  # open a specific file
```

## Architecture

```
app.py      — Modal pipeline (all 5 stages in one file)
viewer.py   — viser + FastAPI 3D viewer
templates/  — HTML template for viewer web UI
```

Based on [OpenFunGraph](https://github.com/ZhangCYG/OpenFunGraph), modified to use MapAnything (replacing VGGT) for higher-quality 3D reconstruction.
