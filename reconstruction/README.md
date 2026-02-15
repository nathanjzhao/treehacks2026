# MapAnything on Modal

3D point cloud reconstruction from video using [MapAnything](https://github.com/facebookresearch/map-anything) (Meta/CMU, 2025) on A100 GPUs.

Supports optional auxiliary geometric inputs (camera intrinsics, poses) for improved reconstruction quality.

## Video to GLB

```bash
modal run app.py --video-path ~/Desktop/video.mov
```

Outputs `.glb` next to the input file.

### Quality flags

| Flag | Default | Description |
|------|---------|-------------|
| `--fps` | 2 | Frames extracted per second. More = denser, slower |
| `--conf` | 25 | Confidence percentile cutoff. Lower = more points, more noise |

```bash
# High quality
modal run app.py --video-path vid.mov --fps 5 --conf 20

# Fast preview
modal run app.py --video-path vid.mov --fps 1 --conf 40
```

### Batch (parallel)

Process multiple videos in parallel across separate A100s:

```bash
modal run run_batch.py::batch --fps 5 --conf 20
```

Edit `VIDEOS` list in `run_batch.py` to specify which files. Outputs go to `examples/`.

### Auxiliary inputs

MapAnything can leverage known geometric information for better results:

```bash
# With known camera intrinsics
modal run app.py --video-path vid.mov --intrinsics intrinsics.json

# With known camera poses (e.g. from IMU/GPS)
modal run app.py --video-path vid.mov --poses poses.json

# Both
modal run app.py --video-path vid.mov --intrinsics intrinsics.json --poses poses.json
```

**intrinsics.json** — 3x3 camera intrinsics matrix:
```json
[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

**poses.json** — list of 4x4 camera-to-world matrices (one per frame):
```json
[[[r00, r01, r02, tx], [r10, r11, r12, ty], [r20, r21, r22, tz], [0, 0, 0, 1]], ...]
```

## Local 3D viewer

View GLB point clouds locally (Viser-based, orbit controls + chat sidebar):

```bash
# Setup (once)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# View a reconstruction (open http://localhost:8081)
.venv/bin/python viewer.py examples/IMG_4723.glb --downsample 20
```

### All examples

```bash
# Small (3.3M pts, 50MB)
.venv/bin/python viewer.py examples/IMG_4723.glb --downsample 20

# Medium (5.9M pts, 90MB)
.venv/bin/python viewer.py examples/IMG_4724.glb --downsample 20

# Medium (8.9M pts, 136MB)
.venv/bin/python viewer.py examples/IMG_4722.glb --downsample 20

# Large (19.2M pts, 293MB)
.venv/bin/python viewer.py examples/IMG_4718.glb --downsample 40

# Large (29.5M pts, 450MB)
.venv/bin/python viewer.py examples/IMG_4720.glb --downsample 40

# Large (31.6M pts, 482MB)
.venv/bin/python viewer.py examples/IMG_4717.glb --downsample 40

# Largest (41.3M pts, 630MB)
.venv/bin/python viewer.py examples/IMG_4721.glb --downsample 60
```

Open `http://localhost:8081` for the 3D viewer. Adjust `--downsample` to trade detail for responsiveness.
