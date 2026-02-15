# hloc Localization

Visual localization pipeline using [hloc](https://github.com/cvg/Hierarchical-Localization) (SuperPoint + LightGlue + PnP). Takes a reference video of a scene, builds a 3D map, then localizes new camera frames against it in 6DoF.

## How it works

**Offline (one-time per scene, runs on Modal A100):**
1. Extract frames from reference video
2. SuperPoint feature extraction + NetVLAD global descriptors
3. LightGlue feature matching
4. pycolmap SfM reconstruction

**Online (per query frame, runs on Modal A100):**
1. Extract SuperPoint features + NetVLAD descriptor from query image
2. Retrieve top-K similar reference images
3. LightGlue matching against retrieved images
4. Build 2D-3D correspondences from matches + SfM model
5. PnP+RANSAC → 6DoF pose (qw, qx, qy, qz, tx, ty, tz)

## Structure

```
hloc_localization/
  backend/
    app.py              # Modal app: build_reference() + localize_frame()
    server.py           # FastAPI server with REST + WebSocket endpoints
    run_batch.py        # Build reference maps for all videos in data/
    debug_localize.py   # Localize a video file and print poses
    visualize_poses.py  # Run localization + save trajectory plot
    plot_trajectory.py  # Plot pre-computed poses (no Modal needed)
  frontend/
    viewer.py           # Viser-based live viewer (webcam → localization → 3D)
    view_trajectory.py  # Offline viewer: GLB point cloud + camera trajectory
    templates/          # Web UI for live viewer
  data/
    hloc_reference/     # Built reference maps (tar.gz per scene)
```

## Quick start

### 1. Build a reference map

```bash
modal run hloc_localization/backend/app.py --video-path data/IMG_4720.MOV
```

Outputs `hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz`.

### 2. Localize frames from another video

```bash
python -m hloc_localization.backend.debug_localize \
  --video data/IMG_4730.MOV \
  --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz \
  --fps 2
```

### 3. View trajectory in 3D

```bash
python -m hloc_localization.frontend.view_trajectory
```

Opens a viser viewer at `http://localhost:8890` showing the GLB point cloud with camera frustums. Play/pause to animate through the trajectory. The COLMAP and GLB coordinate systems are aligned automatically via ICP.

### 4. Run the live server (streaming)

```bash
python -m hloc_localization.backend.server --port 8090
```

Endpoints:
- `POST /localize` — upload a JPEG, get back a 6DoF pose
- `WS /stream` — stream frames in, stream poses out
- `GET /reference/status` — list available reference maps
- `POST /reference/build` — upload video to build a new reference

## DPVO Odometry

For real-time tracking, DPVO provides fast visual odometry anchored to world coordinates via HLoc.

**How it works:**
1. HLoc localizes anchor frames (every 2-3 frames) to get absolute world poses
2. DPVO runs on all frames for fast incremental pose estimation
3. Umeyama similarity alignment maps DPVO trajectory → world coordinates (recovers scale)

```bash
modal run hloc_localization/backend/dpvo_app.py \
  --video-path data/IMG_4730.MOV \
  --reference-path hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz
```

## Benchmarks (A100 GPU)

### Per-frame latency

| Component | hloc | DPVO |
|-----------|------|------|
| **Total per frame** | **13.8s** | **31.8ms** |
| **Throughput** | **0.07 fps** | **31.5 fps** |

### hloc breakdown

| Step | Time | % of total |
|------|------|------------|
| Image decode | 0.02s | 0.2% |
| SuperPoint extraction | 0.3s | 4.7% |
| **NetVLAD descriptor** | **11.4s** | **82.5%** |
| Top-K retrieval | 0.16s | 1.2% |
| Feature merge | 0.07s | 0.5% |
| LightGlue matching (10 pairs) | 1.4s | 10.2% |
| 2D-3D correspondences | 0.06s | 0.5% |
| PnP+RANSAC | 0.02s | 0.1% |

NetVLAD is the bottleneck — hloc re-initializes the model each frame. With model caching or a lighter global descriptor, hloc could drop to ~1.7s/frame.

### DPVO breakdown

| Step | Time |
|------|------|
| Preprocessing (resize + GPU transfer) | 1.6ms |
| DPVO inference | 31.8ms |
| Init (one-time) | 123ms |
| Terminate/finalize | 198ms |
| Warmup (first 2 frames) | ~650ms |

### Benchmarking scripts

```bash
# hloc per-frame benchmark
modal run hloc_localization/backend/benchmark.py \
  --video data/IMG_4724.mov \
  --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz \
  --fps 2 --max-frames 10

# DPVO per-frame benchmark
modal run hloc_localization/backend/benchmark_dpvo.py \
  --video data/IMG_4724.mov --fps 15 --max-frames 100
```

### Tradeoffs

| | hloc | DPVO |
|---|---|---|
| Pose type | Absolute (world coords) | Relative (needs anchors) |
| Drift | None | Accumulates over time |
| Speed | ~0.07 fps | ~31 fps |
| Best for | Anchor frames, loop closure | Real-time tracking between anchors |

### Reducing DPVO drift

**Parameter tuning** (in `dpvo_app.py`):
- `PATCHES_PER_FRAME`: 48→96 (default) — denser features, more robust tracking
- `BUFFER_SIZE`: 256→512+ — larger optimization window
- `OPTIMIZATION_WINDOW`: increase for tighter bundle adjustment
- `PATCH_LIFETIME`: increase for longer temporal consistency

**Architectural:**
- More frequent HLoc anchors (every 2 frames instead of 3)
- Piecewise alignment — align trajectory segments between anchors independently
- Outlier rejection — detect and smooth pose jumps
- Loop closure — re-optimize when camera revisits a location (see DPV-SLAM)
- Depth constraints — use monocular depth estimation to anchor metric scale

## Dependencies

- **Modal** (GPU compute): `pip install modal`
- **Local**: `pip install numpy opencv-python trimesh viser scipy`
- hloc, pycolmap, torch, LightGlue are installed in the Modal image automatically
