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

## Dependencies

- **Modal** (GPU compute): `pip install modal`
- **Local**: `pip install numpy opencv-python trimesh viser scipy`
- hloc, pycolmap, torch, LightGlue are installed in the Modal image automatically
