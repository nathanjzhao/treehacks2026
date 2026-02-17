# Viewers

All viewers require the conda base environment:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate base
```

## Viewer Summary

| Port | Viewer | Description |
|------|--------|-------------|
| 8890 | Locate | 3D object finder — GLB point cloud + raycasted object positions |
| 8891 | Segmentation | Side-by-side source / tracking / masked depth video |
| 8892 | Depth | Side-by-side source / depth video |
| 8893 | MapAnything | 3D point cloud browser (viser) |
| 8895 | OpenFunGraph | 3D scene graph with objects, parts, edges (viser) |

---

## 1. MapAnything Viewer (port 8893)

**What it shows:** Interactive 3D point cloud reconstructions with camera frustums.

### Generate data

```bash
modal run reconstruction/app.py --video-path data/IMG_4717.MOV
```

**Outputs:** `data/IMG_4717.glb` (move to `data/mapanything/`)

### Run viewer

```bash
python3 reconstruction/viewer.py data/mapanything/ --downsample 80 --port 8893 --viser-port 8894
```

**Inputs:**
- Directory of `.glb` files (dropdown to switch between them)
- `--downsample`: point skip factor (higher = fewer points, faster load)

---

## 2. Depth Viewer (port 8892)

**What it shows:** Side-by-side original video and depth estimation.

### Generate data (per-frame)

```bash
modal run depthanything/app.py --video-path data/IMG_4718.MOV
```

**Outputs:** `depthanything/examples/{stem}_depth.mp4`

### Generate data (temporally consistent)

```bash
modal run depthanything/video_app.py --video-path data/IMG_4717.MOV
```

**Outputs:** `depthanything/examples/{stem}_depth.mp4` (same format, replaces per-frame)

### Run viewer

```bash
python3 depthanything/viewer.py depthanything/examples/ --source-dir data/ --port 8892
```

**Inputs:**
- Directory of `*_depth.mp4` files (dropdown to switch)
- `--source-dir`: where to find matching source videos (e.g. `data/`)
- Without `--source-dir`, both panels show depth

---

## 3. Segmentation Viewer (port 8891)

**What it shows:** Three panels — original source, SAM2 tracking overlay, masked depth.

### Generate tracking data

```bash
modal run segmentation/app.py --video-path data/IMG_4723.MOV --text-prompt "person. chair."
```

**Outputs** to `data/segmentation/`:
- `{stem}_tracked.mp4` — video with tracking overlay
- `{stem}_detections.json` — per-frame bounding boxes

### Generate depth + segmentation composite

```bash
modal run segmentation/depth_app.py --video-path data/IMG_4723.MOV --text-prompt "person. chair."
```

**Outputs** to `data/segmentation/`:
- `{stem}_masked_depth.mp4` — depth only on segmented objects
- `{stem}_composite.mp4` — dimmed full depth + bright object depth + outlines
- `{stem}_seg_depth.json` — per-frame detections with depth

### Run viewer

```bash
python3 segmentation/seg_viewer.py data/segmentation/ --source-dir data/ --port 8891
```

**Inputs:**
- Directory containing `*_tracked.mp4`, `*_masked_depth.mp4`, `*_composite.mp4`
- `--source-dir`: where to find original videos
- Defaults to IMG_4723

---

## 4. OpenFunGraph Viewer (port 8895)

**What it shows:** 3D scene graph — labeled objects with point clouds, parts, and relationship edges.

### Generate data

```bash
modal run scenegraph/app.py --video-path data/IMG_4717.MOV
```

**Outputs:** `data/IMG_4717.json` (scene graph with objects, parts, edges, camera poses)

### Run viewer

```bash
python3 scenegraph/viewer.py data/IMG_4717.json --port 8895 --viser-port 8896
```

**Inputs:**
- `.json` scene graph file (or directory of them — dropdown to switch)
- Existing scene graphs: `data/IMG_4717.json`, `data/IMG_4724.json`, `data/IMG_4739.json`, `data/IMG_4740.json`

---

## 5. Locate Viewer (port 8890)

**What it shows:** 3D point cloud with raycasted object positions (sphere markers), camera path playback, video panels.

### Prerequisites

Requires outputs from three pipelines:

1. **MapAnything** (GLB point cloud):
   ```bash
   modal run reconstruction/app.py --video-path data/IMG_4720.MOV
   ```

2. **HLoc reference** (SfM map for alignment):
   ```bash
   modal run hloc_localization/app.py --video-path data/IMG_4720.MOV
   ```
   Produces `hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz`

3. **Object localization** (3D object positions):
   ```bash
   modal run segmentation/locate_app.py \
     --video-path data/IMG_4730.MOV \
     --text-prompt "painting. chair." \
     --reference-path hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz
   ```
   Produces `data/segmentation/IMG_4730_objects3d.json`

### Run viewer

```bash
python3 segmentation/locate_viewer.py \
  data/mapanything/IMG_4720.glb \
  data/segmentation/IMG_4730_objects3d.json \
  --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz \
  --video data/IMG_4730.MOV \
  --results-dir data/segmentation/
```

---

## Start All

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate base

python3 segmentation/locate_viewer.py data/mapanything/IMG_4720.glb data/segmentation/IMG_4730_objects3d.json --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz --video data/IMG_4730.MOV --results-dir data/segmentation/ &
python3 segmentation/seg_viewer.py data/segmentation/ --source-dir data/ --port 8891 &
python3 depthanything/viewer.py depthanything/examples/ --source-dir data/ --port 8892 &
python3 reconstruction/viewer.py data/mapanything/ --downsample 80 --port 8893 --viser-port 8894 &
python3 scenegraph/viewer.py data/IMG_4717.json --port 8895 --viser-port 8896 &
```

## Stop All

```bash
pkill -f "viewer.py"
```
