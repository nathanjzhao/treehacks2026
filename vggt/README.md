# VGGT on Modal

3D reconstruction from video using [VGGT](https://github.com/facebookresearch/vggt) (Meta, CVPR 2025) on A100 GPUs.

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
| `--depth` | True | Use depth branch (denser) vs pointmap (faster) |

```bash
# High quality — slow, large file
modal run app.py --video-path vid.mov --fps 3 --conf 10

# Fast preview
modal run app.py --video-path vid.mov --fps 1 --conf 50 --no-depth
```

## Local 3D viewer

View GLB point clouds locally with orbit controls and a chat sidebar (Viser-based):

```bash
pip install viser==0.2.23 trimesh numpy
python viewer.py examples/IMG_4708.glb
```

Open `http://localhost:8080`. Use `--downsample N` to reduce point count for faster loading.

## Gradio demo

```bash
modal serve app.py
```
