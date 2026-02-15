"""
Video Depth Anything on Modal — Temporally-consistent depth estimation.

Uses Video Depth Anything (CVPR 2025) for temporally consistent depth maps,
replacing per-frame Depth Anything V2 with a spatial-temporal model.

Deploy:  modal deploy depthanything/video_app.py
Dev:     modal serve depthanything/video_app.py
Video:   modal run depthanything/video_app.py --video-path ~/video.mp4
"""

import pathlib

import modal

app = modal.App("video-depthanything")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

ENCODER = "vitl"
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

video_depth_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "Pillow",
        "huggingface_hub",
        "opencv-python",
        "tqdm",
        "scipy",
        "matplotlib",
    )
    .run_commands(
        "git clone https://github.com/DepthAnything/Video-Depth-Anything.git /opt/VideoDepthAnything",
    )
    .run_commands(
        "cd /opt/VideoDepthAnything && pip install -r requirements.txt",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
    })
    # Download model weights
    .run_commands(
        "mkdir -p /opt/VideoDepthAnything/checkpoints && "
        "python -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download("
        "  repo_id='depth-anything/Video-Depth-Anything-Large',"
        "  filename='video_depth_anything_vitl.pth',"
        "  local_dir='/opt/VideoDepthAnything/checkpoints'"
        ")\"",
    )
    # Validate model loads
    .run_commands(
        "python -c \""
        "import sys; sys.path.insert(0, '/opt/VideoDepthAnything'); "
        "import torch; "
        "from video_depth_anything.video_depth import VideoDepthAnything; "
        "model = VideoDepthAnything(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]); "
        "model.load_state_dict(torch.load('/opt/VideoDepthAnything/checkpoints/video_depth_anything_vitl.pth', map_location='cpu'), strict=True); "
        "model = model.to('cuda').eval(); "
        "print('Video Depth Anything vitl loaded OK')\"",
        gpu="any",
    )
)

with video_depth_image.imports():
    import os
    import sys


def _load_model(encoder: str = ENCODER):
    """Load Video Depth Anything model."""
    import torch

    sys.path.insert(0, "/opt/VideoDepthAnything")
    from video_depth_anything.video_depth import VideoDepthAnything

    cfg = MODEL_CONFIGS[encoder]
    model = VideoDepthAnything(**cfg)
    ckpt = f"/opt/VideoDepthAnything/checkpoints/video_depth_anything_{encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model = model.to("cuda").eval()
    print(f"Loaded Video Depth Anything ({encoder})")
    return model


def _extract_frames_as_array(video_bytes: bytes, tmpdir: str, target_fps: int):
    """Extract frames from video, return as numpy array (N, H, W, 3) BGR."""
    import cv2
    import numpy as np

    video_path = os.path.join(tmpdir, "input.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    vs = cv2.VideoCapture(video_path)
    source_fps = vs.get(cv2.CAP_PROP_FPS)
    frame_w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(source_fps / target_fps))

    frames = []
    count = 0
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            frames.append(frame)
    vs.release()

    print(f"Extracted {len(frames)} frames from {count} total (interval={frame_interval})")
    return np.stack(frames) if frames else np.zeros((0, frame_h, frame_w, 3), dtype=np.uint8), source_fps, frame_w, frame_h


@app.function(
    image=video_depth_image,
    gpu="A100",
    timeout=1800,
    memory=32768,
)
def predict_video(
    video_bytes: bytes,
    target_fps: int = 6,
    input_size: int = 518,
    grayscale: bool = False,
) -> dict:
    """
    Run Video Depth Anything on a video for temporally consistent depth.

    Args:
        video_bytes: Raw video file content.
        target_fps: Frames to extract per second.
        input_size: Inference resolution (higher = finer, slower).
        grayscale: If True, output grayscale depth instead of colormap.
    """
    import tempfile

    import cv2
    import numpy as np
    import torch

    model = _load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        frames, source_fps, frame_w, frame_h = _extract_frames_as_array(
            video_bytes, tmpdir, target_fps
        )

        if len(frames) == 0:
            raise ValueError("No frames extracted from video")

        out_fps = min(target_fps, source_fps)

        # Video Depth Anything processes all frames together for temporal consistency
        print(f"Running video depth estimation on {len(frames)} frames (input_size={input_size})...")
        with torch.no_grad():
            depths, _ = model.infer_video_depth(
                frames, out_fps, input_size=input_size, device="cuda"
            )
        # depths: (N, H, W) numpy array of relative depth values

        print(f"Depth estimation complete. Writing output video...")
        out_path = os.path.join(tmpdir, "depth_vis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (frame_w, frame_h))

        for i in range(len(depths)):
            depth = depths[i]
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)

            if grayscale:
                depth_vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            else:
                depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

            depth_vis = cv2.resize(depth_vis, (frame_w, frame_h))
            writer.write(depth_vis)

        writer.release()

        # Re-encode for compatibility
        final_path = os.path.join(tmpdir, "output.mp4")
        os.system(
            f"ffmpeg -y -i {out_path} -c:v libx264 -preset fast -crf 23 "
            f"-pix_fmt yuv420p {final_path} 2>/dev/null"
        )
        if not os.path.exists(final_path):
            final_path = out_path

        with open(final_path, "rb") as f:
            video_out = f.read()

    torch.cuda.empty_cache()

    print(f"Output: {len(video_out) / 1024 / 1024:.1f} MB video, {len(depths)} frames")
    return {
        "video": video_out,
        "num_frames": len(depths),
        "source_fps": source_fps,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


@app.local_entrypoint()
def main(
    video_path: str = "",
    fps: int = 15,
    input_size: int = 518,
    grayscale: bool = False,
    outdir: str = "",
):
    """
    Run Video Depth Anything on local video files (temporally consistent).

    Pass a single file, a comma-separated list, or a glob pattern.
    Results go to --outdir (default: depthanything/examples/).
    """
    import glob as globmod

    if outdir:
        out_dir = pathlib.Path(outdir).expanduser().resolve()
    else:
        out_dir = pathlib.Path(__file__).parent / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for part in video_path.split(","):
        part = part.strip()
        expanded = globmod.glob(str(pathlib.Path(part).expanduser()))
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(part)

    files = [pathlib.Path(p).expanduser().resolve() for p in paths if p]
    files = [f for f in files if f.exists()]

    if not files:
        print("No video files found. Pass --video-path <file_or_glob>")
        return

    print(f"Processing {len(files)} videos with Video Depth Anything (fps={fps}, input_size={input_size})")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    args = [
        (f.read_bytes(), fps, input_size, grayscale)
        for f in files
    ]

    results = list(predict_video.starmap(args))

    for f, result in zip(files, results):
        out_path = out_dir / f"{f.stem}_depth.mp4"
        out_path.write_bytes(result["video"])
        print(
            f"  {f.name} -> {out_path.name} "
            f"({len(result['video']) / 1024 / 1024:.1f} MB, "
            f"{result['num_frames']} frames)"
        )

    print(f"\nAll done! Results in {out_dir}")
