"""
Depth Anything V2 on Modal — Monocular depth estimation.

Runs per-frame depth estimation on video using an A100. Returns
a colorized depth video and raw 16-bit depth frames packed as bytes.

Deploy:  modal deploy depthanything/app.py
Dev:     modal serve depthanything/app.py
Video:   modal run depthanything/app.py --video-path ~/video.mp4
"""

import pathlib

import modal

app = modal.App("depthanything")

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

HF_WEIGHT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large"
    "/resolve/main/depth_anything_v2_vitl.pth"
)

depthanything_image = (
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
        "git clone https://github.com/DepthAnything/Depth-Anything-V2.git /opt/DepthAnythingV2",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
    })
    # Pre-download model weights into the image
    .run_commands(
        "mkdir -p /opt/DepthAnythingV2/checkpoints && "
        f"python -c \""
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download("
        f"  repo_id='depth-anything/Depth-Anything-V2-Large',"
        f"  filename='depth_anything_v2_vitl.pth',"
        f"  local_dir='/opt/DepthAnythingV2/checkpoints'"
        f")\"",
    )
    # Warm up model on GPU to validate it loads
    .run_commands(
        "python -c \""
        "import sys; sys.path.insert(0, '/opt/DepthAnythingV2'); "
        "import torch; "
        "from depth_anything_v2.dpt import DepthAnythingV2; "
        "model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]); "
        "model.load_state_dict(torch.load('/opt/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu')); "
        "model = model.to('cuda').eval(); "
        "print('Depth Anything V2 vitl loaded OK')\"",
        gpu="any",
    )
)

with depthanything_image.imports():
    import os
    import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_frames(video_bytes: bytes, tmpdir: str, target_fps: int):
    """Extract frames from video bytes at target FPS. Returns (image_paths, source_fps, frame_w, frame_h)."""
    import cv2

    video_path = os.path.join(tmpdir, "input.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    images_dir = os.path.join(tmpdir, "images")
    os.makedirs(images_dir)

    vs = cv2.VideoCapture(video_path)
    source_fps = vs.get(cv2.CAP_PROP_FPS)
    frame_w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(source_fps / target_fps))

    image_paths = []
    count = 0
    frame_num = 0
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            path = os.path.join(images_dir, f"{frame_num:06d}.png")
            cv2.imwrite(path, frame)
            image_paths.append(path)
            frame_num += 1
    vs.release()
    print(f"Extracted {len(image_paths)} frames from {count} total (interval={frame_interval})")
    return image_paths, source_fps, frame_w, frame_h


def _load_model(encoder: str = ENCODER):
    """Load the Depth Anything V2 model (cached in the container image)."""
    import torch

    sys.path.insert(0, "/opt/DepthAnythingV2")
    from depth_anything_v2.dpt import DepthAnythingV2

    cfg = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    ckpt = f"/opt/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to("cuda").eval()
    print(f"Loaded Depth Anything V2 ({encoder})")
    return model


# ---------------------------------------------------------------------------
# Batch video depth estimation
# ---------------------------------------------------------------------------

@app.function(
    image=depthanything_image,
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
    Run Depth Anything V2 on a video. Returns colorized depth MP4 and
    raw 16-bit depth frames as a packed numpy array.

    Args:
        video_bytes: Raw video file content.
        target_fps: Frames to extract per second.
        input_size: Inference resolution (higher = finer detail, slower).
        grayscale: If True, output grayscale depth instead of colormap.
    """
    import tempfile

    import cv2
    import numpy as np
    import torch

    model = _load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths, source_fps, frame_w, frame_h = _extract_frames(
            video_bytes, tmpdir, target_fps
        )

        if len(image_paths) == 0:
            raise ValueError("No frames extracted from video")

        out_fps = min(target_fps, source_fps)

        # Output video
        out_path = os.path.join(tmpdir, "depth_vis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (frame_w, frame_h))

        print(f"Running depth estimation on {len(image_paths)} frames (input_size={input_size})...")
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)

            with torch.cuda.amp.autocast():
                depth = model.infer_image(frame, input_size)

            # Normalize to 0-255 for visualization
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)

            if grayscale:
                depth_vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            else:
                depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

            # Resize to original frame dimensions
            depth_vis = cv2.resize(depth_vis, (frame_w, frame_h))
            writer.write(depth_vis)

            if i % 10 == 0:
                print(f"  Frame {i}/{len(image_paths)}")

        writer.release()

        # Re-encode with ffmpeg for broader compatibility
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

    print(f"Output: {len(video_out) / 1024 / 1024:.1f} MB video, {len(image_paths)} frames")
    return {
        "video": video_out,
        "num_frames": len(image_paths),
        "source_fps": source_fps,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


# ---------------------------------------------------------------------------
# Single-frame depth estimation (for streaming / auxiliary use)
# ---------------------------------------------------------------------------

@app.function(
    image=depthanything_image,
    gpu="A100",
    timeout=300,
    memory=32768,
)
def predict_frame(
    frame_bytes: bytes,
    input_size: int = 518,
) -> dict:
    """
    Run depth estimation on a single JPEG/PNG frame.

    Args:
        frame_bytes: Raw image bytes (JPEG or PNG).
        input_size: Inference resolution.

    Returns:
        Dictionary with raw depth (float32 bytes), shape, and colorized JPEG.
    """
    import cv2
    import numpy as np
    import torch

    model = _load_model()

    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")

    with torch.cuda.amp.autocast():
        depth = model.infer_image(frame, input_size)

    # Colorized visualization
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)

    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    _, vis_buf = cv2.imencode(".jpg", depth_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    torch.cuda.empty_cache()

    return {
        "depth_raw": depth.astype(np.float32).tobytes(),
        "depth_vis": vis_buf.tobytes(),
        "height": depth.shape[0],
        "width": depth.shape[1],
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    video_path: str = "",
    fps: int = 15,
    input_size: int = 518,
    grayscale: bool = False,
    outdir: str = "",
):
    """
    Run Depth Anything V2 on one or more local video files in parallel.

    Pass a single file, a comma-separated list, or a glob pattern.
    Results go to --outdir (default: examples/ next to this script).

    Args:
        video_path: Path(s) to video file(s), comma-separated or glob.
        fps: Frames to extract per second.
        input_size: Inference resolution (518 default, higher = finer).
        grayscale: Output grayscale depth instead of colormap.
        outdir: Output directory for results.
    """
    import glob as globmod

    # Resolve output dir
    if outdir:
        out_dir = pathlib.Path(outdir).expanduser().resolve()
    else:
        out_dir = pathlib.Path(__file__).parent / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
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

    print(f"Processing {len(files)} videos in parallel (fps={fps}, input_size={input_size})")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    # Launch all jobs in parallel using starmap
    args = [
        (f.read_bytes(), fps, input_size, grayscale)
        for f in files
    ]

    results = list(predict_video.starmap(args))

    # Save results
    for f, result in zip(files, results):
        out_path = out_dir / f"{f.stem}_depth.mp4"
        out_path.write_bytes(result["video"])

        print(
            f"  {f.name} -> {out_path.name} "
            f"({len(result['video']) / 1024 / 1024:.1f} MB, "
            f"{result['num_frames']} frames)"
        )

    print(f"\nAll done! Results in {out_dir}")
