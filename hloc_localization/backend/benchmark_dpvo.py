"""
Benchmark DPVO per-frame timing (pure odometry, no HLoc anchoring).

Usage:
  modal run hloc_localization/backend/benchmark_dpvo.py \
    --video data/IMG_4724.mov \
    --fps 15 --max-frames 100
"""

import pathlib
import time

import modal

bench_app = modal.App("dpvo-benchmark")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

dpvo_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git", "ffmpeg", "libgl1", "libglib2.0-0",
        "cmake", "build-essential", "wget", "unzip",
    )
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "opencv-python",
        "scipy",
        "tqdm",
        "Pillow",
        "einops",
        "pypose",
        "kornia",
        "yacs",
        "numba",
        "plyfile",
        "matplotlib",
    )
    .run_commands(
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CXX=g++ pip install git+https://github.com/princeton-vl/lietorch.git",
        gpu="any",
    )
    .run_commands(
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CC=gcc CXX=g++ pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu124.html --no-build-isolation",
        gpu="any",
    )
    .run_commands(
        "git clone https://github.com/princeton-vl/DPVO.git /opt/dpvo",
        "cd /opt/dpvo && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
        " && unzip -q eigen-3.4.0.zip -d thirdparty && rm eigen-3.4.0.zip",
    )
    .run_commands(
        "cd /opt/dpvo && TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CC=gcc CXX=g++ pip install . --no-build-isolation",
        gpu="any",
    )
    .run_commands(
        "cd /opt/dpvo && wget -q https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip"
        " && unzip -q models.zip && rm models.zip",
    )
    .env({"TORCH_HOME": "/opt/torch_cache"})
)


@bench_app.function(
    image=dpvo_image,
    gpu="A100",
    timeout=600,
    memory=32768,
)
def benchmark_dpvo(
    video_bytes: bytes,
    target_fps: float = 15.0,
    max_frames: int = 100,
) -> dict:
    """Benchmark DPVO per-frame timing."""
    import os
    import sys
    import tempfile

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, "/opt/dpvo")
    from dpvo.config import cfg as dpvo_cfg
    from dpvo.dpvo import DPVO

    # --- Extract frames ---
    with tempfile.NamedTemporaryFile(suffix=".mov", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name

    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / target_fps))

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        frame_count += 1
    cap.release()
    os.unlink(video_path)

    h, w = frames[0].shape[:2]
    print(f"Extracted {len(frames)} frames ({w}x{h}) at ~{target_fps}fps from {source_fps:.0f}fps source")

    # --- Setup ---
    dpvo_h, dpvo_w = 480, 640
    focal = max(h, w) * 1.2
    intrinsics = np.array([focal, focal, w / 2.0, h / 2.0], dtype=np.float32)
    sx, sy = dpvo_w / w, dpvo_h / h
    intrinsics[0] *= sx
    intrinsics[1] *= sy
    intrinsics[2] *= sx
    intrinsics[3] *= sy
    intrinsics_tensor = torch.from_numpy(intrinsics).cuda()

    print(f"DPVO resolution: {dpvo_w}x{dpvo_h}")
    print(f"Intrinsics: fx={intrinsics[0]:.1f} fy={intrinsics[1]:.1f} cx={intrinsics[2]:.1f} cy={intrinsics[3]:.1f}")

    # --- Initialize DPVO ---
    t_init_start = time.time()
    dpvo_cfg.merge_from_file("/opt/dpvo/config/default.yaml")
    dpvo_cfg.BUFFER_SIZE = 256
    dpvo_cfg.PATCHES_PER_FRAME = 48

    ckpt_path = None
    for p in ["/opt/dpvo/models/dpvo.pth", "/opt/dpvo/checkpoints/dpvo.pth", "/opt/dpvo/dpvo.pth"]:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        for root, dirs, files in os.walk("/opt/dpvo"):
            for f in files:
                if f.endswith(".pth"):
                    ckpt_path = os.path.join(root, f)
                    break
            if ckpt_path:
                break

    slam = DPVO(dpvo_cfg, network=ckpt_path, ht=dpvo_h, wd=dpvo_w, viz=False)
    t_init = time.time() - t_init_start
    print(f"DPVO init: {t_init:.3f}s")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Warmup: process first frame to trigger JIT/CUDA warmup ---
    resized = cv2.resize(frames[0], (dpvo_w, dpvo_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda()
    torch.cuda.synchronize()

    # --- Benchmark per-frame timing ---
    per_frame_times = []
    preprocess_times = []

    print(f"\nBenchmarking {len(frames)} frames...\n")
    t_total_start = time.time()

    for i, frame in enumerate(frames):
        # Preprocess
        t_pre = time.time()
        resized = cv2.resize(frame, (dpvo_w, dpvo_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda()
        torch.cuda.synchronize()
        preprocess_times.append(time.time() - t_pre)

        # DPVO inference
        t_frame = time.time()
        slam(i, img_tensor, intrinsics_tensor)
        torch.cuda.synchronize()
        per_frame_times.append(time.time() - t_frame)

        if (i + 1) % 25 == 0:
            avg = sum(per_frame_times[-25:]) / min(25, len(per_frame_times[-25:]))
            print(f"  Frame {i+1:4d}/{len(frames)} | last 25 avg: {avg*1000:.1f}ms ({1/avg:.1f} fps)")

    t_total_inference = time.time() - t_total_start

    # Terminate (includes final optimization)
    t_term = time.time()
    poses, tstamps = slam.terminate()
    t_terminate = time.time() - t_term

    if hasattr(poses, 'cpu'):
        poses = poses.cpu().numpy()

    # --- Print results ---
    import statistics

    print("\n" + "=" * 70)
    print("DPVO BENCHMARK RESULTS")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Frames: {len(frames)}, Resolution: {dpvo_w}x{dpvo_h}")
    print(f"Poses recovered: {len(poses)}")
    print(f"DPVO init time: {t_init:.3f}s")
    print(f"Terminate time: {t_terminate:.3f}s")

    # Skip first 5 frames (warmup)
    warmup = min(5, len(per_frame_times) - 1)
    steady = per_frame_times[warmup:]
    steady_pre = preprocess_times[warmup:]

    print(f"\nPer-frame DPVO inference (excluding first {warmup} warmup frames):")
    print(f"  Mean:   {statistics.mean(steady)*1000:.1f}ms  ({1/statistics.mean(steady):.1f} fps)")
    print(f"  Median: {statistics.median(steady)*1000:.1f}ms  ({1/statistics.median(steady):.1f} fps)")
    print(f"  Min:    {min(steady)*1000:.1f}ms")
    print(f"  Max:    {max(steady)*1000:.1f}ms")
    print(f"  Std:    {statistics.stdev(steady)*1000:.1f}ms")

    print(f"\nPreprocessing (resize + color convert + to GPU):")
    print(f"  Mean:   {statistics.mean(steady_pre)*1000:.1f}ms")

    total_per_frame = [a + b for a, b in zip(steady, steady_pre)]
    print(f"\nTotal per-frame (preprocess + inference):")
    print(f"  Mean:   {statistics.mean(total_per_frame)*1000:.1f}ms  ({1/statistics.mean(total_per_frame):.1f} fps)")

    print(f"\nEnd-to-end: {t_total_inference:.1f}s for {len(frames)} frames = {len(frames)/t_total_inference:.1f} fps")

    # First 5 frames breakdown (warmup)
    print(f"\nWarmup frames (first {warmup}):")
    for i in range(warmup):
        print(f"  Frame {i}: {per_frame_times[i]*1000:.1f}ms")

    return {
        "num_frames": len(frames),
        "num_poses": len(poses),
        "init_time": t_init,
        "terminate_time": t_terminate,
        "total_inference_time": t_total_inference,
        "mean_per_frame_ms": statistics.mean(steady) * 1000,
        "median_per_frame_ms": statistics.median(steady) * 1000,
        "mean_preprocess_ms": statistics.mean(steady_pre) * 1000,
        "mean_total_per_frame_ms": statistics.mean(total_per_frame) * 1000,
        "e2e_fps": len(frames) / t_total_inference,
        "steady_fps": 1 / statistics.mean(steady),
    }


@bench_app.local_entrypoint()
def main(
    video: str,
    fps: float = 15.0,
    max_frames: int = 100,
):
    """
    Benchmark DPVO per-frame timing.

    Usage:
      modal run hloc_localization/backend/benchmark_dpvo.py \
        --video data/IMG_4724.mov --fps 15
    """
    video_path = pathlib.Path(video).expanduser().resolve()
    print(f"Video: {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")

    video_bytes = video_path.read_bytes()

    t0 = time.time()
    result = benchmark_dpvo.remote(video_bytes, target_fps=fps, max_frames=max_frames)
    wall_time = time.time() - t0

    print(f"\nWall time (including Modal overhead): {wall_time:.1f}s")
    print(f"DPVO steady-state: {result['mean_per_frame_ms']:.1f}ms/frame = {result['steady_fps']:.1f} fps")
