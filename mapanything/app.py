"""
MapAnything on Modal — Universal Metric 3D Reconstruction (Meta/CMU, 2025).

Runs inference on an A100 with model weights baked into the image.
Accepts video + optional auxiliary geometric inputs (intrinsics, poses, depth).

Deploy:  modal deploy mapanything/app.py
Dev:     modal serve mapanything/app.py
Video:   modal run mapanything/app.py --video-path ~/Desktop/video.mov
"""

import json
import pathlib

import modal

app = modal.App("mapanything")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

_local_dir = pathlib.Path(__file__).parent

mapanything_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "Pillow",
        "pillow-heif",
        "huggingface_hub",
        "safetensors",
        "opencv-python",
        "trimesh",
        "tqdm",
        "hydra-core",
        "omegaconf",
        "scipy",
        "requests",
        "plyfile",
        "python-box",
        "natsort",
        "orjson",
        "einops",
        "tensorboard",
        "matplotlib",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/map-anything.git /opt/mapanything",
        "cd /opt/mapanything && pip install -e .",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
    })
    # Pre-download model weights into the image
    .run_commands(
        "python -c \""
        "from mapanything.models import MapAnything; "
        "MapAnything.from_pretrained('facebook/map-anything'); "
        "print('MapAnything weights downloaded')\"",
        gpu="any",
    )
)

with mapanything_image.imports():
    import os
    import sys


@app.function(
    image=mapanything_image,
    gpu="A100",
    timeout=900,
    memory=32768,
)
def predict_video(
    video_bytes: bytes,
    target_fps: int = 2,
    conf_thres: float = 25.0,
    intrinsics_json: str | None = None,
    poses_json: str | None = None,
) -> dict:
    """
    Run MapAnything on a video. Extracts frames, runs inference, returns GLB.

    Args:
        video_bytes: Raw video file content.
        target_fps: Frames to extract per second (default: 2).
        conf_thres: Confidence percentile threshold — lower = more points (default: 25).
        intrinsics_json: Optional JSON string with camera intrinsics (3x3 matrix).
        poses_json: Optional JSON string with per-frame camera poses (list of 4x4 matrices).
    """
    import tempfile

    import cv2
    import numpy as np
    import torch
    import trimesh

    sys.path.insert(0, "/opt/mapanything")
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda"

    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save video to temp file
        video_path = os.path.join(tmpdir, "input.mov")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Extract frames at target_fps
        images_dir = os.path.join(tmpdir, "images")
        os.makedirs(images_dir)

        vs = cv2.VideoCapture(video_path)
        source_fps = vs.get(cv2.CAP_PROP_FPS)
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

        if len(image_paths) == 0:
            raise ValueError("No frames extracted from video")

        # Load and preprocess images
        views = load_images(images_dir)

        # Inject auxiliary geometric inputs if provided
        if intrinsics_json is not None:
            intrinsics = np.array(json.loads(intrinsics_json), dtype=np.float32)
            if intrinsics.shape == (3, 3):
                # Broadcast to all views
                for i in range(len(views)):
                    views[i]["intrinsics"] = torch.from_numpy(intrinsics).to(device)
            print(f"Using provided camera intrinsics")

        if poses_json is not None:
            poses = json.loads(poses_json)
            for i, pose in enumerate(poses):
                if i < len(views):
                    pose_mat = np.array(pose, dtype=np.float32)
                    views[i]["camera_poses"] = torch.from_numpy(pose_mat).to(device)
            print(f"Using provided camera poses for {min(len(poses), len(views))} frames")

        # Run inference (no manual autocast — MapAnything post-processing
        # uses linalg ops that require float32)
        print("Running MapAnything inference...")
        with torch.no_grad():
            predictions = model.infer(views, memory_efficient_inference=True)

        # predictions is a list of dicts (one per view), each with:
        #   pts3d (B,H,W,3), conf (B,H,W), camera_poses (B,4,4),
        #   img_no_norm (B,H,W,3), intrinsics (B,3,3), etc.
        print(f"Got {len(predictions)} prediction dicts")
        print(f"Keys in first prediction: {list(predictions[0].keys())}")

        all_points_list = []
        all_colors_list = []
        all_conf_list = []
        cam_poses_list = []

        for i, pred in enumerate(predictions):
            # pts3d: (B, H, W, 3) — squeeze batch dim
            pts = pred["pts3d"]
            if isinstance(pts, torch.Tensor):
                pts = pts.cpu().numpy()
            if pts.ndim == 4 and pts.shape[0] == 1:
                pts = pts.squeeze(0)  # (H, W, 3)
            elif pts.ndim == 4:
                pts = pts[0]
            all_points_list.append(pts.reshape(-1, 3))

            # conf: (B, H, W) — squeeze batch dim
            c = pred["conf"]
            if isinstance(c, torch.Tensor):
                c = c.cpu().numpy()
            if c.ndim == 3 and c.shape[0] == 1:
                c = c.squeeze(0)
            elif c.ndim == 3:
                c = c[0]
            all_conf_list.append(c.reshape(-1))

            # Colors from denormalized images if available, else from disk
            H, W = pts.shape[0], pts.shape[1]
            if "img_no_norm" in pred:
                img = pred["img_no_norm"]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if img.ndim == 4:
                    img = img[0]
                # img_no_norm is float [0,1] — convert to uint8
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H))
            else:
                img = cv2.imread(image_paths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H))
            all_colors_list.append(img.reshape(-1, 3))

            # Camera poses
            if "camera_poses" in pred:
                cp = pred["camera_poses"]
                if isinstance(cp, torch.Tensor):
                    cp = cp.cpu().numpy()
                if cp.ndim == 3:
                    cp = cp[0]
                cam_poses_list.append(cp)

        all_points = np.concatenate(all_points_list, axis=0)
        all_colors = np.concatenate(all_colors_list, axis=0)
        all_conf = np.concatenate(all_conf_list, axis=0)

        # Filter by confidence threshold (percentile-based)
        if conf_thres > 0:
            threshold = np.percentile(all_conf, conf_thres)
            mask = all_conf >= threshold
            all_points = all_points[mask]
            all_colors = all_colors[mask]
            print(f"Confidence filter: kept {mask.sum():,}/{len(mask):,} points (threshold={threshold:.4f})")

        # Remove invalid points (NaN, Inf)
        valid = np.isfinite(all_points).all(axis=1)
        if valid.sum() < len(valid):
            all_points = all_points[valid]
            all_colors = all_colors[valid]
            print(f"Removed {(~valid).sum():,} invalid points")

        # Remove statistical outliers
        centroid = np.median(all_points, axis=0)
        dists = np.linalg.norm(all_points - centroid, axis=1)
        dist_thresh = np.percentile(dists, 99)
        inlier = dists < dist_thresh
        all_points = all_points[inlier]
        all_colors = all_colors[inlier]

        print(f"Final point cloud: {len(all_points):,} points")

        # Build GLB scene
        scene = trimesh.Scene()

        cloud = trimesh.PointCloud(
            vertices=all_points,
            colors=np.column_stack([all_colors, np.full(len(all_colors), 255, dtype=np.uint8)]),
        )
        scene.add_geometry(cloud, geom_name="pointcloud")

        # Add camera frustums
        for i, pose in enumerate(cam_poses_list[:50]):
            if pose.shape == (4, 4):
                cone = trimesh.creation.cone(radius=0.02, height=0.05)
                cone.apply_transform(pose)
                cone.visual.face_colors = [100, 100, 255, 180]
                scene.add_geometry(cone, geom_name=f"camera_{i:03d}")

        glb_path = os.path.join(tmpdir, "output.glb")
        scene.export(file_obj=glb_path)

        with open(glb_path, "rb") as f:
            glb_bytes = f.read()

    print(f"GLB size: {len(glb_bytes) / 1024 / 1024:.1f} MB")
    return {
        "glb": glb_bytes,
        "num_frames": len(image_paths),
        "source_fps": source_fps,
        "num_points": len(all_points),
    }


@app.local_entrypoint()
def main(
    video_path: str,
    fps: int = 2,
    conf: float = 25.0,
    intrinsics: str = "",
    poses: str = "",
):
    """
    Run MapAnything on a local video file. Saves .glb next to the input.

    Args:
        video_path: Path to video file.
        fps: Frames to extract per second.
        conf: Confidence percentile threshold (lower = more points).
        intrinsics: Path to JSON file with 3x3 camera intrinsics matrix.
        poses: Path to JSON file with list of 4x4 camera pose matrices.
    """
    p = pathlib.Path(video_path).expanduser().resolve()
    print(f"Reading {p.name} ({p.stat().st_size / 1024:.0f} KB)")
    print(f"Settings: fps={fps}, conf_thres={conf}")

    # Load optional auxiliary inputs
    intrinsics_json = None
    if intrinsics:
        intrinsics_path = pathlib.Path(intrinsics).expanduser().resolve()
        intrinsics_json = intrinsics_path.read_text()
        print(f"Using intrinsics from {intrinsics_path.name}")

    poses_json = None
    if poses:
        poses_path = pathlib.Path(poses).expanduser().resolve()
        poses_json = poses_path.read_text()
        print(f"Using poses from {poses_path.name}")

    result = predict_video.remote(
        p.read_bytes(),
        target_fps=fps,
        conf_thres=conf,
        intrinsics_json=intrinsics_json,
        poses_json=poses_json,
    )

    out = p.with_suffix(".glb")
    out.write_bytes(result["glb"])
    print(f"Wrote {out} ({len(result['glb']) / 1024 / 1024:.1f} MB)")
    print(f"Processed {result['num_frames']} frames from {result['source_fps']:.1f} fps video")
    print(f"Point cloud: {result['num_points']:,} points")
