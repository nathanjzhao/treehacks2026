"""
DPVO visual odometry on Modal — near-real-time camera tracking.

Uses HLoc for anchor-frame localization, then DPVO for incremental pose estimation.
All DPVO poses are aligned to world coordinates via the anchor.

Deploy:  modal deploy hloc_localization/backend/dpvo_app.py
Dev:     modal serve hloc_localization/backend/dpvo_app.py
Run:     modal run hloc_localization/backend/dpvo_app.py \
           --video-path data/IMG_4730.MOV \
           --reference-path hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz
"""

import pathlib

import modal

app = modal.App("dpvo-odometry")

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
        "pycolmap>=3.10.0",
    )
    # Install lietorch (DPVO dependency — needs CUDA + g++ at build time)
    # Set TORCH_CUDA_ARCH_LIST to cover A10G/A100/H100 runtime GPUs
    .run_commands(
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CXX=g++ pip install git+https://github.com/princeton-vl/lietorch.git",
        gpu="any",
    )
    # Install torch_scatter (needs torch + gcc/g++ at build time)
    .run_commands(
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CC=gcc CXX=g++ pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu124.html --no-build-isolation",
        gpu="any",
    )
    # Clone DPVO, download Eigen, and install
    .run_commands(
        "git clone https://github.com/princeton-vl/DPVO.git /opt/dpvo",
        "cd /opt/dpvo && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
        " && unzip -q eigen-3.4.0.zip -d thirdparty && rm eigen-3.4.0.zip",
    )
    .run_commands(
        "cd /opt/dpvo && TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CC=gcc CXX=g++ pip install . --no-build-isolation",
        gpu="any",
    )
    # Download DPVO checkpoint from Dropbox
    .run_commands(
        "cd /opt/dpvo && wget -q https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip"
        " && unzip -q models.zip && rm models.zip",
    )
    .env({
        "TORCH_HOME": "/opt/torch_cache",
    })
)

with dpvo_image.imports():
    import io
    import os
    import sys
    import tarfile
    import tempfile


def _quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z)."""
    import numpy as np
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quaternion_inverse(q):
    """Inverse of a unit quaternion (w, x, y, z)."""
    import numpy as np
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_to_matrix(q):
    """Quaternion (w, x, y, z) to 3x3 rotation matrix."""
    import numpy as np
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def _pose_to_4x4(translation, quat_wxyz):
    """Build a 4x4 transform from translation and quaternion (w,x,y,z)."""
    import numpy as np
    T = np.eye(4)
    T[:3, :3] = _quat_to_matrix(quat_wxyz)
    T[:3, 3] = translation
    return T


def _matrix_to_quat(R):
    """3x3 rotation matrix to quaternion (w, x, y, z)."""
    import numpy as np
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _dpvo_pose_to_cam_position(dpvo_pose):
    """Extract camera position from a DPVO pose [tx,ty,tz,qx,qy,qz,qw] (cam-to-world)."""
    import numpy as np
    return dpvo_pose[:3].copy()


def _hloc_pose_to_cam_position(hloc_pose):
    """Extract camera position from HLoc pose dict (COLMAP cam_from_world convention)."""
    import numpy as np
    q = np.array([hloc_pose["qw"], hloc_pose["qx"], hloc_pose["qy"], hloc_pose["qz"]])
    t = np.array([hloc_pose["tx"], hloc_pose["ty"], hloc_pose["tz"]])
    R = _quat_to_matrix(q)
    return -R.T @ t  # camera position in world coords


def _umeyama_similarity(src, dst):
    """
    Umeyama alignment: find s, R, t such that dst ≈ s * R @ src + t.

    src, dst: (N, 3) arrays of corresponding 3D points.
    Returns: scale (float), R (3x3), t (3,)
    """
    import numpy as np

    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    var_src = np.sum(src_centered ** 2) / n

    # Cross-covariance
    H = (dst_centered.T @ src_centered) / n

    U, D, Vt = np.linalg.svd(H)

    # Handle reflection
    d = np.linalg.det(U) * np.linalg.det(Vt)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / var_src if var_src > 1e-12 else 1.0
    t = mu_dst - scale * R @ mu_src

    return scale, R, t


def _align_dpvo_to_world(dpvo_poses, dpvo_tstamps, hloc_anchors):
    """
    Align DPVO trajectory to COLMAP world coordinates using similarity transform.

    Uses Umeyama alignment with 2+ HLoc-localized frames to recover
    rotation, translation, AND scale (critical for monocular DPVO).

    dpvo_poses: (N, 7) array [tx, ty, tz, qx, qy, qz, qw] — DPVO cam-to-world
    dpvo_tstamps: (N,) frame indices from DPVO
    hloc_anchors: list of (frame_idx, hloc_pose_dict) — 2+ HLoc-localized frames
    """
    import numpy as np

    # Collect corresponding points: DPVO cam positions ↔ HLoc cam positions
    dpvo_pts = []
    hloc_pts = []
    for frame_idx, hloc_pose in hloc_anchors:
        # Find closest DPVO timestamp
        dpvo_idx = int(np.argmin(np.abs(dpvo_tstamps - frame_idx)))
        dpvo_pos = _dpvo_pose_to_cam_position(dpvo_poses[dpvo_idx])
        hloc_pos = _hloc_pose_to_cam_position(hloc_pose)
        dpvo_pts.append(dpvo_pos)
        hloc_pts.append(hloc_pos)
        print(f"  Anchor frame {frame_idx}: DPVO pos={dpvo_pos}, HLoc pos={hloc_pos}")

    dpvo_pts = np.array(dpvo_pts)
    hloc_pts = np.array(hloc_pts)

    if len(hloc_anchors) >= 3:
        # Full Umeyama similarity alignment
        scale, R_align, t_align = _umeyama_similarity(dpvo_pts, hloc_pts)
    elif len(hloc_anchors) == 2:
        # 2 points: compute scale from distance ratio, rotation from first anchor
        dpvo_dist = np.linalg.norm(dpvo_pts[1] - dpvo_pts[0])
        hloc_dist = np.linalg.norm(hloc_pts[1] - hloc_pts[0])
        scale = hloc_dist / dpvo_dist if dpvo_dist > 1e-8 else 1.0

        # Use first anchor's rotation for alignment
        anchor_dpvo = dpvo_poses[int(np.argmin(np.abs(dpvo_tstamps - hloc_anchors[0][0])))]
        q_xyzw = anchor_dpvo[3:]
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        R_dpvo = _quat_to_matrix(q_wxyz)

        hloc_q = np.array([hloc_anchors[0][1]["qw"], hloc_anchors[0][1]["qx"],
                           hloc_anchors[0][1]["qy"], hloc_anchors[0][1]["qz"]])
        R_hloc_cfw = _quat_to_matrix(hloc_q)
        R_hloc_c2w = R_hloc_cfw.T  # camera-to-world

        # R_align maps DPVO orientation to world orientation
        R_align = R_hloc_c2w @ R_dpvo.T
        t_align = hloc_pts[0] - scale * R_align @ dpvo_pts[0]
    else:
        # Single anchor fallback (no scale recovery — BAD for monocular)
        scale = 1.0
        anchor_dpvo = dpvo_poses[int(np.argmin(np.abs(dpvo_tstamps - hloc_anchors[0][0])))]
        q_xyzw = anchor_dpvo[3:]
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        R_dpvo = _quat_to_matrix(q_wxyz)

        hloc_q = np.array([hloc_anchors[0][1]["qw"], hloc_anchors[0][1]["qx"],
                           hloc_anchors[0][1]["qy"], hloc_anchors[0][1]["qz"]])
        R_hloc_cfw = _quat_to_matrix(hloc_q)
        R_hloc_c2w = R_hloc_cfw.T
        R_align = R_hloc_c2w @ R_dpvo.T
        t_align = _hloc_pose_to_cam_position(hloc_anchors[0][1]) - R_align @ dpvo_pts[0]

    print(f"  Alignment: scale={scale:.4f}, "
          f"translation={t_align}, "
          f"rotation det={np.linalg.det(R_align):.6f}")

    # Verify alignment on anchor points
    for i, (frame_idx, hloc_pose) in enumerate(hloc_anchors):
        aligned = scale * R_align @ dpvo_pts[i] + t_align
        hloc_pos = hloc_pts[i]
        err = np.linalg.norm(aligned - hloc_pos)
        print(f"  Anchor {frame_idx} alignment error: {err:.6f}")

    # Apply similarity transform to all DPVO poses
    world_poses = []
    trajectory_points = []
    anchor_frame_set = {idx for idx, _ in hloc_anchors}

    for i in range(len(dpvo_poses)):
        p = dpvo_poses[i]
        dpvo_t = p[:3]
        q_xyzw = p[3:]
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        R_dpvo_i = _quat_to_matrix(q_wxyz)

        # Similarity transform on cam-to-world pose:
        # position: s * R_align @ dpvo_t + t_align
        # orientation: R_align @ R_dpvo_i
        world_pos = scale * R_align @ dpvo_t + t_align
        world_R = R_align @ R_dpvo_i

        trajectory_points.append([float(world_pos[0]), float(world_pos[1]), float(world_pos[2])])

        # Convert cam-to-world → cam_from_world (COLMAP convention)
        R_cfw = world_R.T
        t_cfw = -R_cfw @ world_pos
        q_cfw = _matrix_to_quat(R_cfw)

        frame_idx = int(dpvo_tstamps[i])
        source = "hloc" if frame_idx in anchor_frame_set else "dpvo"
        world_poses.append({
            "frame_idx": frame_idx,
            "qw": float(q_cfw[0]),
            "qx": float(q_cfw[1]),
            "qy": float(q_cfw[2]),
            "qz": float(q_cfw[3]),
            "tx": float(t_cfw[0]),
            "ty": float(t_cfw[1]),
            "tz": float(t_cfw[2]),
            "source": source,
        })

    return world_poses, trajectory_points, scale


@app.function(
    image=dpvo_image,
    gpu="A100",
    timeout=1800,
    memory=32768,
)
def run_dpvo_odometry(
    video_bytes: bytes,
    reference_tar: bytes,
    calib: list[float] | None = None,
    anchor_frame_idx: int = 0,
    target_fps: int = 15,
) -> dict:
    """
    Run DPVO odometry on a video, anchored to world coordinates via HLoc.

    Args:
        video_bytes: Input video file bytes.
        reference_tar: HLoc reference model tar.gz bytes.
        calib: Optional [fx, fy, cx, cy]. If None, estimated from frame size.
        anchor_frame_idx: Which extracted frame to use as HLoc anchor.
        target_fps: Frame extraction rate for DPVO processing.

    Returns:
        {poses, anchor_pose, trajectory_points, num_frames, dpvo_fps}
    """
    import time

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, "/opt/dpvo")
    from dpvo.config import cfg as dpvo_cfg
    from dpvo.dpvo import DPVO

    # --- Extract frames from video ---
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
        frame_count += 1
    cap.release()
    os.unlink(video_path)

    if len(frames) < 5:
        raise ValueError(f"Only extracted {len(frames)} frames — need at least 5 for DPVO")

    print(f"Extracted {len(frames)} frames from {frame_count} total "
          f"(interval={frame_interval}, source_fps={source_fps:.1f})")

    # --- Determine calibration from the COLMAP reconstruction ---
    h, w = frames[0].shape[:2]
    if calib is None:
        # Extract the SfM-optimized camera intrinsics from the reference
        import pycolmap
        ref_dir_tmp = tempfile.mkdtemp()
        buf = io.BytesIO(reference_tar)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            # Only extract the sfm dir for camera params
            members = [m for m in tar.getmembers() if m.name.startswith("sfm/")]
            tar.extractall(ref_dir_tmp, members=members)
        try:
            rec = pycolmap.Reconstruction(os.path.join(ref_dir_tmp, "sfm"))
            rec_cam = list(rec.cameras.values())[0]
            # Extract focal length and principal point
            # SIMPLE_RADIAL: [f, cx, cy, k1], SIMPLE_PINHOLE: [f, cx, cy]
            params = list(rec_cam.params)
            model_str = str(rec_cam.model)
            if "SIMPLE" in model_str:
                focal = params[0]
                cx, cy = params[1], params[2]
                calib = [focal, focal, cx, cy]
            else:
                calib = [params[0], params[1], params[2], params[3]]
            print(f"Using SfM camera: model={model_str}, focal={calib[0]:.1f}")
        except Exception as e:
            print(f"Warning: could not read SfM camera ({e}), falling back to estimate")
            focal = max(h, w) * 1.2
            calib = [focal, focal, w / 2.0, h / 2.0]
    print(f"Calibration: fx={calib[0]:.1f} fy={calib[1]:.1f} cx={calib[2]:.1f} cy={calib[3]:.1f}")

    # --- Localize many frames with HLoc for similarity alignment ---
    # DPVO is monocular => scale is arbitrary and drifts over time.
    # Localizing many frames gives Umeyama a robust fit.
    localize_fn = modal.Function.from_name("hloc-localization", "localize_frame")

    # 3 spread-out HLoc anchors (first, middle, last) for Umeyama similarity alignment.
    # Wide baseline gives robust scale recovery. Adjacent frames (e.g. [0,1,2]) have
    # tiny DPVO displacement making scale noisy. Minimum 2 anchors needed for scale;
    # 1 anchor gives position+orientation but no scale (bad for monocular DPVO).
    n = len(frames)
    anchor_indices = sorted(set([0, n // 2, n - 1]))

    print(f"Localizing {len(anchor_indices)} frames with HLoc: {anchor_indices}")

    anchor_jpgs = []
    for idx in anchor_indices:
        _, jpg = cv2.imencode(".jpg", frames[idx], [cv2.IMWRITE_JPEG_QUALITY, 95])
        anchor_jpgs.append(jpg.tobytes())

    hloc_anchors = []
    args_list = [(jpg, reference_tar) for jpg in anchor_jpgs]
    results = list(localize_fn.starmap(args_list))
    for idx, pose in zip(anchor_indices, results):
        if pose.get("success"):
            hloc_anchors.append((idx, pose))
            print(f"  Frame {idx}: t=({pose['tx']:.3f}, {pose['ty']:.3f}, {pose['tz']:.3f}), "
                  f"inliers={pose['num_inliers']}")
        else:
            print(f"  Frame {idx}: FAILED — {pose.get('error', 'unknown')}")

    if len(hloc_anchors) < 2:
        return {
            "success": False,
            "error": f"Need 2+ HLoc-localized frames for scale recovery, got {len(hloc_anchors)}",
            "hloc_anchors": [{"frame_idx": idx, **p} for idx, p in hloc_anchors],
        }

    anchor_pose = hloc_anchors[0][1]
    print(f"Successfully localized {len(hloc_anchors)}/{len(anchor_indices)} frames")

    # --- Free memory before DPVO ---
    # reference_tar is ~165MB, video_bytes is ~5MB — free them
    del reference_tar, video_bytes
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.0f} MB")
    dpvo_h, dpvo_w = 480, 640
    intrinsics = np.array(calib, dtype=np.float32)
    # Scale intrinsics to DPVO resolution
    sx = dpvo_w / w
    sy = dpvo_h / h
    intrinsics[0] *= sx  # fx
    intrinsics[1] *= sy  # fy
    intrinsics[2] *= sx  # cx
    intrinsics[3] *= sy  # cy

    intrinsics_tensor = torch.from_numpy(intrinsics).cuda()

    # --- Run DPVO ---
    print("Initializing DPVO...")
    dpvo_cfg.merge_from_file("/opt/dpvo/config/default.yaml")
    dpvo_cfg.BUFFER_SIZE = 256
    dpvo_cfg.PATCHES_PER_FRAME = 48  # default is 96, reduce for memory

    # Find checkpoint (models.zip extracts to models/ or checkpoints/)
    ckpt_path = None
    for p in ["/opt/dpvo/models/dpvo.pth", "/opt/dpvo/checkpoints/dpvo.pth", "/opt/dpvo/dpvo.pth"]:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        # List what's actually there for debugging
        for root, dirs, files in os.walk("/opt/dpvo"):
            for f in files:
                if f.endswith(".pth"):
                    ckpt_path = os.path.join(root, f)
                    break
            if ckpt_path:
                break
    if ckpt_path is None:
        raise FileNotFoundError("DPVO checkpoint not found under /opt/dpvo")
    print(f"Using DPVO checkpoint: {ckpt_path}")

    slam = DPVO(dpvo_cfg, network=ckpt_path,
                ht=dpvo_h, wd=dpvo_w, viz=False)

    print(f"Processing {len(frames)} frames through DPVO...")
    t0 = time.time()

    for i, frame in enumerate(frames):
        # Resize and convert to tensor
        resized = cv2.resize(frame, (dpvo_w, dpvo_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda()

        slam(i, img_tensor, intrinsics_tensor)

        if (i + 1) % 50 == 0:
            print(f"  Frame {i + 1}/{len(frames)}")

    # Finalize
    poses, tstamps = slam.terminate()
    dpvo_time = time.time() - t0

    if hasattr(poses, 'cpu'):
        poses = poses.cpu().numpy()
    if hasattr(tstamps, 'cpu'):
        tstamps = tstamps.cpu().numpy()

    print(f"DPVO done: {len(poses)} poses in {dpvo_time:.1f}s "
          f"({len(frames) / dpvo_time:.1f} fps)")

    # --- Align to world coordinates (similarity transform with scale) ---
    print("Aligning DPVO trajectory to world coordinates...")
    world_poses, trajectory_points, scale = _align_dpvo_to_world(
        poses, tstamps, hloc_anchors,
    )

    # Ensure all values are plain Python types (no numpy) for serialization
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj

    return _clean({
        "success": True,
        "poses": world_poses,
        "anchor_pose": anchor_pose,
        "hloc_anchors": [{"frame_idx": idx, **p} for idx, p in hloc_anchors],
        "trajectory_points": trajectory_points,
        "alignment_scale": scale,
        "num_frames": len(frames),
        "num_dpvo_poses": len(poses),
        "dpvo_fps": len(frames) / dpvo_time,
        "dpvo_time_s": dpvo_time,
    })


@app.local_entrypoint()
def main(
    video_path: str,
    reference_path: str,
    calib_path: str = "",
    anchor_frame: int = 0,
    fps: int = 15,
):
    """
    Run DPVO odometry on a video with HLoc anchoring.

    Usage:
      modal run hloc_localization/backend/dpvo_app.py \
        --video-path data/IMG_4730.MOV \
        --reference-path hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz
    """
    import json

    video_p = pathlib.Path(video_path).expanduser().resolve()
    ref_p = pathlib.Path(reference_path).expanduser().resolve()

    print(f"Video: {video_p.name} ({video_p.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Reference: {ref_p.name} ({ref_p.stat().st_size / 1024 / 1024:.1f} MB)")

    calib = None
    if calib_path:
        calib_p = pathlib.Path(calib_path).expanduser().resolve()
        parts = calib_p.read_text().strip().split()
        calib = [float(x) for x in parts[:4]]
        print(f"Calibration: {calib}")

    result = run_dpvo_odometry.remote(
        video_p.read_bytes(),
        ref_p.read_bytes(),
        calib=calib,
        anchor_frame_idx=anchor_frame,
        target_fps=fps,
    )

    if not result.get("success"):
        print(f"\nFailed: {result.get('error', 'unknown')}")
        return

    # Save result
    out_dir = pathlib.Path(__file__).parent.parent / "data" / "dpvo_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_p.stem}_dpvo.json"

    # Remove non-serializable fields
    output = {k: v for k, v in result.items() if k != "anchor_pose" or isinstance(v, dict)}
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\nResults saved to {out_path}")
    print(f"Poses: {result['num_dpvo_poses']}")
    print(f"DPVO FPS: {result['dpvo_fps']:.1f}")
    print(f"Anchor: t=({result['anchor_pose']['tx']:.3f}, "
          f"{result['anchor_pose']['ty']:.3f}, {result['anchor_pose']['tz']:.3f})")
