"""
3D Object Location Viewer — Shows detected objects positioned in 3D space.

Loads a GLB point cloud + the objects3d.json from locate_app.py,
aligns COLMAP→GLB coordinate systems via ICP, and renders object
positions as labeled colored spheres in a viser 3D scene.

Includes camera path playback with smooth interpolation and optional
video panels (source, segmentation mask, depth).

Usage:
  python groundedsam2/locate_viewer.py \
    data/mapanything/IMG_4720.glb \
    data/groundedsam2/IMG_4730_objects3d.json \
    --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz \
    --video data/IMG_4730.MOV \
    --results-dir data/groundedsam2/
"""

import argparse
import io
import json
import struct
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rotation(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def procrustes_align(src, dst):
    """Similarity transform (scale, R, t) mapping src → dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    ss = np.sqrt((sc**2).sum() / len(src))
    sd = np.sqrt((dc**2).sum() / len(dst))
    scale = sd / ss
    H = sc.T @ dc / len(src)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def apply_similarity(pts, scale, R, t):
    return (scale * (R @ pts.T)).T + t


def slerp_wxyz(q0, q1, t):
    """Spherical linear interpolation between two wxyz quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_glb_points(glb_path, max_points=500_000):
    scene = trimesh.load(str(glb_path))
    points, colors = [], []
    geoms = scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]
    for geom in geoms:
        if hasattr(geom, "vertices"):
            pts = np.array(geom.vertices)
            if hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
                cols = np.array(geom.visual.vertex_colors)[:, :3] / 255.0
            else:
                cols = np.ones((len(pts), 3)) * 0.7
            points.append(pts)
            colors.append(cols)
    all_pts = np.concatenate(points) if points else np.zeros((0, 3))
    all_cols = np.concatenate(colors) if colors else np.zeros((0, 3))
    if len(all_pts) > max_points:
        idx = np.random.choice(len(all_pts), max_points, replace=False)
        all_pts, all_cols = all_pts[idx], all_cols[idx]
    return all_pts, all_cols


def extract_colmap_points(tar_path):
    """Extract 3D points and camera centers from COLMAP reconstruction."""
    with tarfile.open(tar_path, "r:gz") as tf:
        points3d_bin = images_bin = None
        for member in tf.getmembers():
            if member.name.endswith("points3D.bin"):
                points3d_bin = tf.extractfile(member).read()
            elif member.name.endswith("images.bin"):
                images_bin = tf.extractfile(member).read()

    if points3d_bin is None:
        raise RuntimeError("points3D.bin not found in reference tar")

    pts = []
    buf = io.BytesIO(points3d_bin)
    num_points = struct.unpack("<Q", buf.read(8))[0]
    for _ in range(num_points):
        struct.unpack("<Q", buf.read(8))
        xyz = struct.unpack("<ddd", buf.read(24))
        rgb = struct.unpack("<BBB", buf.read(3))
        struct.unpack("<d", buf.read(8))
        track_length = struct.unpack("<Q", buf.read(8))[0]
        buf.read(track_length * 8)
        pts.append((*xyz, *rgb))

    pts = np.array(pts)
    points_xyz = pts[:, :3]
    points_rgb = pts[:, 3:6] / 255.0

    cam_centers = []
    if images_bin:
        buf2 = io.BytesIO(images_bin)
        num_images = struct.unpack("<Q", buf2.read(8))[0]
        for _ in range(num_images):
            struct.unpack("<I", buf2.read(4))
            qw, qx, qy, qz = struct.unpack("<dddd", buf2.read(32))
            tx, ty, tz = struct.unpack("<ddd", buf2.read(24))
            struct.unpack("<I", buf2.read(4))
            name = b""
            while True:
                c = buf2.read(1)
                if c == b"\x00":
                    break
                name += c
            num_p2d = struct.unpack("<Q", buf2.read(8))[0]
            buf2.read(num_p2d * 24)
            R = quat_to_rotation(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            cam_centers.append(-R.T @ t)

    return points_xyz, points_rgb, np.array(cam_centers) if cam_centers else np.zeros((0, 3))


def compute_alignment(colmap_pts, glb_pts, n_iters=20):
    """ICP alignment from COLMAP → GLB coordinate space."""
    rng = np.random.RandomState(42)  # deterministic for reproducible alignment
    n_align = min(10_000, len(colmap_pts))
    colmap_sub = colmap_pts[rng.choice(len(colmap_pts), min(n_align, len(colmap_pts)), replace=False)]
    glb_sub = glb_pts[rng.choice(len(glb_pts), min(50_000, len(glb_pts)), replace=False)]

    scale = np.std(glb_sub) / np.std(colmap_sub)
    R = np.eye(3)
    t = glb_sub.mean(0) - scale * colmap_sub.mean(0)

    for i in range(n_iters):
        transformed = apply_similarity(colmap_sub, scale, R, t)
        tree = KDTree(glb_sub)
        dists, indices = tree.query(transformed)
        threshold = np.percentile(dists, 70)
        mask = dists < threshold
        if mask.sum() < 10:
            break
        scale, R, t = procrustes_align(colmap_sub[mask], glb_sub[indices[mask]])
        if i % 5 == 0:
            rmse = np.sqrt(np.mean(dists[mask]**2))
            print(f"  ICP iter {i}: RMSE={rmse:.4f}, scale={scale:.4f}")

    rmse = np.sqrt(np.mean(dists[mask]**2))
    print(f"  Final: scale={scale:.4f}, RMSE={rmse:.4f}")
    return scale, R, t


def extract_video_frames_at(video_path, frame_indices, max_w=480):
    """Extract specific frame indices from a video, return as RGB numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    for fi in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            s = min(max_w / w, max_w / h)
            if s < 1.0:
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frames


def load_binary_masks(video_path, frame_indices, threshold=10):
    """Load full-resolution frames from masked_depth video and threshold to binary masks.

    Returns dict of frame_idx -> binary mask (H x W bool array).
    Non-black pixels (any channel > threshold) = object present.
    """
    cap = cv2.VideoCapture(str(video_path))
    masks = {}
    for fi in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            # Any channel above threshold = object pixel
            masks[fi] = np.any(frame > threshold, axis=2)
    cap.release()
    return masks


# ---------------------------------------------------------------------------
# Stable object colors
# ---------------------------------------------------------------------------

COLORS = [
    (255, 80, 80),    # red
    (80, 200, 80),    # green
    (80, 120, 255),   # blue
    (255, 200, 50),   # yellow
    (200, 80, 255),   # purple
    (50, 220, 220),   # cyan
    (255, 140, 50),   # orange
    (255, 100, 200),  # pink
]


def main():
    parser = argparse.ArgumentParser(description="3D Object Location Viewer")
    parser.add_argument("glb_path", help="GLB point cloud file")
    parser.add_argument("objects3d_json", help="objects3d.json from locate_app.py")
    parser.add_argument("--reference", required=True, help="HLoc reference.tar.gz for alignment")
    parser.add_argument("--video", default=None, help="Source video (.MOV) to show frames from")
    parser.add_argument("--results-dir", default=None, help="Dir with tracked/depth videos (data/groundedsam2/)")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--interp-steps", type=int, default=20,
                        help="Interpolation steps between camera keyframes for smooth path")
    args = parser.parse_args()

    # Load data
    print(f"Loading GLB: {args.glb_path}")
    glb_pts, glb_cols = load_glb_points(args.glb_path, args.max_points)
    print(f"  {len(glb_pts):,} points")

    print(f"Loading reference: {args.reference}")
    colmap_pts, colmap_rgb, colmap_cams = extract_colmap_points(args.reference)
    print(f"  {len(colmap_pts):,} COLMAP points, {len(colmap_cams)} cameras")

    print("Computing COLMAP→GLB alignment...")
    align_scale, R_align, t_align = compute_alignment(colmap_pts, glb_pts)

    print(f"Loading objects: {args.objects3d_json}")
    data = json.loads(Path(args.objects3d_json).read_text())
    object_positions = data["object_positions"]
    object_summary = data["object_summary"]
    camera_poses = data["camera_poses"]
    objects_detected = data["objects_detected"]
    focal_length = data.get("focal_length", 2304.0)
    frame_w = data.get("frame_w", 1920)
    frame_h = data.get("frame_h", 1080)
    fx = fy = focal_length
    px, py = frame_w / 2.0, frame_h / 2.0

    print(f"  {len(object_positions)} observations, {len(object_summary)} unique objects")
    print(f"  {len(camera_poses)} camera poses")

    # ---- Load video frames ----
    sorted_frames = sorted(camera_poses.keys(), key=lambda x: int(x))
    sorted_frame_ints = [int(f) for f in sorted_frames]

    # Detect stem from json filename for auto-finding videos
    json_stem = Path(args.objects3d_json).stem.replace("_objects3d", "")
    results_dir = Path(args.results_dir) if args.results_dir else Path(args.objects3d_json).parent

    source_frames = {}  # cam_idx -> RGB ndarray
    tracked_frames = {}
    depth_frames = {}

    if args.video and Path(args.video).exists():
        print(f"Loading source video: {args.video}")
        raw = extract_video_frames_at(args.video, sorted_frame_ints)
        source_frames = {i: raw[fi] for i, fi in enumerate(sorted_frame_ints) if fi in raw}
        print(f"  {len(source_frames)} source frames")

    # Auto-find tracked video
    tracked_path = results_dir / f"{json_stem}_tracked.mp4"
    if tracked_path.exists():
        print(f"Loading tracked video: {tracked_path}")
        raw = extract_video_frames_at(tracked_path, sorted_frame_ints)
        tracked_frames = {i: raw[fi] for i, fi in enumerate(sorted_frame_ints) if fi in raw}
        print(f"  {len(tracked_frames)} tracked frames")

    # Auto-find composite depth video
    composite_path = results_dir / f"{json_stem}_composite.mp4"
    if composite_path.exists():
        print(f"Loading depth composite: {composite_path}")
        raw = extract_video_frames_at(composite_path, sorted_frame_ints)
        depth_frames = {i: raw[fi] for i, fi in enumerate(sorted_frame_ints) if fi in raw}
        print(f"  {len(depth_frames)} depth frames")

    # Also try masked depth
    masked_depth_path = results_dir / f"{json_stem}_masked_depth.mp4"
    masked_depth_frames = {}
    if masked_depth_path.exists():
        print(f"Loading masked depth: {masked_depth_path}")
        raw = extract_video_frames_at(masked_depth_path, sorted_frame_ints)
        masked_depth_frames = {i: raw[fi] for i, fi in enumerate(sorted_frame_ints) if fi in raw}
        print(f"  {len(masked_depth_frames)} masked depth frames")

    has_video_panels = bool(source_frames or tracked_frames or depth_frames or masked_depth_frames)

    # ---- Transform camera poses from COLMAP → GLB space ----
    cam_positions_aligned = []
    cam_transforms_aligned = []
    cam_wxyz_list = []
    frame_to_cam_idx = {}
    for fi_str in sorted_frames:
        p = camera_poses[fi_str]
        R_cam = quat_to_rotation(p["qw"], p["qx"], p["qy"], p["qz"])
        t_cam = np.array([p["tx"], p["ty"], p["tz"]])
        cam_pos = -R_cam.T @ t_cam
        cam_pos_aligned = apply_similarity(cam_pos.reshape(1, 3), align_scale, R_align, t_align)[0]
        R_aligned = R_align @ R_cam.T
        T = np.eye(4)
        T[:3, :3] = R_aligned
        T[:3, 3] = cam_pos_aligned
        cam_positions_aligned.append(cam_pos_aligned)
        cam_transforms_aligned.append(T)
        cam_wxyz_list.append(vtf.SO3.from_matrix(R_aligned).wxyz)
        frame_to_cam_idx[int(fi_str)] = len(cam_transforms_aligned) - 1

    cam_positions_aligned = np.array(cam_positions_aligned) if cam_positions_aligned else np.zeros((0, 3))
    num_cams = len(cam_transforms_aligned)

    # ---- Build smooth interpolated camera path ----
    interp_steps = args.interp_steps
    smooth_positions = []
    smooth_wxyz = []
    smooth_keyframe_indices = []  # which smooth idx corresponds to keyframes

    if num_cams >= 2:
        for i in range(num_cams - 1):
            for s in range(interp_steps):
                t_frac = s / interp_steps
                pos = (1 - t_frac) * cam_positions_aligned[i] + t_frac * cam_positions_aligned[i + 1]
                wxyz = slerp_wxyz(np.array(cam_wxyz_list[i]), np.array(cam_wxyz_list[i + 1]), t_frac)
                smooth_positions.append(pos)
                smooth_wxyz.append(wxyz)
                if s == 0:
                    smooth_keyframe_indices.append(len(smooth_positions) - 1)
        # Add final keyframe
        smooth_positions.append(cam_positions_aligned[-1])
        smooth_wxyz.append(np.array(cam_wxyz_list[-1]))
        smooth_keyframe_indices.append(len(smooth_positions) - 1)
    elif num_cams == 1:
        smooth_positions.append(cam_positions_aligned[0])
        smooth_wxyz.append(np.array(cam_wxyz_list[0]))
        smooth_keyframe_indices.append(0)

    smooth_positions = np.array(smooth_positions) if smooth_positions else np.zeros((0, 3))
    num_smooth = len(smooth_positions)

    # ---- Ray-cast object positions into GLB cloud ----
    glb_tree = KDTree(glb_pts)

    print("Computing object positions via ray-casting into GLB cloud...")
    for obs in object_positions:
        fi = obs["frame_idx"]
        cam_idx = frame_to_cam_idx.get(fi)
        if cam_idx is None:
            pos = np.array(obs["position_3d"])
            obs["position_3d_aligned"] = apply_similarity(pos.reshape(1, 3), align_scale, R_align, t_align)[0].tolist()
            continue

        T = cam_transforms_aligned[cam_idx]
        cam_pos = T[:3, 3]
        R_cam_aligned = T[:3, :3]

        cx_obj, cy_obj = obs["pixel_centroid"]
        dir_cam = np.array([(cx_obj - px) / fx, (cy_obj - py) / fy, 1.0])
        dir_cam /= np.linalg.norm(dir_cam)
        dir_glb = R_cam_aligned @ dir_cam

        ray_max = np.linalg.norm(glb_pts.max(0) - glb_pts.min(0)) * 1.5
        t_vals = np.linspace(0.1, ray_max, 200)
        ray_pts = cam_pos + np.outer(t_vals, dir_glb)
        dists, indices = glb_tree.query(ray_pts)
        best = np.argmin(dists)
        obs["position_3d_aligned"] = glb_pts[indices[best]].tolist()

    # Aggregate per object
    obj_positions_agg = defaultdict(list)
    for obs in object_positions:
        if "position_3d_aligned" in obs:
            obj_positions_agg[obs["obj_id"]].append(obs["position_3d_aligned"])

    for obj in object_summary:
        oid = obj["obj_id"]
        if oid in obj_positions_agg and obj_positions_agg[oid]:
            positions = np.array(obj_positions_agg[oid])
            obj["mean_position_3d_aligned"] = positions.mean(axis=0).tolist()
            print(f"  {obj['label']}: ray-cast pos = {obj['mean_position_3d_aligned']}")
        else:
            pos = np.array(obj["mean_position_3d"])
            obj["mean_position_3d_aligned"] = apply_similarity(pos.reshape(1, 3), align_scale, R_align, t_align)[0].tolist()

    # ---- Projection diagnostic ----
    print("\nProjection verification (reprojecting 3D → pixel):")
    for obs in object_positions:
        fi = obs["frame_idx"]
        cam_idx = frame_to_cam_idx.get(fi)
        if cam_idx is None:
            continue
        T = cam_transforms_aligned[cam_idx]
        cam_pos_dbg = T[:3, 3]
        R_w2c_dbg = T[:3, :3].T

        # Reproject the aligned position back to pixel
        pos_aligned = np.array(obs.get("position_3d_aligned", [0, 0, 0]))
        p_cam_dbg = R_w2c_dbg @ (pos_aligned - cam_pos_dbg)
        u_dbg = fx * p_cam_dbg[0] / p_cam_dbg[2] + px
        v_dbg = fy * p_cam_dbg[1] / p_cam_dbg[2] + py
        expected_cx, expected_cy = obs["pixel_centroid"]
        print(f"  Frame {fi} ({obs['label']}): "
              f"expected=({expected_cx:.0f},{expected_cy:.0f}), "
              f"reprojected=({u_dbg:.0f},{v_dbg:.0f}), "
              f"z_cam={p_cam_dbg[2]:.2f}")

        # Draw crosshair on source frame for visual verification
        if cam_idx in source_frames:
            frame_rgb = source_frames[cam_idx].copy()
            fh, fw = frame_rgb.shape[:2]
            # Scale projected pixel to thumbnail size
            sx, sy = fw / frame_w, fh / frame_h
            ux, vy = int(u_dbg * sx), int(v_dbg * sy)
            ex, ey = int(expected_cx * sx), int(expected_cy * sy)
            # Green crosshair = reprojected, Red = expected centroid
            r = 8
            if 0 <= ux < fw and 0 <= vy < fh:
                frame_rgb[max(0,vy-r):vy+r, max(0,ux-1):ux+2] = [0, 255, 0]
                frame_rgb[max(0,vy-1):vy+2, max(0,ux-r):ux+r] = [0, 255, 0]
            if 0 <= ex < fw and 0 <= ey < fh:
                frame_rgb[max(0,ey-r):ey+r, max(0,ex-1):ex+2] = [255, 0, 0]
                frame_rgb[max(0,ey-1):ey+2, max(0,ex-r):ex+r] = [255, 0, 0]
            source_frames[cam_idx] = frame_rgb

    # ---- Color map for objects ----
    obj_color_map = {}
    for i, obj in enumerate(object_summary):
        obj_color_map[obj["obj_id"]] = COLORS[i % len(COLORS)]

    # ---- Per-frame object data ----
    frame_to_obs = {}
    for obs in object_positions:
        fi = obs["frame_idx"]
        if fi not in frame_to_obs:
            frame_to_obs[fi] = []
        frame_to_obs[fi].append(obs)

    obs_by_obj = {}
    for obs in object_positions:
        oid = obs["obj_id"]
        if oid not in obs_by_obj:
            obs_by_obj[oid] = []
        obs_by_obj[oid].append(obs["position_3d_aligned"])

    # ==================================================================
    # Viser scene
    # ==================================================================
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\nViser server at http://localhost:{args.port}")

    # GLB point cloud
    server.scene.add_point_cloud(
        "/scene/glb",
        points=glb_pts.astype(np.float32),
        colors=glb_cols.astype(np.float32),
        point_size=0.004,
        point_shape="rounded",
    )

    # Camera frustums
    for i, T in enumerate(cam_transforms_aligned):
        server.scene.add_camera_frustum(
            f"/scene/cameras/cam_{i:03d}",
            fov=60.0, aspect=16/9, scale=0.08,
            wxyz=cam_wxyz_list[i], position=T[:3, 3],
            color=(80, 80, 180),
        )

    # Smooth camera trajectory line
    if len(smooth_positions) > 1:
        server.scene.add_point_cloud(
            "/scene/trajectory",
            points=smooth_positions.astype(np.float32),
            colors=np.array([[1.0, 0.5, 0.0]] * len(smooth_positions), dtype=np.float32),
            point_size=0.008,
            point_shape="circle",
        )

    # Object markers
    for obj in object_summary:
        color = obj_color_map[obj["obj_id"]]
        oid = obj["obj_id"]
        label = obj["label"]
        pos = np.array(obj["mean_position_3d_aligned"], dtype=np.float32)
        server.scene.add_icosphere(
            f"/scene/objects/{label}_{oid}/sphere",
            radius=0.06, color=color, position=pos,
        )
        server.scene.add_label(
            f"/scene/objects/{label}_{oid}/label",
            text=f"{label} (d={obj['mean_depth']:.1f})",
            position=pos + np.array([0, -0.12, 0], dtype=np.float32),
        )

    # Per-object observation trails
    for oid, positions in obs_by_obj.items():
        pts = np.array(positions, dtype=np.float32)
        color = obj_color_map.get(oid, (200, 200, 200))
        color_f = np.array([[c / 255.0 for c in color]] * len(pts), dtype=np.float32)
        label = next((o["label"] for o in object_summary if o["obj_id"] == oid), f"obj_{oid}")
        server.scene.add_point_cloud(
            f"/scene/objects/{label}_{oid}/trail",
            points=pts, colors=color_f,
            point_size=0.015, point_shape="circle",
        )

    # ==================================================================
    # Camera path navigation (smooth)
    # ==================================================================
    playback_active = [False]
    current_smooth_idx = [0]

    # Image handles dict (set later)
    img_handles = {}

    def keyframe_for_smooth(smooth_idx):
        """Return the nearest keyframe index for a smooth position index."""
        best = 0
        for ki, si in enumerate(smooth_keyframe_indices):
            if si <= smooth_idx:
                best = ki
        return best

    def set_smooth_frame(idx):
        """Move camera to smooth interpolated position idx."""
        idx = max(0, min(idx, num_smooth - 1))
        current_smooth_idx[0] = idx
        pos = smooth_positions[idx]
        wxyz = smooth_wxyz[idx]
        R_cam = vtf.SO3(wxyz).as_matrix()
        look_target = pos + R_cam @ np.array([0.0, 0.0, 0.5])
        up_dir = R_cam @ np.array([0.0, -1.0, 0.0])

        # Highlight nearest keyframe frustum
        nearest_kf = keyframe_for_smooth(idx)
        for j in range(num_cams):
            color = (255, 200, 50) if j == nearest_kf else (80, 80, 180)
            server.scene.add_camera_frustum(
                f"/scene/cameras/cam_{j:03d}",
                fov=60.0, aspect=16/9,
                scale=0.12 if j == nearest_kf else 0.08,
                wxyz=cam_wxyz_list[j], position=cam_transforms_aligned[j][:3, 3],
                color=color,
            )

        # Frame info
        fi_int = sorted_frame_ints[nearest_kf] if nearest_kf < len(sorted_frame_ints) else -1
        visible_obs = frame_to_obs.get(fi_int, [])
        visible_labels = [o["label"] for o in visible_obs]
        info_text = f"**Frame {fi_int}** (pose {nearest_kf+1}/{num_cams})"
        if visible_labels:
            info_text += f"\n\nVisible: {', '.join(visible_labels)}"
        frame_info_handle.content = info_text

        # Update video panels
        for name, frames_dict in [("source", source_frames), ("tracked", tracked_frames),
                                   ("depth", depth_frames), ("masked_depth", masked_depth_frames)]:
            if name in img_handles and nearest_kf in frames_dict:
                img_handles[name].image = frames_dict[nearest_kf]

        # Move client cameras
        for client in server.get_clients().values():
            with client.atomic():
                client.camera.position = tuple(pos)
                client.camera.look_at = tuple(look_target)
                client.camera.up_direction = tuple(up_dir)

    # ==================================================================
    # GUI
    # ==================================================================
    with server.gui.add_folder("Camera Path", expand_by_default=True):
        path_slider = server.gui.add_slider(
            "Path", min=0, max=max(num_smooth - 1, 0), step=1, initial_value=0,
        )
        frame_info_handle = server.gui.add_markdown(
            f"**Frame {sorted_frame_ints[0]}** (pose 1/{num_cams})" if sorted_frame_ints else "No poses"
        )
        play_btn = server.gui.add_button("Play Path")
        speed_slider = server.gui.add_slider(
            "Speed (ms/step)", min=10, max=200, step=10, initial_value=50,
        )

        @path_slider.on_update
        def _on_slider(event):
            if event.client is not None:
                set_smooth_frame(path_slider.value)

        @play_btn.on_click
        def _on_play(event):
            playback_active[0] = not playback_active[0]

    # Video panels
    if has_video_panels:
        with server.gui.add_folder("Video Panels", expand_by_default=True):
            placeholder = np.zeros((120, 160, 3), dtype=np.uint8)
            if source_frames:
                first = source_frames.get(0, placeholder)
                img_handles["source"] = server.gui.add_image(first, format="jpeg", label="Source")
            if tracked_frames:
                first = tracked_frames.get(0, placeholder)
                img_handles["tracked"] = server.gui.add_image(first, format="jpeg", label="Segmentation")
            if depth_frames:
                first = depth_frames.get(0, placeholder)
                img_handles["depth"] = server.gui.add_image(first, format="jpeg", label="Depth Composite")
            if masked_depth_frames:
                first = masked_depth_frames.get(0, placeholder)
                img_handles["masked_depth"] = server.gui.add_image(first, format="jpeg", label="Object Depth")

    with server.gui.add_folder("Objects"):
        for obj in object_summary:
            pos = obj["mean_position_3d_aligned"]
            server.gui.add_markdown(
                f"**{obj['label']}** (id={obj['obj_id']})\n\n"
                f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n\n"
                f"Depth: {obj['mean_depth']:.1f} | Obs: {obj['num_observations']}"
            )

        obj_toggles = {}
        for obj in object_summary:
            oid = obj["obj_id"]
            label = obj["label"]
            cb = server.gui.add_checkbox(f"Show {label} ({oid})", initial_value=True)
            obj_toggles[oid] = (cb, label)

            def _make_toggle(oid_cap, label_cap):
                def _toggle(event):
                    if obj_toggles[oid_cap][0].value:
                        color = obj_color_map[oid_cap]
                        pos = np.array(
                            next(o["mean_position_3d_aligned"] for o in object_summary if o["obj_id"] == oid_cap),
                            dtype=np.float32,
                        )
                        server.scene.add_icosphere(
                            f"/scene/objects/{label_cap}_{oid_cap}/sphere",
                            radius=0.06, color=color, position=pos,
                        )
                        mean_depth = next(o["mean_depth"] for o in object_summary if o["obj_id"] == oid_cap)
                        server.scene.add_label(
                            f"/scene/objects/{label_cap}_{oid_cap}/label",
                            text=f"{label_cap} (d={mean_depth:.1f})",
                            position=pos + np.array([0, -0.12, 0], dtype=np.float32),
                        )
                        if oid_cap in obs_by_obj:
                            pts = np.array(obs_by_obj[oid_cap], dtype=np.float32)
                            c = obj_color_map[oid_cap]
                            color_f = np.array([[cc / 255.0 for cc in c]] * len(pts), dtype=np.float32)
                            server.scene.add_point_cloud(
                                f"/scene/objects/{label_cap}_{oid_cap}/trail",
                                points=pts, colors=color_f,
                                point_size=0.015, point_shape="circle",
                            )
                    else:
                        server.scene.remove(f"/scene/objects/{label_cap}_{oid_cap}")
                return _toggle
            cb.on_update(_make_toggle(oid, label))

    with server.gui.add_folder("Info"):
        server.gui.add_markdown(
            f"**Alignment:** scale={align_scale:.4f}\n\n"
            f"**Camera poses:** {len(camera_poses)}\n\n"
            f"**Objects:** {', '.join(objects_detected)}\n\n"
            f"**Observations:** {len(object_positions)}\n\n"
            f"**Smooth path:** {num_smooth} steps ({interp_steps} per segment)"
        )

    with server.gui.add_folder("Visibility"):
        show_glb = server.gui.add_checkbox("Show GLB", initial_value=True)
        show_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)

        def _redraw_glb():
            if not show_glb.value:
                server.scene.add_point_cloud(
                    "/scene/glb", points=np.zeros((1, 3), dtype=np.float32),
                    colors=np.ones((1, 3), dtype=np.float32),
                    point_size=0.0, point_shape="rounded",
                )
                return
            server.scene.add_point_cloud(
                "/scene/glb", points=glb_pts.astype(np.float32),
                colors=glb_cols.astype(np.float32),
                point_size=0.004, point_shape="rounded",
            )

        @show_glb.on_update
        def _toggle_glb(event):
            _redraw_glb()

        @show_cameras.on_update
        def _toggle_cameras(event):
            if show_cameras.value:
                for j in range(num_cams):
                    server.scene.add_camera_frustum(
                        f"/scene/cameras/cam_{j:03d}",
                        fov=60.0, aspect=16/9, scale=0.08,
                        wxyz=cam_wxyz_list[j], position=cam_transforms_aligned[j][:3, 3],
                        color=(80, 80, 180),
                    )
            else:
                for j in range(num_cams):
                    server.scene.remove(f"/scene/cameras/cam_{j:03d}")

    print("\nReady! Objects displayed in 3D:")
    for obj in object_summary:
        pos = obj["mean_position_3d_aligned"]
        print(f"  {obj['label']}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # Main loop — smooth playback
    while True:
        if playback_active[0]:
            idx = current_smooth_idx[0] + 1
            if idx >= num_smooth:
                idx = 0
            path_slider.value = idx
            set_smooth_frame(idx)
            time.sleep(speed_slider.value / 1000.0)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
