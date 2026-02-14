"""
3D Object Location Viewer — Shows detected objects positioned in 3D space.

Loads a GLB point cloud + the objects3d.json from locate_app.py,
aligns COLMAP→GLB coordinate systems via ICP, and renders object
positions as labeled colored spheres in a viser 3D scene.

Usage:
  python groundedsam2/locate_viewer.py \
    data/mapanything/IMG_4720.glb \
    data/groundedsam2/IMG_4730_objects3d.json \
    --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz
"""

import argparse
import io
import json
import struct
import tarfile
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as vtf


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
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
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
        struct.unpack("<Q", buf.read(8))  # point3D_id
        xyz = struct.unpack("<ddd", buf.read(24))
        rgb = struct.unpack("<BBB", buf.read(3))
        struct.unpack("<d", buf.read(8))  # error
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
    from scipy.spatial import KDTree

    n_align = min(10_000, len(colmap_pts))
    colmap_sub = colmap_pts[np.random.choice(len(colmap_pts), min(n_align, len(colmap_pts)), replace=False)]
    glb_sub = glb_pts[np.random.choice(len(glb_pts), min(50_000, len(glb_pts)), replace=False)]

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
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--max-points", type=int, default=500_000)
    args = parser.parse_args()

    # Load data
    print(f"Loading GLB: {args.glb_path}")
    glb_pts, glb_cols = load_glb_points(args.glb_path, args.max_points)
    print(f"  {len(glb_pts):,} points")

    print(f"Loading reference: {args.reference}")
    colmap_pts, colmap_rgb, colmap_cams = extract_colmap_points(args.reference)
    print(f"  {len(colmap_pts):,} COLMAP points, {len(colmap_cams)} cameras")

    print("Computing COLMAP→GLB alignment...")
    scale, R_align, t_align = compute_alignment(colmap_pts, glb_pts)

    print(f"Loading objects: {args.objects3d_json}")
    data = json.loads(Path(args.objects3d_json).read_text())
    object_positions = data["object_positions"]
    object_summary = data["object_summary"]
    camera_poses = data["camera_poses"]
    objects_detected = data["objects_detected"]

    print(f"  {len(object_positions)} observations, {len(object_summary)} unique objects")
    print(f"  {len(camera_poses)} camera poses")

    # Transform camera poses from COLMAP → GLB space
    cam_positions_aligned = []
    cam_transforms_aligned = []
    sorted_frames = sorted(camera_poses.keys(), key=lambda x: int(x))
    frame_to_cam_idx = {}  # frame_idx (int) -> index into cam_transforms_aligned
    for fi_str in sorted_frames:
        p = camera_poses[fi_str]
        R_cam = quat_to_rotation(p["qw"], p["qx"], p["qy"], p["qz"])
        t_cam = np.array([p["tx"], p["ty"], p["tz"]])
        cam_pos = -R_cam.T @ t_cam
        cam_pos_aligned = apply_similarity(cam_pos.reshape(1, 3), scale, R_align, t_align)[0]
        R_aligned = R_align @ R_cam.T
        T = np.eye(4)
        T[:3, :3] = R_aligned
        T[:3, 3] = cam_pos_aligned
        cam_positions_aligned.append(cam_pos_aligned)
        cam_transforms_aligned.append(T)
        frame_to_cam_idx[int(fi_str)] = len(cam_transforms_aligned) - 1

    cam_positions_aligned = np.array(cam_positions_aligned) if cam_positions_aligned else np.zeros((0, 3))

    # Compute object positions by ray-casting into the GLB point cloud.
    # DepthAnything produces relative (not metric) depth, so the raw backprojected
    # COLMAP-space positions are at the wrong scale. Instead, we shoot a ray from
    # each camera through the object's pixel centroid and find the nearest GLB
    # points along that ray.
    from scipy.spatial import KDTree
    glb_tree = KDTree(glb_pts)
    focal_length = data.get("focal_length", 2304.0)
    frame_w = data.get("frame_w", 1920)
    frame_h = data.get("frame_h", 1080)
    fx = fy = focal_length
    px, py = frame_w / 2.0, frame_h / 2.0

    print("Computing object positions via ray-casting into GLB cloud...")
    for obs in object_positions:
        fi = obs["frame_idx"]
        cam_idx = frame_to_cam_idx.get(fi)
        if cam_idx is None:
            # Fallback: transform from COLMAP space
            pos = np.array(obs["position_3d"])
            obs["position_3d_aligned"] = apply_similarity(pos.reshape(1, 3), scale, R_align, t_align)[0].tolist()
            continue

        T = cam_transforms_aligned[cam_idx]
        cam_pos = T[:3, 3]
        R_cam_aligned = T[:3, :3]  # columns are camera axes in GLB space

        # Ray direction: pixel (cx_obj, cy_obj) -> camera-space direction -> GLB space
        cx_obj, cy_obj = obs["pixel_centroid"]
        dir_cam = np.array([
            (cx_obj - px) / fx,
            (cy_obj - py) / fy,
            1.0,
        ])
        dir_cam /= np.linalg.norm(dir_cam)
        dir_glb = R_cam_aligned @ dir_cam

        # Sample points along the ray and find closest GLB points
        # Use a cone search: find GLB points near the ray
        ray_max = np.linalg.norm(glb_pts.max(0) - glb_pts.min(0)) * 1.5
        n_samples = 200
        t_vals = np.linspace(0.1, ray_max, n_samples)
        ray_pts = cam_pos + np.outer(t_vals, dir_glb)

        # For each ray sample, find distance to nearest GLB point
        dists, indices = glb_tree.query(ray_pts)
        # Find the ray sample with minimum distance to a GLB point
        best_sample = np.argmin(dists)
        obs["position_3d_aligned"] = glb_pts[indices[best_sample]].tolist()

    # Aggregate: mean of ray-cast positions per object
    from collections import defaultdict
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
            obj["mean_position_3d_aligned"] = apply_similarity(pos.reshape(1, 3), scale, R_align, t_align)[0].tolist()

    # ---- Viser scene ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\nViser server at http://localhost:{args.port}")

    # -- Build color map for objects --
    obj_color_map = {}
    for i, obj in enumerate(object_summary):
        obj_color_map[obj["obj_id"]] = COLORS[i % len(COLORS)]

    # -- Color GLB points near objects using ray-cone intersection --
    # For each observation, highlight GLB points near the ray from camera to object.
    # This paints the actual surface region the object occupies.
    glb_cols_highlighted = glb_cols.copy()
    obj_point_indices = {}  # obj_id -> set of GLB point indices

    for obs in object_positions:
        if "position_3d_aligned" not in obs:
            continue
        fi = obs["frame_idx"]
        cam_idx = frame_to_cam_idx.get(fi)
        if cam_idx is None:
            continue

        oid = obs["obj_id"]
        T = cam_transforms_aligned[cam_idx]
        cam_pos = T[:3, 3]
        obj_pos = np.array(obs["position_3d_aligned"])
        ray_dir = obj_pos - cam_pos
        ray_len = np.linalg.norm(ray_dir)
        if ray_len < 1e-6:
            continue
        ray_dir_n = ray_dir / ray_len

        # Cone half-angle based on mask area (larger mask = wider cone)
        mask_area = obs.get("mask_area", 10000)
        # Rough: mask radius in pixels -> angle
        mask_radius_px = np.sqrt(mask_area / np.pi)
        cone_half_angle = np.arctan2(mask_radius_px, focal_length) * scale  # scaled to GLB

        # Find points near the object position (within a radius proportional to cone)
        search_radius = max(0.05, min(0.3, ray_len * np.tan(cone_half_angle) * 2))
        nearby_idx = glb_tree.query_ball_point(obj_pos, search_radius)

        if nearby_idx:
            nearby_idx = np.array(nearby_idx)
            if oid not in obj_point_indices:
                obj_point_indices[oid] = set()
            obj_point_indices[oid].update(nearby_idx.tolist())

    for oid, idx_set in obj_point_indices.items():
        idx_arr = np.array(list(idx_set))
        color = obj_color_map.get(oid, (200, 200, 200))
        color_f = np.array([c / 255.0 for c in color])
        glb_cols_highlighted[idx_arr] = 0.4 * glb_cols[idx_arr] + 0.6 * color_f
        label = next((o["label"] for o in object_summary if o["obj_id"] == oid), f"obj_{oid}")
        print(f"  Highlighted {len(idx_arr)} points for {label}")

    # GLB point cloud (with highlighted regions)
    server.scene.add_point_cloud(
        "/scene/glb",
        points=glb_pts.astype(np.float32),
        colors=glb_cols_highlighted.astype(np.float32),
        point_size=0.004,
        point_shape="rounded",
    )

    # Camera frustums
    for i, T in enumerate(cam_transforms_aligned):
        wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
        server.scene.add_camera_frustum(
            f"/scene/cameras/cam_{i:03d}",
            fov=60.0, aspect=16/9, scale=0.08,
            wxyz=wxyz, position=T[:3, 3],
            color=(80, 80, 180),
        )

    # Camera trajectory
    if len(cam_positions_aligned) > 1:
        server.scene.add_point_cloud(
            "/scene/trajectory",
            points=cam_positions_aligned.astype(np.float32),
            colors=np.array([[1.0, 0.5, 0.0]] * len(cam_positions_aligned), dtype=np.float32),
            point_size=0.02,
            point_shape="circle",
        )

    # Object positions — render per unique object
    for i, obj in enumerate(object_summary):
        color = obj_color_map[obj["obj_id"]]
        obj_id = obj["obj_id"]
        label = obj["label"]
        pos = np.array(obj["mean_position_3d_aligned"], dtype=np.float32)

        # Sphere marker
        server.scene.add_icosphere(
            f"/scene/objects/{label}_{obj_id}/sphere",
            radius=0.06,
            color=color,
            position=pos,
        )

        # Label
        server.scene.add_label(
            f"/scene/objects/{label}_{obj_id}/label",
            text=f"{label} (d={obj['mean_depth']:.1f})",
            position=pos + np.array([0, -0.12, 0], dtype=np.float32),
        )

    # Per-observation trail (smaller dots showing all observations)
    for obs in object_positions:
        pos = np.array(obs["position_3d_aligned"], dtype=np.float32)
        color = obj_color_map.get(obs["obj_id"], (200, 200, 200))
        color_f = np.array([[c / 255.0 for c in color]], dtype=np.float32)

    # Group observations by object for trail visualization
    obs_by_obj = {}
    for obs in object_positions:
        oid = obs["obj_id"]
        if oid not in obs_by_obj:
            obs_by_obj[oid] = []
        obs_by_obj[oid].append(obs["position_3d_aligned"])

    for oid, positions in obs_by_obj.items():
        pts = np.array(positions, dtype=np.float32)
        color = obj_color_map.get(oid, (200, 200, 200))
        color_f = np.array([[c / 255.0 for c in color]] * len(pts), dtype=np.float32)
        label = next((o["label"] for o in object_summary if o["obj_id"] == oid), f"obj_{oid}")
        server.scene.add_point_cloud(
            f"/scene/objects/{label}_{oid}/trail",
            points=pts,
            colors=color_f,
            point_size=0.015,
            point_shape="circle",
        )

    # -- Build per-frame object visibility data --
    # Map frame_idx -> list of observations visible at that frame
    frame_to_obs = {}
    for obs in object_positions:
        fi = obs["frame_idx"]
        if fi not in frame_to_obs:
            frame_to_obs[fi] = []
        frame_to_obs[fi].append(obs)

    # ---- Camera path navigation ----
    num_cams = len(cam_transforms_aligned)
    playback_active = [False]
    current_frame_idx = [0]

    def set_camera_frame(idx):
        """Move all clients to camera pose idx and highlight active frustum."""
        idx = max(0, min(idx, num_cams - 1))
        current_frame_idx[0] = idx
        T = cam_transforms_aligned[idx]
        cam_pos = T[:3, 3]
        R_cam = T[:3, :3]
        # Camera looks along +Z in viser, -Y is up
        look_target = cam_pos + R_cam @ np.array([0.0, 0.0, 0.5])
        up_dir = R_cam @ np.array([0.0, -1.0, 0.0])

        # Highlight active frustum
        for j in range(num_cams):
            color = (255, 200, 50) if j == idx else (80, 80, 180)
            Tj = cam_transforms_aligned[j]
            wxyz_j = vtf.SO3.from_matrix(Tj[:3, :3]).wxyz
            server.scene.add_camera_frustum(
                f"/scene/cameras/cam_{j:03d}",
                fov=60.0, aspect=16/9,
                scale=0.12 if j == idx else 0.08,
                wxyz=wxyz_j, position=Tj[:3, 3],
                color=color,
            )

        # Show which objects are visible at this frame
        fi_str = sorted_frames[idx] if idx < len(sorted_frames) else None
        fi_int = int(fi_str) if fi_str else -1
        visible_obs = frame_to_obs.get(fi_int, [])
        visible_labels = [o["label"] for o in visible_obs]
        if visible_labels:
            frame_info_handle.content = (
                f"**Frame {fi_int}** (pose {idx+1}/{num_cams})\n\n"
                f"Visible: {', '.join(visible_labels)}"
            )
        else:
            frame_info_handle.content = f"**Frame {fi_int}** (pose {idx+1}/{num_cams})"

        # Move client cameras
        for client in server.get_clients().values():
            with client.atomic():
                client.camera.position = tuple(cam_pos)
                client.camera.look_at = tuple(look_target)
                client.camera.up_direction = tuple(up_dir)

    # GUI
    with server.gui.add_folder("Camera Path", expand_by_default=True):
        frame_slider = server.gui.add_slider(
            "Pose", min=0, max=max(num_cams - 1, 0), step=1, initial_value=0,
        )
        frame_info_handle = server.gui.add_markdown(f"**Frame {sorted_frames[0]}** (pose 1/{num_cams})")
        play_btn = server.gui.add_button("Play Path")
        speed_slider = server.gui.add_slider(
            "Speed (s/pose)", min=0.1, max=2.0, step=0.1, initial_value=0.5,
        )

        @frame_slider.on_update
        def _on_slider(event):
            if event.client is not None:
                set_camera_frame(frame_slider.value)

        @play_btn.on_click
        def _on_play(event):
            playback_active[0] = not playback_active[0]

    with server.gui.add_folder("Objects"):
        for obj in object_summary:
            pos = obj["mean_position_3d_aligned"]
            color = obj_color_map[obj["obj_id"]]
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            server.gui.add_markdown(
                f"<span style='color:{hex_color}'>&#9679;</span> "
                f"**{obj['label']}** (id={obj['obj_id']})\n\n"
                f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n\n"
                f"Depth: {obj['mean_depth']:.1f} | Obs: {obj['num_observations']}"
            )

        # Per-object visibility toggles
        obj_toggles = {}
        for obj in object_summary:
            oid = obj["obj_id"]
            label = obj["label"]
            cb = server.gui.add_checkbox(f"Show {label} ({oid})", initial_value=True)
            obj_toggles[oid] = (cb, label)

            def _make_toggle(oid_cap, label_cap):
                def _toggle(event):
                    cb_val = obj_toggles[oid_cap][0].value
                    if cb_val:
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
                            color_f = np.array([[c / 255.0 for c in color]] * len(pts), dtype=np.float32)
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
            f"**Alignment:** scale={scale:.4f}\n\n"
            f"**Camera poses:** {len(camera_poses)}\n\n"
            f"**Objects:** {', '.join(objects_detected)}\n\n"
            f"**Observations:** {len(object_positions)}"
        )

    with server.gui.add_folder("Visibility"):
        show_glb = server.gui.add_checkbox("Show GLB", initial_value=True)
        show_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)

        show_highlights = server.gui.add_checkbox("Highlight objects on cloud", initial_value=True)

        def _redraw_glb():
            if not show_glb.value:
                server.scene.add_point_cloud(
                    "/scene/glb", points=np.zeros((1, 3), dtype=np.float32),
                    colors=np.ones((1, 3), dtype=np.float32),
                    point_size=0.0, point_shape="rounded",
                )
                return
            cols = glb_cols_highlighted if show_highlights.value else glb_cols
            server.scene.add_point_cloud(
                "/scene/glb", points=glb_pts.astype(np.float32),
                colors=cols.astype(np.float32),
                point_size=0.004, point_shape="rounded",
            )

        @show_glb.on_update
        def _toggle_glb(event):
            _redraw_glb()

        @show_highlights.on_update
        def _toggle_highlights(event):
            _redraw_glb()

        @show_cameras.on_update
        def _toggle_cameras(event):
            if show_cameras.value:
                for j, Tj in enumerate(cam_transforms_aligned):
                    wxyz_j = vtf.SO3.from_matrix(Tj[:3, :3]).wxyz
                    server.scene.add_camera_frustum(
                        f"/scene/cameras/cam_{j:03d}",
                        fov=60.0, aspect=16/9, scale=0.08,
                        wxyz=wxyz_j, position=Tj[:3, 3],
                        color=(80, 80, 180),
                    )
            else:
                for j in range(num_cams):
                    server.scene.remove(f"/scene/cameras/cam_{j:03d}")

    print("\nReady! Objects displayed in 3D:")
    for obj in object_summary:
        pos = obj["mean_position_3d_aligned"]
        print(f"  {obj['label']}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # Main loop — handles playback
    while True:
        if playback_active[0]:
            idx = current_frame_idx[0] + 1
            if idx >= num_cams:
                idx = 0
            frame_slider.value = idx
            set_camera_frame(idx)
            time.sleep(speed_slider.value)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
