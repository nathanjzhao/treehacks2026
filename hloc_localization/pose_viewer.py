"""
Viser-based 3D viewer: shows point cloud + live camera frustum from HLoc localization.

Loads the MapAnything GLB point cloud and the COLMAP reconstruction from
the HLoc reference tar, computes ICP alignment (COLMAP→GLB), then renders
live localized poses as camera frustums in the correct GLB coordinate frame.

Usage:
    python -m hloc_localization.pose_viewer basedir/IMG_4741.glb
"""

import argparse
import io
import struct
import tarfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import trimesh
import viser
import viser.transforms as vtf


# ── Reuse proven alignment code from view_trajectory.py ──


def quat_to_rotation(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


def procrustes_align(src, dst):
    """Similarity transform src→dst via Procrustes. Returns (scale, R, t)."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    s_src = np.sqrt((src_c ** 2).sum() / len(src))
    s_dst = np.sqrt((dst_c ** 2).sum() / len(dst))
    scale = s_dst / s_src

    H = (src_c.T @ dst_c) / len(src)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def apply_similarity(points, scale, R, t):
    """Apply similarity transform to Nx3 points."""
    return (scale * (R @ points.T)).T + t


def extract_colmap_data(tar_path):
    """Extract 3D points, colors, and camera poses from COLMAP reference tar."""
    with tarfile.open(tar_path, "r:gz") as tf:
        points3d = images_bin = None
        # Prefer the full model (models/0/) over the top-level sfm/ files
        points3d_candidates = {}
        images_candidates = {}
        for member in tf.getmembers():
            if member.name.endswith("points3D.bin"):
                points3d_candidates[member.name] = member
            elif member.name.endswith("images.bin"):
                images_candidates[member.name] = member

        # Pick largest points3D.bin (the full reconstruction)
        if points3d_candidates:
            best_pts = max(points3d_candidates.values(), key=lambda m: m.size)
            points3d = tf.extractfile(best_pts).read()
            print(f"  Using {best_pts.name} ({best_pts.size:,} bytes)")
        if images_candidates:
            best_img = max(images_candidates.values(), key=lambda m: m.size)
            images_bin = tf.extractfile(best_img).read()
            print(f"  Using {best_img.name} ({best_img.size:,} bytes)")

        if points3d is None:
            raise RuntimeError("points3D.bin not found in tar")

        # Parse points3D.bin
        pts = []
        buf = io.BytesIO(points3d)
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

        # Parse images.bin for camera centers
        cam_centers = []
        if images_bin:
            buf2 = io.BytesIO(images_bin)
            num_images = struct.unpack("<Q", buf2.read(8))[0]
            for _ in range(num_images):
                struct.unpack("<I", buf2.read(4))  # image_id
                qw, qx, qy, qz = struct.unpack("<dddd", buf2.read(32))
                tx, ty, tz = struct.unpack("<ddd", buf2.read(24))
                struct.unpack("<I", buf2.read(4))  # camera_id
                while buf2.read(1) != b"\x00":
                    pass
                num_points2D = struct.unpack("<Q", buf2.read(8))[0]
                buf2.read(num_points2D * 24)

                R = quat_to_rotation(qw, qx, qy, qz)
                t_vec = np.array([tx, ty, tz])
                cam_centers.append(-R.T @ t_vec)

        return points_xyz, points_rgb, np.array(cam_centers)


def load_glb_points(path: str, max_points: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Load point cloud from GLB. Returns (points, colors_float)."""
    scene = trimesh.load(str(path))
    points_list, colors_list = [], []

    if isinstance(scene, trimesh.Scene):
        for name, geom in scene.geometry.items():
            if hasattr(geom, "vertices"):
                transform = scene.graph.get(name)
                if transform is not None:
                    matrix, _ = transform
                    geom = geom.copy()
                    geom.apply_transform(matrix)

                pts = np.array(geom.vertices)
                if hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
                    cols = np.array(geom.visual.vertex_colors)[:, :3] / 255.0
                else:
                    cols = np.ones((len(pts), 3)) * 0.7
                points_list.append(pts)
                colors_list.append(cols)

    all_points = np.concatenate(points_list) if points_list else np.zeros((0, 3))
    all_colors = np.concatenate(colors_list) if colors_list else np.zeros((0, 3))

    if len(all_points) > max_points:
        idx = np.random.choice(len(all_points), max_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]

    print(f"Loaded {len(all_points):,} points from {Path(path).name}")
    return all_points, all_colors


def compute_icp_alignment(colmap_pts, glb_points, n_iters=20):
    """ICP alignment of COLMAP→GLB point clouds. Returns (scale, R, t)."""
    from scipy.spatial import KDTree

    n_align = min(10_000, len(colmap_pts))
    colmap_sub = colmap_pts[np.random.choice(len(colmap_pts), n_align, replace=False)]
    glb_sub = glb_points[np.random.choice(len(glb_points), min(50_000, len(glb_points)), replace=False)]

    scale = np.std(glb_sub) / np.std(colmap_sub)
    R = np.eye(3)
    t = glb_sub.mean(axis=0) - scale * colmap_sub.mean(axis=0)

    print(f"  Initial scale: {scale:.4f}")

    for i in range(n_iters):
        transformed = apply_similarity(colmap_sub, scale, R, t)
        tree = KDTree(glb_sub)
        dists, indices = tree.query(transformed)

        threshold = np.percentile(dists, 70)
        mask = dists < threshold
        if mask.sum() < 10:
            break

        scale, R, t = procrustes_align(colmap_sub[mask], glb_sub[indices[mask]])
        rmse = np.sqrt(np.mean(dists[mask] ** 2))
        if i % 5 == 0:
            print(f"  ICP iter {i}: RMSE={rmse:.4f}, scale={scale:.4f}, inliers={mask.sum()}")

    print(f"  Final: scale={scale:.4f}, RMSE={rmse:.4f}")
    return scale, R, t


def pose_to_world(qw, qx, qy, qz, tx, ty, tz):
    """COLMAP cam_from_world → (cam_pos, T_c2w 4x4)."""
    R = quat_to_rotation(qw, qx, qy, qz)
    t = np.array([tx, ty, tz])
    cam_pos = -R.T @ t
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = cam_pos
    return cam_pos, T


def main():
    parser = argparse.ArgumentParser(description="Pose viewer: point cloud + live camera frustum")
    parser.add_argument("glb", help="Path to .glb point cloud file")
    parser.add_argument("--port", type=int, default=8082, help="Viser port (default: 8082)")
    parser.add_argument("--dpvo-url", default="http://localhost:8091", help="DPVO server URL")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Pose poll interval in seconds")
    parser.add_argument("--reference-tar", default=None, help="Path to reference.tar.gz")
    parser.add_argument("--max-points", type=int, default=500_000, help="Max GLB points to load")
    parser.add_argument("--stream-url", default="http://localhost:3000", help="Next.js stream server URL")
    args = parser.parse_args()

    # Auto-detect reference tar from GLB name
    ref_tar = args.reference_tar
    if ref_tar is None:
        glb_stem = Path(args.glb).stem
        candidate = Path("hloc_localization/data/hloc_reference") / glb_stem / "reference.tar.gz"
        if candidate.exists():
            ref_tar = str(candidate)
            print(f"Auto-detected reference: {ref_tar}")

    if ref_tar is None:
        print("ERROR: No reference tar found. Specify --reference-tar or ensure "
              "hloc_localization/data/hloc_reference/<name>/reference.tar.gz exists.")
        return

    # Load GLB point cloud
    glb_points, glb_colors = load_glb_points(args.glb, args.max_points)

    # Extract COLMAP reconstruction
    print(f"Extracting COLMAP reconstruction from {Path(ref_tar).name}...")
    colmap_pts, colmap_rgb, colmap_cams = extract_colmap_data(ref_tar)
    print(f"  COLMAP 3D points: {len(colmap_pts):,}, cameras: {len(colmap_cams)}")

    # ICP alignment
    print("Computing COLMAP→GLB alignment via ICP...")
    scale, R_align, t_align = compute_icp_alignment(colmap_pts, glb_points)

    # Transform COLMAP points for debug overlay
    colmap_pts_aligned = apply_similarity(colmap_pts, scale, R_align, t_align)

    # Start viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    # GLB point cloud
    server.scene.add_point_cloud(
        "/scene/glb",
        points=glb_points.astype(np.float32),
        colors=glb_colors.astype(np.float32),
        point_size=0.004,
        point_shape="rounded",
    )

    # COLMAP points (aligned)
    if len(colmap_pts_aligned) > 100_000:
        vis_idx = np.random.choice(len(colmap_pts_aligned), 100_000, replace=False)
        colmap_vis_pts = colmap_pts_aligned[vis_idx].astype(np.float32)
        colmap_vis_rgb = colmap_rgb[vis_idx].astype(np.float32)
    else:
        colmap_vis_pts = colmap_pts_aligned.astype(np.float32)
        colmap_vis_rgb = colmap_rgb.astype(np.float32)

    server.scene.add_point_cloud(
        "/scene/colmap_pts",
        points=colmap_vis_pts,
        colors=colmap_vis_rgb,
        point_size=0.006,
        point_shape="circle",
    )

    # Reference cameras (aligned) as small blue frustums
    colmap_cams_aligned = apply_similarity(colmap_cams, scale, R_align, t_align)
    step = max(1, len(colmap_cams_aligned) // 10)
    for i in range(0, len(colmap_cams_aligned), step):
        server.scene.add_camera_frustum(
            f"/scene/ref_cameras/{i}",
            fov=60.0,
            aspect=16 / 9,
            scale=0.06,
            color=(80, 80, 200),
            wxyz=(1, 0, 0, 0),
            position=tuple(colmap_cams_aligned[i]),
            line_width=1.0,
        )

    # Axes
    server.scene.add_frame("/origin", axes_length=0.3, axes_radius=0.008)

    # Initial view
    centroid = glb_points.mean(axis=0)
    bbox_extent = glb_points.max(axis=0) - glb_points.min(axis=0)
    cam_distance = float(np.linalg.norm(bbox_extent)) * 0.8

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = tuple(centroid + np.array([0, -cam_distance * 0.3, -cam_distance * 0.5]))
        client.camera.look_at = tuple(centroid)
        client.camera.up_direction = (0.0, -1.0, 0.0)

    # GUI
    with server.gui.add_folder("Live Feed"):
        gui_feed = server.gui.add_image(
            np.zeros((120, 213, 3), dtype=np.uint8),
            label="Camera",
            format="jpeg",
            jpeg_quality=70,
        )

    with server.gui.add_folder("Controls"):
        gui_follow = server.gui.add_checkbox("Follow camera (POV)", initial_value=True)
        gui_trail = server.gui.add_checkbox("Show trail", initial_value=True)
        gui_show_colmap = server.gui.add_checkbox("Show COLMAP pts", initial_value=False)
        gui_show_ref_cams = server.gui.add_checkbox("Show ref cameras", initial_value=True)
        gui_status = server.gui.add_text("Status", initial_value="Waiting for pose...", disabled=True)

    with server.gui.add_folder("Alignment"):
        server.gui.add_markdown(
            f"**Scale:** {scale:.4f}\n\n"
            f"**COLMAP pts:** {len(colmap_pts):,}\n\n"
            f"**GLB pts:** {len(glb_points):,}\n\n"
            f"**Ref cameras:** {len(colmap_cams)}"
        )

    @gui_show_colmap.on_update
    def _toggle_colmap(_):
        if gui_show_colmap.value:
            server.scene.add_point_cloud("/scene/colmap_pts", points=colmap_vis_pts,
                                         colors=colmap_vis_rgb, point_size=0.006, point_shape="circle")
        else:
            server.scene.add_point_cloud("/scene/colmap_pts", points=np.zeros((1, 3), dtype=np.float32),
                                         colors=np.ones((1, 3), dtype=np.float32), point_size=0.0, point_shape="circle")

    # Pose polling
    trail_positions: list[np.ndarray] = []
    pose_count = 0

    def poll_pose():
        nonlocal pose_count
        last_pose_str = None

        while True:
            try:
                resp = requests.get(f"{args.dpvo_url}/status", timeout=5)
                data = resp.json()
                anchor = data.get("anchor_pose")

                if anchor and anchor.get("success"):
                    pose_str = f"{anchor['tx']:.6f},{anchor['ty']:.6f},{anchor['tz']:.6f}"
                    if pose_str == last_pose_str:
                        time.sleep(args.poll_interval)
                        continue
                    last_pose_str = pose_str

                    # Convert COLMAP pose → world → apply ICP alignment
                    cam_pos, T_c2w = pose_to_world(
                        anchor["qw"], anchor["qx"], anchor["qy"], anchor["qz"],
                        anchor["tx"], anchor["ty"], anchor["tz"],
                    )

                    # Apply similarity transform to position
                    pos_aligned = apply_similarity(cam_pos.reshape(1, 3), scale, R_align, t_align)[0]

                    # Apply rotation alignment: R_aligned = R_icp @ R_c2w
                    R_c2w = T_c2w[:3, :3]
                    R_aligned = R_align @ R_c2w

                    pose_count += 1
                    inliers = anchor.get("num_inliers", 0)

                    print(f"Pose #{pose_count}: pos=({pos_aligned[0]:.2f}, {pos_aligned[1]:.2f}, {pos_aligned[2]:.2f}), "
                          f"inliers={inliers}")

                    # COLMAP: camera looks along +Z, Y is down
                    forward = R_aligned @ np.array([0.0, 0.0, 1.0])
                    cam_up = R_aligned @ np.array([0.0, -1.0, 0.0])

                    # Move all connected viser clients to this viewpoint
                    if gui_follow.value:
                        look_at = pos_aligned + forward * 1.0
                        for client in server.get_clients().values():
                            with client.atomic():
                                client.camera.position = tuple(pos_aligned)
                                client.camera.look_at = tuple(look_at)
                                client.camera.up_direction = tuple(cam_up)

                    # Trail markers
                    if gui_trail.value:
                        trail_positions.append(pos_aligned.astype(np.float32))
                        if len(trail_positions) >= 2:
                            trail_pts = np.array(trail_positions, dtype=np.float32)
                            # Green trail
                            trail_colors = np.full((len(trail_pts), 3), [0, 220, 80], dtype=np.uint8)
                            server.scene.add_point_cloud(
                                name="/scene/trail",
                                points=trail_pts,
                                colors=trail_colors,
                                point_size=0.025,
                                point_shape="circle",
                            )

                    gui_status.value = (
                        f"Pose #{pose_count}: "
                        f"({pos_aligned[0]:.2f}, {pos_aligned[1]:.2f}, {pos_aligned[2]:.2f}) "
                        f"inliers={inliers}"
                    )

                else:
                    gui_status.value = "No anchor pose yet"

            except requests.ConnectionError:
                gui_status.value = f"DPVO server offline ({args.dpvo_url})"
            except Exception as e:
                gui_status.value = f"Error: {e}"
                print(f"Poll error: {e}")

            time.sleep(args.poll_interval)

    thread = threading.Thread(target=poll_pose, daemon=True)
    thread.start()

    # Frame-fetching thread: pull latest JPEG from Next.js and show in GUI
    def poll_frame():
        frame_url = f"{args.stream_url}/api/stream/frame/jpeg"
        target_w = 320
        while True:
            try:
                resp = requests.get(frame_url, timeout=2)
                if resp.status_code == 200:
                    jpg_bytes = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Resize to fit GUI panel
                        h, w = img.shape[:2]
                        scale_f = target_w / w
                        img = cv2.resize(img, (target_w, int(h * scale_f)))
                        # BGR -> RGB for viser
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        gui_feed.image = img
            except Exception:
                pass
            time.sleep(0.15)  # ~7 FPS in the GUI panel

    frame_thread = threading.Thread(target=poll_frame, daemon=True)
    frame_thread.start()

    print(f"\nPose viewer at http://localhost:{args.port}")
    print(f"Polling DPVO server at {args.dpvo_url}")
    print(f"Alignment: scale={scale:.4f}")
    print("Waiting for localized poses...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
