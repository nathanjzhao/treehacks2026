"""
View localized camera trajectory in 3D point cloud using viser.

Loads a GLB scene, extracts COLMAP 3D points from the reference tar,
computes a similarity transform to align COLMAP→GLB coordinate systems,
then overlays aligned camera frustums on the GLB point cloud.

Usage:
  python -m hloc_localization.frontend.view_trajectory
"""

import io
import tarfile
import time
import pathlib

import numpy as np
import trimesh
import viser
import viser.transforms as vtf


def quat_to_rotation(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


def procrustes_align(src, dst):
    """
    Compute similarity transform (scale, rotation, translation) that maps src → dst.
    Uses Procrustes analysis. Both are (N, 3) arrays.
    Returns: scale, R (3x3), t (3,) such that dst ≈ scale * R @ src + t
    """
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Scale
    s_src = np.sqrt((src_c ** 2).sum() / len(src))
    s_dst = np.sqrt((dst_c ** 2).sum() / len(dst))
    scale = s_dst / s_src

    # Rotation via SVD
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


def extract_colmap_points(tar_path):
    """Extract 3D points and camera centers from COLMAP reconstruction in the reference tar."""
    import struct

    with tarfile.open(tar_path, "r:gz") as tf:
        # Read points3D.bin
        points3d = None
        images_bin = None
        for member in tf.getmembers():
            if member.name.endswith("points3D.bin"):
                points3d = tf.extractfile(member).read()
            elif member.name.endswith("images.bin"):
                images_bin = tf.extractfile(member).read()

        if points3d is None:
            raise RuntimeError("points3D.bin not found in tar")

        # Parse points3D.bin (COLMAP binary format)
        pts = []
        buf = io.BytesIO(points3d)
        num_points = struct.unpack("<Q", buf.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", buf.read(8))[0]
            xyz = struct.unpack("<ddd", buf.read(24))
            rgb = struct.unpack("<BBB", buf.read(3))
            error = struct.unpack("<d", buf.read(8))[0]
            track_length = struct.unpack("<Q", buf.read(8))[0]
            # Skip track entries (image_id + point2D_idx, each 4 bytes)
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
                image_id = struct.unpack("<I", buf2.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<dddd", buf2.read(32))
                tx, ty, tz = struct.unpack("<ddd", buf2.read(24))
                camera_id = struct.unpack("<I", buf2.read(4))[0]
                # Read name (null-terminated)
                name = b""
                while True:
                    c = buf2.read(1)
                    if c == b"\x00":
                        break
                    name += c
                # Read points2D
                num_points2D = struct.unpack("<Q", buf2.read(8))[0]
                buf2.read(num_points2D * 24)  # x, y, point3D_id per point

                R = quat_to_rotation(qw, qx, qy, qz)
                t = np.array([tx, ty, tz])
                cam_centers.append(-R.T @ t)

        return points_xyz, points_rgb, np.array(cam_centers)


def load_glb_points(glb_path, max_points=500_000):
    """Load points and colors from a GLB file."""
    scene = trimesh.load(str(glb_path))

    points = []
    colors = []
    if isinstance(scene, trimesh.Scene):
        for name, geom in scene.geometry.items():
            if hasattr(geom, "vertices"):
                pts = np.array(geom.vertices)
                if hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
                    cols = np.array(geom.visual.vertex_colors)[:, :3] / 255.0
                else:
                    cols = np.ones((len(pts), 3)) * 0.7
                points.append(pts)
                colors.append(cols)
    elif hasattr(scene, "vertices"):
        points.append(np.array(scene.vertices))
        if hasattr(scene, "visual") and hasattr(scene.visual, "vertex_colors"):
            colors.append(np.array(scene.visual.vertex_colors)[:, :3] / 255.0)
        else:
            colors.append(np.ones((len(scene.vertices), 3)) * 0.7)

    all_points = np.concatenate(points) if points else np.zeros((0, 3))
    all_colors = np.concatenate(colors) if colors else np.zeros((0, 3))

    if len(all_points) > max_points:
        idx = np.random.choice(len(all_points), max_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]

    return all_points, all_colors


# Localization results: IMG_4730 against IMG_4720 reference
POSES = [
    {"tx": -1.356, "ty": -0.480, "tz": 6.893, "qw": 0.988, "qx": -0.090, "qy": 0.126, "qz": 0.022, "inliers": 8380},
    {"tx": -0.753, "ty": -0.636, "tz": 6.195, "qw": 0.983, "qx": -0.063, "qy": 0.170, "qz": 0.038, "inliers": 5786},
    {"tx": -0.267, "ty": -0.522, "tz": 4.587, "qw": 0.973, "qx": -0.029, "qy": 0.225, "qz": 0.050, "inliers": 7941},
    {"tx": 0.403, "ty": -0.283, "tz": 3.377, "qw": 0.970, "qx": -0.013, "qy": 0.238, "qz": 0.050, "inliers": 7474},
    {"tx": 1.351, "ty": -0.013, "tz": 1.723, "qw": 0.976, "qx": -0.005, "qy": 0.216, "qz": 0.041, "inliers": 8651},
    {"tx": 1.864, "ty": 0.116, "tz": -0.059, "qw": 0.985, "qx": -0.002, "qy": 0.172, "qz": 0.024, "inliers": 8776},
    {"tx": 2.440, "ty": -0.543, "tz": -1.577, "qw": 0.991, "qx": -0.034, "qy": 0.129, "qz": 0.007, "inliers": 4045},
    {"tx": 2.781, "ty": -1.547, "tz": -1.993, "qw": 0.989, "qx": -0.099, "qy": 0.114, "qz": 0.000, "inliers": 3549},
    {"tx": 2.986, "ty": -1.735, "tz": -1.792, "qw": 0.988, "qx": -0.118, "qy": 0.098, "qz": 0.004, "inliers": 3597},
    {"tx": 2.993, "ty": -1.829, "tz": -1.613, "qw": 0.987, "qx": -0.125, "qy": 0.097, "qz": 0.005, "inliers": 4290},
]


def pose_to_world(p):
    """Convert cam_from_world (R, t) to camera position + 4x4 camera-to-world matrix."""
    R = quat_to_rotation(p["qw"], p["qx"], p["qy"], p["qz"])
    t = np.array([p["tx"], p["ty"], p["tz"]])
    cam_pos = -R.T @ t
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = cam_pos
    return cam_pos, T


def main():
    base = pathlib.Path(__file__).parent.parent.parent
    glb_path = base / "data" / "mapanything" / "IMG_4720.glb"
    ref_tar_path = base / "hloc_localization" / "data" / "hloc_reference" / "IMG_4720" / "reference.tar.gz"

    # Load GLB point cloud
    print(f"Loading GLB: {glb_path.name} ({glb_path.stat().st_size / 1024 / 1024:.0f} MB)")
    glb_points, glb_colors = load_glb_points(glb_path, max_points=500_000)
    print(f"  GLB points: {len(glb_points):,}")

    # Extract COLMAP points from reference tar
    print(f"Extracting COLMAP reconstruction from {ref_tar_path.name}...")
    colmap_pts, colmap_rgb, colmap_cams = extract_colmap_points(str(ref_tar_path))
    print(f"  COLMAP 3D points: {len(colmap_pts):,}")
    print(f"  COLMAP cameras: {len(colmap_cams)}")

    # Compute alignment: COLMAP → GLB using camera centers vs GLB centroid heuristic
    # Better approach: use COLMAP 3D points and GLB points with subsampled ICP
    print("Computing COLMAP→GLB alignment via Procrustes on point cloud centroids...")

    # Subsample both point clouds for fast alignment
    n_align = min(10_000, len(colmap_pts), len(glb_points))

    # Use nearest-neighbor matching between subsampled clouds
    from scipy.spatial import KDTree

    # Subsample COLMAP points
    colmap_sub_idx = np.random.choice(len(colmap_pts), min(n_align, len(colmap_pts)), replace=False)
    colmap_sub = colmap_pts[colmap_sub_idx]

    # Build KDTree of GLB points (subsample for speed)
    glb_sub_idx = np.random.choice(len(glb_points), min(50_000, len(glb_points)), replace=False)
    glb_sub = glb_points[glb_sub_idx]

    # Iterative closest point (simple version)
    # Start with Procrustes on centroids + scale
    scale_init = np.std(glb_sub) / np.std(colmap_sub)
    R_best = np.eye(3)
    t_best = glb_sub.mean(axis=0) - scale_init * colmap_sub.mean(axis=0)
    scale_best = scale_init

    print(f"  Initial scale estimate: {scale_init:.4f}")

    for icp_iter in range(20):
        # Transform COLMAP points
        transformed = apply_similarity(colmap_sub, scale_best, R_best, t_best)

        # Find nearest neighbors in GLB
        tree = KDTree(glb_sub)
        dists, indices = tree.query(transformed)

        # Filter outliers (keep closest 70%)
        threshold = np.percentile(dists, 70)
        mask = dists < threshold
        if mask.sum() < 10:
            break

        src_matched = colmap_sub[mask]
        dst_matched = glb_sub[indices[mask]]

        # Re-estimate transform
        scale_best, R_best, t_best = procrustes_align(src_matched, dst_matched)

        rmse = np.sqrt(np.mean(dists[mask] ** 2))
        if icp_iter % 5 == 0:
            print(f"  ICP iter {icp_iter}: RMSE={rmse:.4f}, scale={scale_best:.4f}, inliers={mask.sum()}")

    print(f"  Final: scale={scale_best:.4f}, RMSE={rmse:.4f}")

    # Transform camera poses into GLB coordinate system
    cam_positions_aligned = []
    cam_transforms_aligned = []
    for p in POSES:
        pos, T = pose_to_world(p)
        # Transform position
        pos_aligned = apply_similarity(pos.reshape(1, 3), scale_best, R_best, t_best)[0]
        # Transform rotation: R_aligned = R_icp @ R_cam
        R_cam = T[:3, :3]
        R_aligned = R_best @ R_cam

        T_aligned = np.eye(4)
        T_aligned[:3, :3] = R_aligned
        T_aligned[:3, 3] = pos_aligned

        cam_positions_aligned.append(pos_aligned)
        cam_transforms_aligned.append(T_aligned)

    cam_positions_aligned = np.array(cam_positions_aligned)

    # Also transform COLMAP points for debug visualization
    colmap_pts_aligned = apply_similarity(colmap_pts, scale_best, R_best, t_best)

    # Start viser
    server = viser.ViserServer(host="0.0.0.0", port=8890)
    print("\nViser server at http://localhost:8890")

    # GLB point cloud
    server.scene.add_point_cloud(
        "/scene/glb",
        points=glb_points.astype(np.float32),
        colors=glb_colors.astype(np.float32),
        point_size=0.004,
        point_shape="rounded",
    )

    # COLMAP points (aligned) — smaller, red-tinted for debugging
    colmap_sub_vis = colmap_pts_aligned
    if len(colmap_sub_vis) > 100_000:
        idx = np.random.choice(len(colmap_sub_vis), 100_000, replace=False)
        colmap_sub_vis = colmap_sub_vis[idx]
        colmap_rgb_vis = colmap_rgb[idx]
    else:
        colmap_rgb_vis = colmap_rgb

    server.scene.add_point_cloud(
        "/scene/colmap_pts",
        points=colmap_sub_vis.astype(np.float32),
        colors=colmap_rgb_vis.astype(np.float32),
        point_size=0.006,
        point_shape="circle",
    )

    # Camera trajectory path
    server.scene.add_point_cloud(
        "/scene/trajectory",
        points=cam_positions_aligned.astype(np.float32),
        colors=np.array([[1.0, 0.5, 0.0]] * len(cam_positions_aligned), dtype=np.float32),
        point_size=0.03,
        point_shape="circle",
    )

    # All frustums (dimmed)
    max_inliers = max(p["inliers"] for p in POSES)
    for i, (p, T) in enumerate(zip(POSES, cam_transforms_aligned)):
        wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
        pos = T[:3, 3]
        brightness = int(100 + 155 * p["inliers"] / max_inliers)
        server.scene.add_camera_frustum(
            f"/scene/cameras/frame_{i:02d}",
            fov=60.0,
            aspect=16/9,
            scale=0.12,
            wxyz=wxyz,
            position=pos,
            color=(80, 80, brightness),
        )

    # Active frustum
    active_frustum = server.scene.add_camera_frustum(
        "/scene/active_camera",
        fov=60.0,
        aspect=16/9,
        scale=0.2,
        wxyz=(1, 0, 0, 0),
        position=(0, 0, 0),
        color=(0, 255, 0),
    )

    # GUI
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(POSES)-1, step=1, initial_value=0)
        play_btn = server.gui.add_button("Play")
        speed_slider = server.gui.add_slider("Speed (s/frame)", min=0.1, max=2.0, step=0.1, initial_value=0.5)

    with server.gui.add_folder("Visibility"):
        show_colmap = server.gui.add_checkbox("Show COLMAP points", initial_value=True)
        show_glb = server.gui.add_checkbox("Show GLB points", initial_value=True)

    with server.gui.add_folder("Info"):
        info_text = server.gui.add_markdown("")

    with server.gui.add_folder("Alignment"):
        server.gui.add_markdown(
            f"**Scale:** {scale_best:.4f}\n\n"
            f"**RMSE:** {rmse:.4f}\n\n"
            f"**COLMAP pts:** {len(colmap_pts):,}\n\n"
            f"**GLB pts:** {len(glb_points):,}"
        )

    playing = [False]

    @frame_slider.on_update
    def _on_frame(event):
        i = frame_slider.value
        T = cam_transforms_aligned[i]
        wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
        pos = T[:3, 3]
        active_frustum.wxyz = wxyz
        active_frustum.position = pos
        info_text.content = (
            f"**Frame {i}** | "
            f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
            f"Inliers: {POSES[i]['inliers']}"
        )

    @play_btn.on_click
    def _on_play(event):
        playing[0] = not playing[0]
        play_btn.name = "Pause" if playing[0] else "Play"

    @show_colmap.on_update
    def _toggle_colmap(event):
        server.scene.add_point_cloud(
            "/scene/colmap_pts",
            points=colmap_sub_vis.astype(np.float32) if show_colmap.value else np.zeros((1, 3), dtype=np.float32),
            colors=colmap_rgb_vis.astype(np.float32) if show_colmap.value else np.ones((1, 3), dtype=np.float32),
            point_size=0.006 if show_colmap.value else 0.0,
            point_shape="circle",
        )

    @show_glb.on_update
    def _toggle_glb(event):
        server.scene.add_point_cloud(
            "/scene/glb",
            points=glb_points.astype(np.float32) if show_glb.value else np.zeros((1, 3), dtype=np.float32),
            colors=glb_colors.astype(np.float32) if show_glb.value else np.ones((1, 3), dtype=np.float32),
            point_size=0.004 if show_glb.value else 0.0,
            point_shape="rounded",
        )

    _on_frame(None)

    print("Ready! Open http://localhost:8890")
    print("Use slider or Play button to animate camera through scene.")

    while True:
        if playing[0]:
            i = (frame_slider.value + 1) % len(POSES)
            frame_slider.value = i
            time.sleep(speed_slider.value)
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
