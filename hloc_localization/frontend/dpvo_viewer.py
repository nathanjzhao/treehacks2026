"""
Offline DPVO result viewer — visualize camera poses over a point cloud.

Usage:
  python -m hloc_localization.frontend.dpvo_viewer data/mapanything/IMG_4720.glb \
    --result hloc_localization/data/dpvo_results/IMG_4730_dpvo.json
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
import viser


def _camera_apex(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.array(mesh.faces)
    verts = np.array(mesh.vertices)
    counts = Counter(faces.flatten().tolist())
    apex_idx = max(counts, key=counts.get)
    return verts[apex_idx]


def load_glb(path: str) -> tuple[np.ndarray, np.ndarray, list[trimesh.Trimesh]]:
    scene = trimesh.load(path)
    points = None
    colors = None
    camera_meshes = []

    for name, geom in scene.geometry.items():
        transform = scene.graph.get(name)
        if transform is not None:
            matrix, _ = transform
            geom = geom.copy()
            geom.apply_transform(matrix)
        if isinstance(geom, trimesh.PointCloud):
            points = np.array(geom.vertices, dtype=np.float32)
            c = np.array(geom.colors, dtype=np.uint8)
            colors = c[:, :3] if c.shape[1] == 4 else c
        elif isinstance(geom, trimesh.Trimesh):
            camera_meshes.append(geom)

    if points is None:
        raise ValueError(f"No PointCloud found in {path}")
    print(f"Loaded {len(points):,} points, {len(camera_meshes)} camera cones")
    return points, colors, camera_meshes


def qvec_to_rotation(qw, qx, qy, qz):
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])


def pose_to_cam_position(qw, qx, qy, qz, tx, ty, tz):
    """COLMAP cam_from_world → camera position in world coords."""
    R = qvec_to_rotation(qw, qx, qy, qz)
    t = np.array([tx, ty, tz])
    return -R.T @ t


def pose_to_wxyz_position(qw, qx, qy, qz, tx, ty, tz):
    """COLMAP cam_from_world → (wxyz quaternion, position) for camera-to-world."""
    R = qvec_to_rotation(qw, qx, qy, qz)
    t = np.array([tx, ty, tz])
    R_c2w = R.T
    t_c2w = -R_c2w @ t
    # rotation matrix to quaternion
    tr = R_c2w[0, 0] + R_c2w[1, 1] + R_c2w[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R_c2w[2, 1] - R_c2w[1, 2]) * s
        y = (R_c2w[0, 2] - R_c2w[2, 0]) * s
        z = (R_c2w[1, 0] - R_c2w[0, 1]) * s
    elif R_c2w[0, 0] > R_c2w[1, 1] and R_c2w[0, 0] > R_c2w[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R_c2w[0, 0] - R_c2w[1, 1] - R_c2w[2, 2])
        w = (R_c2w[2, 1] - R_c2w[1, 2]) / s
        x = 0.25 * s
        y = (R_c2w[0, 1] + R_c2w[1, 0]) / s
        z = (R_c2w[0, 2] + R_c2w[2, 0]) / s
    elif R_c2w[1, 1] > R_c2w[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R_c2w[1, 1] - R_c2w[0, 0] - R_c2w[2, 2])
        w = (R_c2w[0, 2] - R_c2w[2, 0]) / s
        x = (R_c2w[0, 1] + R_c2w[1, 0]) / s
        y = 0.25 * s
        z = (R_c2w[1, 2] + R_c2w[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R_c2w[2, 2] - R_c2w[0, 0] - R_c2w[1, 1])
        w = (R_c2w[1, 0] - R_c2w[0, 1]) / s
        x = (R_c2w[0, 2] + R_c2w[2, 0]) / s
        y = (R_c2w[1, 2] + R_c2w[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z]), t_c2w


def main():
    parser = argparse.ArgumentParser(description="DPVO Result Viewer")
    parser.add_argument("glb", help="Path to .glb point cloud")
    parser.add_argument("--result", required=True, help="Path to DPVO result JSON")
    parser.add_argument("--port", type=int, default=8083, help="Viser port")
    parser.add_argument("--downsample", type=int, default=4, help="Keep every Nth point")
    parser.add_argument("--frustum-scale", type=float, default=0.3,
                        help="Camera frustum size")
    args = parser.parse_args()

    # Load point cloud
    points, colors, camera_meshes = load_glb(args.glb)
    if args.downsample > 1:
        idx = np.arange(0, len(points), args.downsample)
        points = points[idx]
        colors = colors[idx]
        print(f"Downsampled to {len(points):,} points")

    # Load DPVO result
    result = json.loads(Path(args.result).read_text())
    poses = result["poses"]
    print(f"Loaded {len(poses)} poses")

    # Compute camera positions
    cam_positions = []
    cam_wxyz = []
    for p in poses:
        wxyz, pos = pose_to_wxyz_position(
            p["qw"], p["qx"], p["qy"], p["qz"],
            p["tx"], p["ty"], p["tz"],
        )
        cam_positions.append(pos)
        cam_wxyz.append(wxyz)
    cam_positions = np.array(cam_positions)

    pc_centroid = points.mean(axis=0)
    traj_centroid = cam_positions.mean(axis=0)
    scene_center = (pc_centroid + traj_centroid) / 2

    print(f"Point cloud centroid: {pc_centroid}")
    print(f"Trajectory centroid:  {traj_centroid}")
    print(f"Camera range: {cam_positions.min(axis=0)} → {cam_positions.max(axis=0)}")

    # Start viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    # Add point cloud
    server.scene.add_point_cloud(
        name="/point_cloud",
        points=points,
        colors=colors,
        point_size=0.005,
        point_shape="rounded",
    )

    # Add camera frustums
    fscale = args.frustum_scale
    for i, p in enumerate(poses):
        wxyz = cam_wxyz[i]
        pos = cam_positions[i]
        is_anchor = p.get("source") == "hloc"

        if is_anchor:
            name = f"/cameras/anchor_{i:03d}"
            color = (0, 255, 100)
            scale = fscale * 1.5
        else:
            name = f"/cameras/dpvo_{i:03d}"
            color = (255, 100, 50)
            scale = fscale

        server.scene.add_camera_frustum(
            name=name,
            fov=np.deg2rad(60),
            aspect=16 / 9,
            scale=scale,
            wxyz=wxyz,
            position=tuple(pos),
            color=color,
        )

    # Trajectory line
    if len(cam_positions) >= 2:
        # Draw line segments between consecutive cameras
        for i in range(len(cam_positions) - 1):
            pts = np.array([cam_positions[i], cam_positions[i + 1]])
            color_val = (255, 100, 50)
            if poses[i].get("source") == "hloc":
                color_val = (0, 255, 100)
            colors_line = np.array([color_val, color_val], dtype=np.uint8)
            server.scene.add_point_cloud(
                name=f"/trajectory/seg_{i:03d}",
                points=pts,
                colors=colors_line,
                point_size=0.015,
                point_shape="circle",
            )

        # Also add full trajectory as a point cloud for visibility
        traj_colors = np.full((len(cam_positions), 3), [255, 180, 50], dtype=np.uint8)
        for i, p in enumerate(poses):
            if p.get("source") == "hloc":
                traj_colors[i] = [0, 255, 100]
        server.scene.add_point_cloud(
            name="/trajectory/points",
            points=cam_positions,
            colors=traj_colors,
            point_size=0.02,
            point_shape="circle",
        )

    # GUI controls
    with server.gui.add_folder("Controls"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.005
        )
        gui_frustum_visible = server.gui.add_checkbox("Show frustums", initial_value=True)
        gui_traj_visible = server.gui.add_checkbox("Show trajectory", initial_value=True)

        @gui_point_size.on_update
        def _(_):
            server.scene.add_point_cloud(
                name="/point_cloud",
                points=points,
                colors=colors if 'colors' in dir() else None,
                point_size=gui_point_size.value,
                point_shape="rounded",
            )

        @gui_frustum_visible.on_update
        def _(_):
            visible = gui_frustum_visible.value
            for i in range(len(poses)):
                is_anchor = poses[i].get("source") == "hloc"
                n = f"/cameras/anchor_{i:03d}" if is_anchor else f"/cameras/dpvo_{i:03d}"
                try:
                    server.scene.set_visibility(n, visible)
                except Exception:
                    pass

        @gui_traj_visible.on_update
        def _(_):
            visible = gui_traj_visible.value
            try:
                server.scene.set_visibility("/trajectory", visible)
            except Exception:
                pass

    # Info display
    with server.gui.add_folder("Info"):
        server.gui.add_markdown(
            f"**Poses**: {len(poses)} ({sum(1 for p in poses if p.get('source')=='hloc')} HLoc anchor, "
            f"{sum(1 for p in poses if p.get('source')=='dpvo')} DPVO)\n\n"
            f"**DPVO FPS**: {result.get('dpvo_fps', 0):.1f}\n\n"
            f"**Trajectory length**: {sum(np.linalg.norm(cam_positions[i+1]-cam_positions[i]) for i in range(len(cam_positions)-1)):.2f}m"
        )

    # Set initial camera to see both point cloud and trajectory
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # Position camera to see everything - look from above/behind the trajectory
        cam_back = cam_positions[0] - (traj_centroid - cam_positions[0]) * 2
        cam_back[2] += 2  # slightly above
        client.camera.position = tuple(cam_back)
        client.camera.look_at = tuple(scene_center)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    print(f"\nViewer running at http://localhost:{args.port}")
    print(f"  {len(poses)} camera frustums rendered")
    print(f"  Green = HLoc anchor, Orange = DPVO odometry")

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
