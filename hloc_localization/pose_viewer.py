"""
Viser-based 3D viewer: shows point cloud + live camera frustum from HLoc localization.

Loads BOTH the MapAnything GLB (for pretty visuals) and the COLMAP reconstruction
(from the HLoc reference tar) so we can place localized poses in the correct frame.

Uses Umeyama alignment to map COLMAP cameras into GLB space so the frustum
appears at the right place in the point cloud.

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

import numpy as np
import requests
import trimesh
import viser
from scipy.spatial.transform import Rotation


def load_glb_points(path: str, downsample: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Load point cloud from a MapAnything GLB file."""
    scene = trimesh.load(path)

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

            if downsample > 1:
                idx = np.arange(0, len(points), downsample)
                points = points[idx]
                colors = colors[idx]

            print(f"Loaded {len(points):,} points from {Path(path).name}")
            return points, colors

    raise ValueError(f"No PointCloud found in {path}")


def load_glb_cameras(path: str) -> list[np.ndarray]:
    """Extract camera positions from GLB camera cone meshes."""
    from collections import Counter
    scene = trimesh.load(path)
    positions = []
    for name, geom in scene.geometry.items():
        transform = scene.graph.get(name)
        if transform is not None:
            matrix, _ = transform
            geom = geom.copy()
            geom.apply_transform(matrix)
        if isinstance(geom, trimesh.Trimesh):
            # Camera cone: apex is the vertex shared by the most faces
            faces = np.array(geom.faces)
            counts = Counter(faces.flatten().tolist())
            apex_idx = max(counts, key=counts.get)
            positions.append(np.array(geom.vertices[apex_idx]))
    return positions


def load_colmap_cameras(tar_path: str) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Load camera world positions and frame names from COLMAP images.bin in a reference tar.

    Returns list of (position, R_wc_quat_wxyz, frame_name).
    """
    cameras = []
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if "images.bin" in m.name:
                f = tf.extractfile(m)
                data = f.read()
                num = struct.unpack("<Q", data[:8])[0]
                offset = 8
                for _ in range(num):
                    img_id = struct.unpack("<I", data[offset:offset+4])[0]; offset += 4
                    qw, qx, qy, qz = struct.unpack("<4d", data[offset:offset+32]); offset += 32
                    tx, ty, tz = struct.unpack("<3d", data[offset:offset+24]); offset += 24
                    cam_id = struct.unpack("<I", data[offset:offset+4])[0]; offset += 4
                    name = b""
                    while data[offset] != 0:
                        name += bytes([data[offset]])
                        offset += 1
                    offset += 1
                    num_pts = struct.unpack("<Q", data[offset:offset+8])[0]; offset += 8
                    offset += num_pts * 24

                    R_cw = Rotation.from_quat([qx, qy, qz, qw])
                    R_wc = R_cw.inv()
                    pos = R_wc.apply(-np.array([tx, ty, tz]))
                    xyzw = R_wc.as_quat()
                    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
                    cameras.append((pos, wxyz, name.decode()))
                break
    print(f"Loaded {len(cameras)} COLMAP cameras from reference")
    return cameras


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute similarity transform (scale, rotation, translation) from src to dst.

    dst = scale * R @ src + t
    Returns (scale, R [3x3], t [3]).
    """
    assert src.shape == dst.shape
    n, d = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / n

    cov = dst_c.T @ src_c / n

    U, S, Vt = np.linalg.svd(cov)

    det_sign = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.eye(d)
    if det_sign < 0:
        D[-1, -1] = -1

    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / var_src
    t = mu_dst - scale * R @ mu_src

    return scale, R, t


def match_cameras_by_frame_order(colmap_cams: list, glb_cam_positions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Match COLMAP cameras to GLB cameras by sorting both by frame number/position.

    Returns matched (colmap_positions, glb_positions) arrays.
    """
    # Sort COLMAP cameras by frame name (frame_000001.jpg -> 1)
    sorted_colmap = sorted(colmap_cams, key=lambda c: c[2])
    colmap_positions = np.array([c[0] for c in sorted_colmap])

    # GLB cameras don't have names, but should be in similar order
    glb_positions = np.array(glb_cam_positions)

    # Use min of both counts
    n = min(len(colmap_positions), len(glb_positions))
    if n < 3:
        print(f"Warning: only {n} matched cameras, alignment may be poor")

    # Subsample evenly if counts differ significantly
    if len(colmap_positions) != len(glb_positions):
        # Pick evenly spaced indices from each
        colmap_idx = np.linspace(0, len(colmap_positions) - 1, n, dtype=int)
        glb_idx = np.linspace(0, len(glb_positions) - 1, n, dtype=int)
        return colmap_positions[colmap_idx], glb_positions[glb_idx]

    return colmap_positions[:n], glb_positions[:n]


def colmap_pose_to_world(qw, qx, qy, qz, tx, ty, tz):
    """Convert COLMAP cam_from_world pose to world-frame position + quaternion."""
    R_cw = Rotation.from_quat([qx, qy, qz, qw])
    t_cw = np.array([tx, ty, tz])
    R_wc = R_cw.inv()
    position = R_wc.apply(-t_cw)
    xyzw = R_wc.as_quat()
    wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
    return position, wxyz


def transform_pose(position, wxyz, scale, R_align, t_align):
    """Apply similarity transform to a pose (position + quaternion).

    New position: scale * R_align @ pos + t_align
    New rotation: R_align @ R_wc
    """
    new_pos = scale * R_align @ position + t_align

    # Compose rotations: R_align @ R_wc
    R_wc = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])  # xyzw
    R_aligned = Rotation.from_matrix(R_align) * R_wc
    xyzw = R_aligned.as_quat()
    new_wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])

    return new_pos, new_wxyz


def main():
    parser = argparse.ArgumentParser(description="Pose viewer: point cloud + live camera frustum")
    parser.add_argument("glb", help="Path to .glb point cloud file")
    parser.add_argument("--port", type=int, default=8082, help="Viser port (default: 8082)")
    parser.add_argument("--downsample", type=int, default=20, help="Keep every Nth point (default: 20)")
    parser.add_argument("--dpvo-url", default="http://localhost:8091", help="DPVO server URL")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Pose poll interval in seconds")
    parser.add_argument("--reference-tar", default=None, help="Path to reference.tar.gz (auto-detected from DPVO server)")
    args = parser.parse_args()

    # Load GLB point cloud
    points, colors = load_glb_points(args.glb, args.downsample)
    centroid = points.mean(axis=0)

    # Load GLB camera positions for alignment
    glb_cam_positions = load_glb_cameras(args.glb)
    print(f"Found {len(glb_cam_positions)} camera cones in GLB")

    # Find reference tar
    ref_tar = args.reference_tar
    if ref_tar is None:
        # Try to auto-detect from the GLB name
        glb_stem = Path(args.glb).stem  # e.g. IMG_4741
        candidate = Path("hloc_localization/data/hloc_reference") / glb_stem / "reference.tar.gz"
        if candidate.exists():
            ref_tar = str(candidate)
            print(f"Auto-detected reference: {ref_tar}")

    # Compute COLMAP -> GLB alignment
    scale, R_align, t_align = 1.0, np.eye(3), np.zeros(3)
    has_alignment = False

    if ref_tar and glb_cam_positions:
        colmap_cams = load_colmap_cameras(ref_tar)
        if len(colmap_cams) >= 3 and len(glb_cam_positions) >= 3:
            colmap_pts, glb_pts = match_cameras_by_frame_order(colmap_cams, glb_cam_positions)
            scale, R_align, t_align = umeyama_alignment(colmap_pts, glb_pts)
            has_alignment = True

            # Verify alignment quality
            transformed = scale * (colmap_pts @ R_align.T) + t_align
            errors = np.linalg.norm(transformed - glb_pts, axis=1)
            print(f"Alignment: scale={scale:.4f}, mean_error={errors.mean():.4f}, max_error={errors.max():.4f}")

            # Show aligned COLMAP cameras as blue frustums for verification
            for i, (pos, wxyz, name) in enumerate(colmap_cams[::max(1, len(colmap_cams)//10)]):
                apos, awxyz = transform_pose(pos, wxyz, scale, R_align, t_align)
        else:
            print("Not enough cameras for alignment, using raw COLMAP coordinates")
    else:
        print("No reference tar or GLB cameras found, using raw COLMAP coordinates")

    # Start viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    point_size = 0.005 * (args.downsample ** 0.5)
    server.scene.add_point_cloud(
        name="/point_cloud",
        points=points,
        colors=colors,
        point_size=point_size,
        point_shape="rounded",
    )

    # Add axes at origin for reference
    server.scene.add_frame("/origin", axes_length=0.3, axes_radius=0.008)

    # Show a few aligned COLMAP reference cameras as blue frustums
    if has_alignment and ref_tar:
        colmap_cams = load_colmap_cameras(ref_tar)
        step = max(1, len(colmap_cams) // 8)
        for i, (pos, wxyz, name) in enumerate(colmap_cams[::step]):
            apos, awxyz = transform_pose(pos, wxyz, scale, R_align, t_align)
            server.scene.add_camera_frustum(
                name=f"/ref_cameras/{i}",
                fov=60.0,
                aspect=16 / 9,
                scale=0.08,
                color=(80, 80, 200),
                wxyz=awxyz,
                position=tuple(apos),
                line_width=1.0,
            )

    # Set initial camera
    bbox_extent = points.max(axis=0) - points.min(axis=0)
    cam_distance = float(np.linalg.norm(bbox_extent)) * 0.8

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = tuple(centroid + np.array([0, -cam_distance * 0.3, -cam_distance * 0.5]))
        client.camera.look_at = tuple(centroid)
        client.camera.up_direction = (0.0, -1.0, 0.0)

    # GUI controls
    with server.gui.add_folder("Controls"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.1, step=0.001, initial_value=point_size
        )
        gui_frustum_scale = server.gui.add_slider(
            "Frustum scale", min=0.05, max=1.0, step=0.05, initial_value=0.3
        )
        gui_trail = server.gui.add_checkbox("Show trail", initial_value=True)
        gui_show_ref = server.gui.add_checkbox("Show ref cameras", initial_value=True)
        gui_status = server.gui.add_text("Status", initial_value="Waiting for pose...", disabled=True)

    @gui_point_size.on_update
    def _on_size(_):
        server.scene.add_point_cloud(
            name="/point_cloud",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
            point_shape="rounded",
        )

    @gui_show_ref.on_update
    def _on_show_ref(_):
        if has_alignment and ref_tar:
            colmap_cams_inner = load_colmap_cameras(ref_tar)
            step = max(1, len(colmap_cams_inner) // 8)
            for i, (pos, wxyz, name) in enumerate(colmap_cams_inner[::step]):
                apos, awxyz = transform_pose(pos, wxyz, scale, R_align, t_align)
                server.scene.add_camera_frustum(
                    name=f"/ref_cameras/{i}",
                    fov=60.0,
                    aspect=16 / 9,
                    scale=0.08,
                    color=(80, 80, 200),
                    wxyz=awxyz,
                    position=tuple(apos),
                    line_width=1.0,
                    visible=gui_show_ref.value,
                )

    # Pose polling thread
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

                    position, wxyz = colmap_pose_to_world(
                        anchor["qw"], anchor["qx"], anchor["qy"], anchor["qz"],
                        anchor["tx"], anchor["ty"], anchor["tz"],
                    )

                    # Apply alignment transform
                    if has_alignment:
                        position, wxyz = transform_pose(position, wxyz, scale, R_align, t_align)

                    pose_count += 1
                    inliers = anchor.get("num_inliers", 0)

                    print(f"Pose #{pose_count}: pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
                          f"inliers={inliers}, aligned={has_alignment}")

                    # Color by quality: green=good, yellow=meh, red=bad
                    if inliers >= 20:
                        color = (0, 255, 100)
                    elif inliers >= 8:
                        color = (255, 200, 0)
                    else:
                        color = (255, 80, 80)

                    server.scene.add_camera_frustum(
                        name="/camera/current",
                        fov=60.0,
                        aspect=16 / 9,
                        scale=gui_frustum_scale.value,
                        color=color,
                        wxyz=wxyz,
                        position=tuple(position),
                        line_width=3.0,
                    )

                    if gui_trail.value:
                        trail_positions.append(np.array(position, dtype=np.float32))
                        if len(trail_positions) >= 2:
                            trail_pts = np.array(trail_positions, dtype=np.float32)
                            trail_colors = np.full((len(trail_pts), 3), list(color), dtype=np.uint8)
                            server.scene.add_point_cloud(
                                name="/camera/trail",
                                points=trail_pts,
                                colors=trail_colors,
                                point_size=0.03,
                                point_shape="circle",
                            )

                    gui_status.value = (
                        f"Pose #{pose_count}: "
                        f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) "
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

    print(f"\nPose viewer running at http://localhost:{args.port}")
    print(f"Polling DPVO server at {args.dpvo_url}")
    print(f"Alignment active: {has_alignment}" + (f" (scale={scale:.4f})" if has_alignment else ""))
    print("Waiting for localized poses...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
