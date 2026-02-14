"""
View localized camera trajectory in 3D point cloud using viser.

Loads a GLB scene and overlays camera frustums from localization results.
Animates through poses with play/pause controls.

Usage:
  python -m hloc_localization.frontend.view_trajectory
"""

import time
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
    # cam_from_world: R @ world_pt + t = cam_pt
    # camera center in world: -R^T @ t
    cam_pos = -R.T @ t
    # camera-to-world transform
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = cam_pos
    return cam_pos, T


def main():
    import pathlib

    glb_path = pathlib.Path(__file__).parent.parent.parent / "data" / "mapanything" / "IMG_4720.glb"

    print(f"Loading GLB: {glb_path.name} ({glb_path.stat().st_size / 1024 / 1024:.0f} MB)")
    scene = trimesh.load(str(glb_path))

    # Extract points and colors from GLB
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

    if points:
        all_points = np.concatenate(points)
        all_colors = np.concatenate(colors)
        # Subsample if too many points
        if len(all_points) > 500_000:
            idx = np.random.choice(len(all_points), 500_000, replace=False)
            all_points = all_points[idx]
            all_colors = all_colors[idx]
        print(f"Point cloud: {len(all_points):,} points")
    else:
        all_points = np.zeros((0, 3))
        all_colors = np.zeros((0, 3))
        print("Warning: no points extracted from GLB")

    # Compute camera world positions
    cam_positions = []
    cam_transforms = []
    for p in POSES:
        pos, T = pose_to_world(p)
        cam_positions.append(pos)
        cam_transforms.append(T)
    cam_positions = np.array(cam_positions)

    # Start viser server
    server = viser.ViserServer(host="0.0.0.0", port=8890)
    print("Viser server at http://localhost:8890")

    # Add point cloud
    if len(all_points) > 0:
        server.scene.add_point_cloud(
            "/scene/pointcloud",
            points=all_points.astype(np.float32),
            colors=all_colors.astype(np.float32),
            point_size=0.005,
            point_shape="rounded",
        )

    # Add trajectory line
    server.scene.add_point_cloud(
        "/scene/trajectory",
        points=cam_positions.astype(np.float32),
        colors=np.array([[1.0, 0.5, 0.0]] * len(cam_positions), dtype=np.float32),
        point_size=0.03,
        point_shape="circle",
    )

    # Add all camera frustums (dimmed)
    for i, (p, T) in enumerate(zip(POSES, cam_transforms)):
        wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
        pos = T[:3, 3]
        color_val = int(255 * p["inliers"] / max(x["inliers"] for x in POSES))
        server.scene.add_camera_frustum(
            f"/scene/cameras/frame_{i:02d}",
            fov=60.0,
            aspect=16/9,
            scale=0.15,
            wxyz=wxyz,
            position=pos,
            color=(100, 100, color_val),
        )

    # Active camera frustum (bright)
    active_frustum = server.scene.add_camera_frustum(
        "/scene/active_camera",
        fov=60.0,
        aspect=16/9,
        scale=0.25,
        wxyz=(1, 0, 0, 0),
        position=(0, 0, 0),
        color=(0, 255, 0),
    )

    # GUI controls
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(POSES)-1, step=1, initial_value=0)
        play_btn = server.gui.add_button("Play")
        speed_slider = server.gui.add_slider("Speed (s/frame)", min=0.1, max=2.0, step=0.1, initial_value=0.5)

    with server.gui.add_folder("Info"):
        info_text = server.gui.add_markdown("**Frame 0** | Inliers: 8380")

    playing = [False]

    @frame_slider.on_update
    def _on_frame(event):
        i = frame_slider.value
        T = cam_transforms[i]
        wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
        pos = T[:3, 3]
        active_frustum.wxyz = wxyz
        active_frustum.position = pos
        info_text.content = f"**Frame {i}** | Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | Inliers: {POSES[i]['inliers']}"

    @play_btn.on_click
    def _on_play(event):
        playing[0] = not playing[0]
        play_btn.name = "Pause" if playing[0] else "Play"

    # Trigger initial frame
    _on_frame(None)

    print("Ready! Open http://localhost:8890 in your browser.")
    print("Use the slider to step through frames, or click Play to animate.")

    while True:
        if playing[0]:
            i = (frame_slider.value + 1) % len(POSES)
            frame_slider.value = i
            time.sleep(speed_slider.value)
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
