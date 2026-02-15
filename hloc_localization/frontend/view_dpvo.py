"""
Standalone DPVO result viewer with trajectory playback.

Usage:
  python3 hloc_localization/frontend/view_dpvo.py \
    data/mapanything/IMG_4720.glb \
    hloc_localization/data/dpvo_results/IMG_4730_dpvo.json
"""
import argparse
import json
import threading
import time
import numpy as np
import trimesh
import viser
from pathlib import Path


def qvec_to_R(qw, qx, qy, qz):
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy],
    ])


def mat_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s=0.5/np.sqrt(tr+1); w=0.25/s; x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return np.array([w,x,y,z])


def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1, 1)
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def main():
    parser = argparse.ArgumentParser(description="DPVO Result Viewer")
    parser.add_argument("glb", help="Path to .glb point cloud")
    parser.add_argument("result", help="Path to DPVO result JSON")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--frustum-scale", type=float, default=0.3)
    args = parser.parse_args()

    # Load point cloud
    scene = trimesh.load(args.glb)
    points = colors = None
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

    idx = np.arange(0, len(points), args.downsample)
    points = points[idx]
    colors = colors[idx]
    print(f"Point cloud: {len(points):,} points (1/{args.downsample})")

    # Load poses
    result = json.loads(Path(args.result).read_text())
    poses = result["poses"]
    n_poses = len(poses)
    print(f"Poses: {n_poses}, alignment scale: {result.get('alignment_scale', '?')}")

    # Precompute camera positions + orientations (camera-to-world)
    cam_positions = []
    cam_wxyz = []
    cam_look_dirs = []  # forward direction for each camera
    for p in poses:
        R = qvec_to_R(p["qw"], p["qx"], p["qy"], p["qz"])
        R_c2w = R.T
        pos = -R_c2w @ np.array([p["tx"], p["ty"], p["tz"]])
        # COLMAP: camera looks along +Z in camera frame
        forward = R_c2w @ np.array([0.0, 0.0, 1.0])
        cam_positions.append(pos)
        cam_wxyz.append(mat_to_quat(R_c2w))
        cam_look_dirs.append(forward)
    cam_positions = np.array(cam_positions)
    cam_look_dirs = np.array(cam_look_dirs)

    print(f"Cam range:  {cam_positions.min(axis=0)} -> {cam_positions.max(axis=0)}")
    print(f"PC range:   {points.min(axis=0)} -> {points.max(axis=0)}")

    # Viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    server.scene.add_point_cloud(
        name="/pc", points=points, colors=colors,
        point_size=0.005, point_shape="rounded",
    )

    # Draw all frustums
    for i, p in enumerate(poses):
        is_hloc = p.get("source") == "hloc"
        color = (0, 255, 100) if is_hloc else (255, 100, 50)
        s = args.frustum_scale * (1.5 if is_hloc else 1.0)
        server.scene.add_camera_frustum(
            name=f"/cams/{i:03d}",
            fov=np.deg2rad(60), aspect=16/9,
            scale=s, line_width=4.0,
            wxyz=cam_wxyz[i],
            position=tuple(cam_positions[i]),
            color=color,
        )

    # Trajectory dots
    traj_colors = np.full((len(cam_positions), 3), [255, 180, 50], dtype=np.uint8)
    for i, p in enumerate(poses):
        if p.get("source") == "hloc":
            traj_colors[i] = [0, 255, 100]
    server.scene.add_point_cloud(
        name="/traj", points=cam_positions, colors=traj_colors,
        point_size=0.025, point_shape="circle",
    )

    # "Current" camera marker (bright cyan, bigger)
    server.scene.add_camera_frustum(
        name="/current_cam",
        fov=np.deg2rad(60), aspect=16/9,
        scale=args.frustum_scale * 2.0, line_width=6.0,
        wxyz=cam_wxyz[0],
        position=tuple(cam_positions[0]),
        color=(0, 255, 255),
    )

    # --- GUI: Playback controls ---
    playback_state = {"playing": False, "frame": 0, "speed": 1.0, "follow": True}

    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider(
            "Frame", min=0, max=n_poses - 1, step=1, initial_value=0
        )
        gui_play = server.gui.add_button("Play")
        gui_stop = server.gui.add_button("Stop")
        gui_speed = server.gui.add_slider(
            "Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0
        )
        gui_follow = server.gui.add_checkbox("Follow camera", initial_value=True)
        gui_info = server.gui.add_markdown("Frame 0 / " + str(n_poses - 1) + " (hloc)")

    with server.gui.add_folder("Display"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.02, step=0.001, initial_value=0.005
        )
        gui_show_frustums = server.gui.add_checkbox("Show all frustums", initial_value=True)
        gui_show_traj = server.gui.add_checkbox("Show trajectory", initial_value=True)

    def move_to_frame(frame_idx):
        """Update the current camera marker and optionally move the viewer."""
        frame_idx = int(np.clip(frame_idx, 0, n_poses - 1))
        pos = cam_positions[frame_idx]
        wxyz = cam_wxyz[frame_idx]
        forward = cam_look_dirs[frame_idx]
        source = poses[frame_idx].get("source", "dpvo")

        # Update current camera marker
        server.scene.add_camera_frustum(
            name="/current_cam",
            fov=np.deg2rad(60), aspect=16/9,
            scale=args.frustum_scale * 2.0, line_width=6.0,
            wxyz=wxyz,
            position=tuple(pos),
            color=(0, 255, 255),
        )

        # Update info
        gui_info.content = f"Frame {frame_idx} / {n_poses - 1} ({source})"

        # Move viewer camera to follow along
        if playback_state["follow"]:
            R = qvec_to_R(poses[frame_idx]["qw"], poses[frame_idx]["qx"],
                          poses[frame_idx]["qy"], poses[frame_idx]["qz"])
            R_c2w = R.T
            # COLMAP: Y is down in camera frame, so world-up = R_c2w @ [0, -1, 0]
            cam_up = R_c2w @ np.array([0.0, -1.0, 0.0])

            # Viewer sits slightly behind and above
            viewer_pos = pos - forward * 0.5 + cam_up * 0.3
            look_at = pos + forward * 1.0

            for client in server.get_clients().values():
                client.camera.position = tuple(viewer_pos)
                client.camera.look_at = tuple(look_at)
                client.camera.up_direction = tuple(cam_up)

    @gui_frame.on_update
    def _(_):
        playback_state["frame"] = gui_frame.value
        move_to_frame(gui_frame.value)

    @gui_follow.on_update
    def _(_):
        playback_state["follow"] = gui_follow.value

    @gui_play.on_click
    def _(_):
        playback_state["playing"] = True

    @gui_stop.on_click
    def _(_):
        playback_state["playing"] = False

    @gui_speed.on_update
    def _(_):
        playback_state["speed"] = gui_speed.value

    @gui_point_size.on_update
    def _(_):
        server.scene.add_point_cloud(
            name="/pc", points=points, colors=colors,
            point_size=gui_point_size.value, point_shape="rounded",
        )

    @gui_show_frustums.on_update
    def _(_):
        for i in range(n_poses):
            server.scene.set_visibility(f"/cams/{i:03d}", gui_show_frustums.value)

    @gui_show_traj.on_update
    def _(_):
        server.scene.set_visibility("/traj", gui_show_traj.value)

    # Initial view: overview
    all_pts = np.vstack([points[::100], cam_positions])
    center = all_pts.mean(axis=0)
    extent = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))

    @server.on_client_connect
    def _(client):
        client.camera.position = tuple(center + np.array([0, -extent * 0.8, extent * 0.5]))
        client.camera.look_at = tuple(center)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    print(f"\nhttp://localhost:{args.port}")
    print(f"  Green = HLoc anchors, Orange = DPVO, Cyan = current frame")
    print(f"  Use slider or Play/Stop to scrub through trajectory")

    # Playback loop
    def playback_loop():
        while True:
            if playback_state["playing"]:
                f = playback_state["frame"]
                f += playback_state["speed"]
                if f >= n_poses:
                    f = 0  # loop
                playback_state["frame"] = f
                gui_frame.value = int(f)
                move_to_frame(int(f))
            time.sleep(0.1)

    thread = threading.Thread(target=playback_loop, daemon=True)
    thread.start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
