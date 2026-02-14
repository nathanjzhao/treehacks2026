"""
Viser-based 3D viewer for OpenFunGraph scene graphs.

Usage: python viewer.py data/              # browse all JSONs in directory
       python viewer.py data/scene.json    # open a specific file
"""

import argparse
import json
from pathlib import Path

import numpy as np
import uvicorn
import viser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from jinja2 import Template


def load_scene_graph(path: str) -> dict:
    """Load a serialized OpenFunGraph scene graph from JSON."""
    with open(path) as f:
        data = json.load(f)
    # Convert list arrays back to numpy
    for obj in data.get("objects", []):
        for key in ("pcd_np", "pcd_color_np", "clip_ft", "text_ft"):
            if key in obj and isinstance(obj[key], list):
                obj[key] = np.array(obj[key])
    for part in data.get("parts", []):
        for key in ("pcd_np", "pcd_color_np", "clip_ft", "text_ft"):
            if key in part and isinstance(part[key], list):
                part[key] = np.array(part[key])
    n_objects = len(data.get("objects", []))
    n_parts = len(data.get("parts", []))
    n_edges = len(data.get("edges", []))
    print(f"Loaded scene graph from {Path(path).name}: {n_objects} objects, {n_parts} parts, {n_edges} edges")
    return data


def generate_colors(n: int) -> list[np.ndarray]:
    """Generate n distinct colors for visualization."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        h = hue * 6.0
        c = 0.9 * 0.8
        x = c * (1 - abs(h % 2 - 1))
        m = 0.9 - c
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors.append(np.array([r + m, g + m, b + m]))
    return colors


def _estimate_up(cam_positions: list, scene_centroid: np.ndarray) -> np.ndarray:
    """Estimate up direction from camera positions using PCA."""
    if len(cam_positions) < 3:
        return np.array([0.0, -1.0, 0.0])

    positions = np.array(cam_positions)
    centered = positions - positions.mean(axis=0)
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    up = eigenvectors[:, 0]

    cam_mean = positions.mean(axis=0)
    if np.dot(up, cam_mean - scene_centroid) < 0:
        up = -up

    return up / (np.linalg.norm(up) + 1e-8)


# ---------- FastAPI app ----------
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
app = FastAPI()
VISER_PORT = 8081


@app.get("/", response_class=HTMLResponse)
async def index():
    template = Template(TEMPLATE_PATH.read_text())
    return template.render(viser_port=VISER_PORT)


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg = data.get("message", "").strip()
            if not msg:
                continue
            reply = f'I received your message: "{msg}". (LLM not connected yet.)'
            await websocket.send_json({"role": "assistant", "content": reply})
    except WebSocketDisconnect:
        pass


def _load_and_display(server: viser.ViserServer, pkl_path: str, point_size: float):
    """Load a scene graph PKL and display it in viser. Returns state dict."""
    scene_data = load_scene_graph(pkl_path)
    objects = scene_data.get("objects", [])
    parts = scene_data.get("parts", [])
    edges = scene_data.get("edges", [])

    # Clear existing scene
    server.scene.remove_by_name("/objects")
    server.scene.remove_by_name("/parts")
    server.scene.remove_by_name("/edges")
    server.scene.remove_by_name("/labels")

    all_points = []
    obj_colors = generate_colors(len(objects))

    # Add object point clouds
    for i, obj in enumerate(objects):
        pts = obj.get("pcd_np")
        if pts is None or len(pts) == 0:
            continue
        all_points.append(pts)

        obj_rgb = obj.get("pcd_color_np")
        if obj_rgb is not None and len(obj_rgb) == len(pts):
            if obj_rgb.max() <= 1.0:
                color_array = (obj_rgb * 255).astype(np.uint8)
            else:
                color_array = obj_rgb.astype(np.uint8)
        else:
            color_array = (np.tile(obj_colors[i], (len(pts), 1)) * 255).astype(np.uint8)

        label = obj.get("class_name", f"object_{i}")
        server.scene.add_point_cloud(
            name=f"/objects/{i}_{label}",
            points=pts.astype(np.float32),
            colors=color_array[:, :3],
            point_size=point_size,
            point_shape="rounded",
        )

        centroid = pts.mean(axis=0)
        refined = obj.get("refined_obj_tag", label)
        server.scene.add_label(
            name=f"/labels/obj_{i}",
            text=f"[{i}] {refined}",
            position=tuple(centroid),
        )

    # Add part point clouds
    part_colors = generate_colors(len(parts))
    for j, part in enumerate(parts):
        pts = part.get("pcd_np")
        if pts is None or len(pts) == 0:
            continue
        all_points.append(pts)

        part_rgb = part.get("pcd_color_np")
        if part_rgb is not None and len(part_rgb) == len(pts):
            if part_rgb.max() <= 1.0:
                color_array = (part_rgb * 255).astype(np.uint8)
            else:
                color_array = part_rgb.astype(np.uint8)
        else:
            color_array = (np.tile(part_colors[j], (len(pts), 1)) * 255).astype(np.uint8)

        label = part.get("class_name", f"part_{j}")
        server.scene.add_point_cloud(
            name=f"/parts/{j}_{label}",
            points=pts.astype(np.float32),
            colors=color_array[:, :3],
            point_size=point_size * 0.7,
            point_shape="rounded",
        )

    # Add edges
    for k, edge in enumerate(edges):
        obj_idx, part_idx, target_idx, description, confidence = edge

        if 0 <= obj_idx < len(objects):
            src_pts = objects[obj_idx].get("pcd_np")
            if src_pts is not None and len(src_pts) > 0:
                src_pos = src_pts.mean(axis=0)
            else:
                continue
        else:
            continue

        if part_idx >= 0 and part_idx < len(parts):
            tgt_pts = parts[part_idx].get("pcd_np")
            if tgt_pts is not None and len(tgt_pts) > 0:
                tgt_pos = tgt_pts.mean(axis=0)
            else:
                continue
        elif target_idx >= 0 and target_idx < len(objects):
            tgt_pts = objects[target_idx].get("pcd_np")
            if tgt_pts is not None and len(tgt_pts) > 0:
                tgt_pos = tgt_pts.mean(axis=0)
            else:
                continue
        else:
            continue

        points_line = np.stack([src_pos, tgt_pos]).astype(np.float32)
        r = int((1 - confidence) * 255)
        g = int(confidence * 255)
        color_line = np.array([[r, g, 50], [r, g, 50]], dtype=np.uint8)

        server.scene.add_point_cloud(
            name=f"/edges/{k}_endpoints",
            points=points_line,
            colors=color_line,
            point_size=point_size * 2,
            point_shape="rounded",
        )

        midpoint = (src_pos + tgt_pos) / 2
        server.scene.add_label(
            name=f"/labels/edge_{k}",
            text=f"{description} ({confidence:.2f})",
            position=tuple(midpoint),
        )

    # Compute scene centroid and up direction
    if all_points:
        all_pts = np.concatenate(all_points, axis=0)
        centroid = all_pts.mean(axis=0)
    else:
        centroid = np.zeros(3)

    # Camera poses (cam-to-world 4x4 matrices)
    cam_poses_raw = scene_data.get("camera_poses", [])
    cam_poses = [np.array(p) for p in cam_poses_raw] if cam_poses_raw else []
    cam_positions = [p[:3, 3] for p in cam_poses] if cam_poses else []

    up = _estimate_up(cam_positions, centroid)

    # Draw camera frustums
    server.scene.remove_by_name("/cameras")
    frustum_size = 0.08
    for ci, pose in enumerate(cam_poses):
        pos = pose[:3, 3]
        R = pose[:3, :3]
        corners_cam = np.array([
            [-1, -0.75, -1],
            [ 1, -0.75, -1],
            [ 1,  0.75, -1],
            [-1,  0.75, -1],
        ]) * frustum_size
        corners_world = (R @ corners_cam.T).T + pos
        apex = pos.astype(np.float32)
        lines_pts = []
        for c in corners_world:
            lines_pts.extend([apex, c.astype(np.float32)])
        for idx in range(4):
            lines_pts.extend([
                corners_world[idx].astype(np.float32),
                corners_world[(idx + 1) % 4].astype(np.float32),
            ])
        lines_arr = np.array(lines_pts, dtype=np.float32)
        color = [255, 220, 50] if ci == 0 else [100, 180, 255]
        colors_arr = np.tile(np.array([color], dtype=np.uint8), (len(lines_arr), 1))
        server.scene.add_point_cloud(
            name=f"/cameras/frustum_{ci}",
            points=lines_arr,
            colors=colors_arr,
            point_size=point_size * 0.4,
            point_shape="circle",
        )

    # Set initial camera to first original camera POV
    if cam_poses:
        pose0 = cam_poses[0]
        init_cam_pos = pose0[:3, 3]
        R0 = pose0[:3, :3]
        look_at = init_cam_pos + R0 @ np.array([0, 0, -1])
        up = R0 @ np.array([0, -1, 0])
    elif all_points:
        bbox_extent = all_pts.max(axis=0) - all_pts.min(axis=0)
        cam_distance = float(np.linalg.norm(bbox_extent)) * 0.8
        init_cam_pos = centroid + up * cam_distance
        look_at = centroid
    else:
        init_cam_pos = np.array([0, -2, 1])
        look_at = centroid

    # Move all connected clients
    for client in server.get_clients().values():
        client.camera.position = tuple(init_cam_pos)
        client.camera.look_at = tuple(look_at)
        client.camera.up_direction = tuple(up)

    return {
        "scene_data": scene_data,
        "centroid": centroid,
        "init_cam_pos": init_cam_pos,
        "look_at": look_at,
        "up": up,
        "objects": objects,
        "obj_colors": obj_colors,
        "cam_poses": cam_poses,
    }


# ---------- main ----------
def main():
    global VISER_PORT

    parser = argparse.ArgumentParser(description="View OpenFunGraph scene graphs")
    parser.add_argument("scene_graph", help="Path to .json file or directory of .json files")
    parser.add_argument("--port", type=int, default=8080, help="Main web UI port")
    parser.add_argument("--viser-port", type=int, default=8081, help="Internal viser port")
    parser.add_argument("--point-size", type=float, default=0.02, help="Point size")
    args = parser.parse_args()
    VISER_PORT = args.viser_port

    # Discover all JSON scene graph files
    sg_input = Path(args.scene_graph)
    if sg_input.is_dir():
        sg_files = sorted(sg_input.glob("*.json"))
    else:
        sg_files = [sg_input]
        siblings = sorted(sg_input.parent.glob("*.json"))
        if len(siblings) > 1:
            sg_files = siblings

    if not sg_files:
        raise ValueError(f"No .json files found at {args.scene_graph}")

    sg_names = [f.stem for f in sg_files]
    sg_map = {f.stem: str(f) for f in sg_files}
    initial_name = sg_input.stem if sg_input.is_file() else sg_names[0]

    print(f"Found {len(sg_files)} scene graph files: {', '.join(sg_names)}")

    # --- Start viser ---
    server = viser.ViserServer(host="0.0.0.0", port=args.viser_port)

    state = _load_and_display(server, sg_map[initial_name], args.point_size)

    with server.gui.add_folder("Viewer"):
        if len(sg_files) > 1:
            gui_file = server.gui.add_dropdown(
                "Scene",
                options=sg_names,
                initial_value=initial_name,
            )

            @gui_file.on_update
            def _on_file_change(_):
                name = gui_file.value
                print(f"\nSwitching to {name}...")
                new_state = _load_and_display(server, sg_map[name], gui_point_size.value)
                state.update(new_state)

        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size
        )

        @gui_point_size.on_update
        def _on_size_change(_):
            objects = state.get("objects", [])
            obj_colors = state.get("obj_colors", [])
            ps = gui_point_size.value
            for i, obj in enumerate(objects):
                pts = obj.get("pcd_np")
                if pts is None or len(pts) == 0:
                    continue
                obj_rgb = obj.get("pcd_color_np")
                if obj_rgb is not None and len(obj_rgb) == len(pts):
                    ca = (obj_rgb * 255).astype(np.uint8) if obj_rgb.max() <= 1.0 else obj_rgb.astype(np.uint8)
                else:
                    ca = (np.tile(obj_colors[i], (len(pts), 1)) * 255).astype(np.uint8)
                label = obj.get("class_name", f"object_{i}")
                server.scene.add_point_cloud(
                    name=f"/objects/{i}_{label}",
                    points=pts.astype(np.float32),
                    colors=ca[:, :3],
                    point_size=ps,
                    point_shape="rounded",
                )

    with server.gui.add_folder("Camera"):
        n_cams = len(state.get("cam_poses", []))
        if n_cams > 0:
            gui_cam_idx = server.gui.add_slider(
                "Frame", min=0, max=n_cams - 1, step=1, initial_value=0
            )

            @gui_cam_idx.on_update
            def _on_cam_change(_):
                cam_poses = state.get("cam_poses", [])
                idx = int(gui_cam_idx.value)
                if 0 <= idx < len(cam_poses):
                    pose = cam_poses[idx]
                    pos = pose[:3, 3]
                    R = pose[:3, :3]
                    look_at = pos + R @ np.array([0, 0, -1])
                    up_dir = R @ np.array([0, -1, 0])
                    for client in server.get_clients().values():
                        client.camera.position = tuple(pos)
                        client.camera.look_at = tuple(look_at)
                        client.camera.up_direction = tuple(up_dir)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = tuple(state["init_cam_pos"])
        client.camera.look_at = tuple(state["look_at"])
        client.camera.up_direction = tuple(state["up"])

    print(f"\nViser running on http://localhost:{args.viser_port}")
    print(f"Web UI at http://localhost:{args.port}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
