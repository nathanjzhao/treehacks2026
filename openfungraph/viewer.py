"""
Viser-based 3D viewer for OpenFunGraph scene graphs.

Usage: python viewer.py output.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import uvicorn
import viser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from jinja2 import Template


def load_scene_graph(path: str) -> dict:
    """Load a serialized OpenFunGraph scene graph from pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    n_objects = len(data.get("objects", []))
    n_parts = len(data.get("parts", []))
    n_edges = len(data.get("edges", []))
    print(f"Loaded scene graph: {n_objects} objects, {n_parts} parts, {n_edges} edges")
    return data


def generate_colors(n: int) -> list[np.ndarray]:
    """Generate n distinct colors for visualization."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        # HSV to RGB (simple conversion, S=0.8, V=0.9)
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


# ---------- main ----------
def main():
    global VISER_PORT

    parser = argparse.ArgumentParser(description="View OpenFunGraph scene graphs")
    parser.add_argument("scene_graph", help="Path to .pkl scene graph file")
    parser.add_argument("--port", type=int, default=8080, help="Main web UI port")
    parser.add_argument("--viser-port", type=int, default=8081, help="Internal viser port")
    parser.add_argument("--point-size", type=float, default=0.005, help="Point size")
    args = parser.parse_args()
    VISER_PORT = args.viser_port

    scene_data = load_scene_graph(args.scene_graph)
    objects = scene_data.get("objects", [])
    parts = scene_data.get("parts", [])
    edges = scene_data.get("edges", [])

    # --- Start viser ---
    server = viser.ViserServer(host="0.0.0.0", port=args.viser_port)

    # Collect all points for centroid computation
    all_points = []

    # Generate colors for objects
    obj_colors = generate_colors(len(objects))

    # Add object point clouds
    for i, obj in enumerate(objects):
        pts = obj.get("pcd_np")
        if pts is None or len(pts) == 0:
            continue
        all_points.append(pts)

        # Use object's own colors if available, otherwise use instance color
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
            point_size=args.point_size,
            point_shape="rounded",
        )

        # Add text label at centroid
        centroid = pts.mean(axis=0)
        refined = obj.get("refined_obj_tag", label)
        server.scene.add_label(
            name=f"/labels/obj_{i}",
            text=f"[{i}] {refined}",
            position=tuple(centroid),
        )

    # Generate colors for parts (use different hue range)
    part_colors = generate_colors(len(parts))

    # Add part point clouds
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
            point_size=args.point_size * 0.7,
            point_shape="rounded",
        )

    # Add edges as lines between connected objects/parts
    for k, edge in enumerate(edges):
        # Edge format: (obj_idx, part_idx_or_-1, target_idx_or_-1, description, confidence)
        obj_idx, part_idx, target_idx, description, confidence = edge

        # Determine source position
        if 0 <= obj_idx < len(objects):
            src_pts = objects[obj_idx].get("pcd_np")
            if src_pts is not None and len(src_pts) > 0:
                src_pos = src_pts.mean(axis=0)
            else:
                continue
        else:
            continue

        # Determine target position
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

        # Draw line between source and target
        points_line = np.stack([src_pos, tgt_pos]).astype(np.float32)
        # Color by confidence (green = high, red = low)
        r = int((1 - confidence) * 255)
        g = int(confidence * 255)
        color_line = np.array([[r, g, 50], [r, g, 50]], dtype=np.uint8)

        server.scene.add_point_cloud(
            name=f"/edges/{k}_endpoints",
            points=points_line,
            colors=color_line,
            point_size=args.point_size * 2,
            point_shape="rounded",
        )

        # Add edge label at midpoint
        midpoint = (src_pos + tgt_pos) / 2
        server.scene.add_label(
            name=f"/labels/edge_{k}",
            text=f"{description} ({confidence:.2f})",
            position=tuple(midpoint),
        )

    # Compute scene centroid and camera position
    if all_points:
        all_pts = np.concatenate(all_points, axis=0)
        centroid = all_pts.mean(axis=0)
    else:
        centroid = np.zeros(3)

    # Use first VGGT camera position if available (matches VGGT viewer)
    cam_positions = scene_data.get("camera_positions", [])
    if cam_positions:
        first_cam_pos = np.array(cam_positions[0])
        look_dir = centroid - first_cam_pos
        look_dir /= np.linalg.norm(look_dir) + 1e-8
        init_cam_pos = first_cam_pos - look_dir * 0.3
    elif all_points:
        bbox_extent = all_pts.max(axis=0) - all_pts.min(axis=0)
        cam_distance = float(np.linalg.norm(bbox_extent)) * 0.8
        init_cam_pos = centroid + np.array([0, -cam_distance, cam_distance * 0.5])
    else:
        init_cam_pos = np.array([0, -2, 1])

    # GUI controls
    with server.gui.add_folder("Viewer"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.05, step=0.001, initial_value=args.point_size
        )

        @gui_point_size.on_update
        def _(_):
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

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = tuple(init_cam_pos)
        client.camera.look_at = tuple(centroid)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    print(f"\nViser running on http://localhost:{args.viser_port} (internal)")
    print(f"Web UI at http://localhost:{args.port}")
    print(f"Scene center: {centroid}, init camera: {init_cam_pos}")

    # --- Run FastAPI on main thread ---
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
