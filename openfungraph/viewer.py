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

# Dim color for non-selected elements
DIM_GRAY = np.array([60, 60, 60], dtype=np.uint8)


def load_scene_graph(path: str) -> dict:
    """Load a serialized OpenFunGraph scene graph from JSON."""
    with open(path) as f:
        data = json.load(f)
    for obj in data.get("objects", []):
        for key in ("pcd_np", "pcd_color_np", "clip_ft", "text_ft"):
            if key in obj and isinstance(obj[key], list):
                obj[key] = np.array(obj[key])
    for part in data.get("parts", []):
        for key in ("pcd_np", "pcd_color_np", "clip_ft", "text_ft"):
            if key in part and isinstance(part[key], list):
                part[key] = np.array(part[key])
    n_obj = len(data.get("objects", []))
    n_parts = len(data.get("parts", []))
    n_edges = len(data.get("edges", []))
    print(f"Loaded {Path(path).name}: {n_obj} objects, {n_parts} parts, {n_edges} edges")
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


def _get_colors(item: dict, fallback_color: np.ndarray, n_pts: int) -> np.ndarray:
    """Get uint8 RGB color array for an object/part."""
    rgb = item.get("pcd_color_np")
    if rgb is not None and len(rgb) == n_pts:
        if rgb.max() <= 1.0:
            return (rgb * 255).astype(np.uint8)[:, :3]
        return rgb.astype(np.uint8)[:, :3]
    return (np.tile(fallback_color, (n_pts, 1)) * 255).astype(np.uint8)


def _estimate_up(cam_positions: list, scene_centroid: np.ndarray) -> np.ndarray:
    """Estimate up direction from camera positions using PCA."""
    if len(cam_positions) < 3:
        return np.array([0.0, -1.0, 0.0])
    positions = np.array(cam_positions)
    centered = positions - positions.mean(axis=0)
    cov = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(cov)
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


# ---------- Scene display ----------

def _load_and_display(server: viser.ViserServer, sg_path: str, point_size: float):
    """Load a scene graph and display it. Returns state dict with handles."""
    scene_data = load_scene_graph(sg_path)
    objects = scene_data.get("objects", [])
    parts = scene_data.get("parts", [])
    edges = scene_data.get("edges", [])

    # Clear existing scene
    for group in ("/objects", "/parts", "/edges", "/labels", "/cameras"):
        server.scene.remove_by_name(group)

    all_points = []
    obj_colors = generate_colors(len(objects))
    part_colors = generate_colors(len(parts))

    # Track handles and names for highlight logic
    obj_handles = {}   # i -> handle
    obj_names = {}     # i -> display name
    part_handles = {}  # j -> handle
    part_names = {}    # j -> display name
    obj_label_handles = {}
    edge_handles = {}
    edge_label_handles = {}

    # --- Objects ---
    for i, obj in enumerate(objects):
        pts = obj.get("pcd_np")
        if pts is None or len(pts) == 0:
            continue
        all_points.append(pts)
        label = obj.get("class_name", f"object_{i}")
        obj_names[i] = f"[{i}] {label}"
        color_array = _get_colors(obj, obj_colors[i], len(pts))
        node_name = f"/objects/obj_{i}"
        obj_handles[i] = server.scene.add_point_cloud(
            name=node_name,
            points=pts.astype(np.float32),
            colors=color_array,
            point_size=point_size,
            point_shape="rounded",
        )
        centroid = pts.mean(axis=0)
        refined = obj.get("refined_obj_tag", label)
        obj_label_handles[i] = server.scene.add_label(
            name=f"/labels/obj_{i}",
            text=f"[{i}] {refined}",
            position=tuple(centroid),
        )

    # --- Parts ---
    for j, part in enumerate(parts):
        pts = part.get("pcd_np")
        if pts is None or len(pts) == 0:
            continue
        all_points.append(pts)
        label = part.get("class_name", f"part_{j}")
        part_names[j] = f"[{j}] {label}"
        color_array = _get_colors(part, part_colors[j], len(pts))
        node_name = f"/parts/part_{j}"
        part_handles[j] = server.scene.add_point_cloud(
            name=node_name,
            points=pts.astype(np.float32),
            colors=color_array,
            point_size=point_size * 0.7,
            point_shape="rounded",
        )

    # --- Edges ---
    for k, edge in enumerate(edges):
        obj_idx, part_idx, target_idx, description, confidence = edge
        src_pts = objects[obj_idx].get("pcd_np") if 0 <= obj_idx < len(objects) else None
        if src_pts is None or len(src_pts) == 0:
            continue
        src_pos = src_pts.mean(axis=0)

        tgt_pos = None
        if 0 <= part_idx < len(parts):
            tgt_pts = parts[part_idx].get("pcd_np")
            if tgt_pts is not None and len(tgt_pts) > 0:
                tgt_pos = tgt_pts.mean(axis=0)
        if tgt_pos is None and 0 <= target_idx < len(objects):
            tgt_pts = objects[target_idx].get("pcd_np")
            if tgt_pts is not None and len(tgt_pts) > 0:
                tgt_pos = tgt_pts.mean(axis=0)
        if tgt_pos is None:
            continue

        points_line = np.stack([src_pos, tgt_pos]).astype(np.float32)
        r = int((1 - confidence) * 255)
        g = int(confidence * 255)
        color_line = np.array([[r, g, 50], [r, g, 50]], dtype=np.uint8)
        edge_handles[k] = server.scene.add_point_cloud(
            name=f"/edges/edge_{k}",
            points=points_line,
            colors=color_line,
            point_size=point_size * 2,
            point_shape="rounded",
        )
        midpoint = (src_pos + tgt_pos) / 2
        edge_label_handles[k] = server.scene.add_label(
            name=f"/labels/edge_{k}",
            text=f"{description} ({confidence:.2f})",
            position=tuple(midpoint),
        )

    # --- Camera frustums ---
    cam_poses_raw = scene_data.get("camera_poses", [])
    cam_poses = [np.array(p) for p in cam_poses_raw] if cam_poses_raw else []
    cam_positions = [p[:3, 3] for p in cam_poses] if cam_poses else []

    # Compute scene centroid and up
    if all_points:
        all_pts = np.concatenate(all_points, axis=0)
        centroid = all_pts.mean(axis=0)
    else:
        centroid = np.zeros(3)

    up = _estimate_up(cam_positions, centroid)

    frustum_size = 0.08
    cam_handles = {}
    for ci, pose in enumerate(cam_poses):
        pos = pose[:3, 3]
        R = pose[:3, :3]
        corners_cam = np.array([
            [-1, -0.75, -1], [1, -0.75, -1],
            [1, 0.75, -1], [-1, 0.75, -1],
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
        cam_handles[ci] = server.scene.add_point_cloud(
            name=f"/cameras/frustum_{ci}",
            points=lines_arr,
            colors=colors_arr,
            point_size=point_size * 0.4,
            point_shape="circle",
        )

    # Initial camera
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
        "parts": parts,
        "edges": edges,
        "obj_colors": obj_colors,
        "part_colors": part_colors,
        "obj_handles": obj_handles,
        "obj_names": obj_names,
        "part_handles": part_handles,
        "part_names": part_names,
        "obj_label_handles": obj_label_handles,
        "edge_handles": edge_handles,
        "edge_label_handles": edge_label_handles,
        "cam_handles": cam_handles,
        "cam_poses": cam_poses,
        "point_size": point_size,
    }


def _highlight_selection(server, state, selected_type, selected_idx):
    """
    Highlight a selected object or part. Dims everything else.
    selected_type: "none", "object", or "part"
    selected_idx: index into objects or parts list
    """
    objects = state["objects"]
    parts = state["parts"]
    obj_colors = state["obj_colors"]
    part_colors = state["part_colors"]
    ps = state["point_size"]
    show_all = selected_type == "none"

    # Update objects
    for i, obj in enumerate(objects):
        pts = obj.get("pcd_np")
        if pts is None or len(pts) == 0:
            continue
        is_selected = show_all or (selected_type == "object" and selected_idx == i)
        if is_selected:
            colors = _get_colors(obj, obj_colors[i], len(pts))
            size = ps * (1.5 if not show_all else 1.0)
        else:
            colors = np.tile(DIM_GRAY, (len(pts), 1))
            size = ps * 0.5
        server.scene.add_point_cloud(
            name=f"/objects/obj_{i}",
            points=pts.astype(np.float32),
            colors=colors,
            point_size=size,
            point_shape="rounded",
        )
        # Show/hide label
        if i in state["obj_label_handles"]:
            state["obj_label_handles"][i].visible = is_selected or show_all

    # Update parts
    for j, part in enumerate(parts):
        pts = part.get("pcd_np")
        if pts is None or len(pts) == 0:
            continue
        is_selected = show_all or (selected_type == "part" and selected_idx == j)
        # Also highlight parts connected to selected object via edges
        if not is_selected and selected_type == "object":
            for edge in state["edges"]:
                obj_idx, part_idx, target_idx, _, _ = edge
                if obj_idx == selected_idx and part_idx == j:
                    is_selected = True
                    break
        if is_selected:
            colors = _get_colors(part, part_colors[j], len(pts))
            size = ps * (1.2 if not show_all else 0.7)
        else:
            colors = np.tile(DIM_GRAY, (len(pts), 1))
            size = ps * 0.3
        server.scene.add_point_cloud(
            name=f"/parts/part_{j}",
            points=pts.astype(np.float32),
            colors=colors,
            point_size=size,
            point_shape="rounded",
        )

    # Update edges — show only relevant ones
    for k, edge in enumerate(state["edges"]):
        obj_idx, part_idx, target_idx, _, _ = edge
        is_relevant = show_all
        if selected_type == "object" and obj_idx == selected_idx:
            is_relevant = True
        if selected_type == "part" and part_idx == selected_idx:
            is_relevant = True
        if k in state["edge_handles"]:
            state["edge_handles"][k].visible = is_relevant
        if k in state["edge_label_handles"]:
            state["edge_label_handles"][k].visible = is_relevant


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

    # Discover JSON files
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

    # === GUI: Scene selector ===
    with server.gui.add_folder("Scene"):
        if len(sg_files) > 1:
            gui_file = server.gui.add_dropdown(
                "File", options=sg_names, initial_value=initial_name,
            )

            @gui_file.on_update
            def _on_file_change(_):
                name = gui_file.value
                print(f"\nSwitching to {name}...")
                new_state = _load_and_display(server, sg_map[name], args.point_size)
                state.update(new_state)
                # Rebuild selection dropdown options
                _rebuild_selection_options()

        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size,
        )

        @gui_point_size.on_update
        def _on_size_change(_):
            state["point_size"] = gui_point_size.value
            # Re-apply current highlight state
            _apply_current_highlight()

        gui_show_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)

        @gui_show_cameras.on_update
        def _on_toggle_cameras(_):
            for h in state.get("cam_handles", {}).values():
                h.visible = gui_show_cameras.value

    # === GUI: Selection / Highlight ===
    def _build_options():
        """Build dropdown options list from current state."""
        opts = ["(none)"]
        for i in sorted(state.get("obj_names", {}).keys()):
            opts.append(f"obj: {state['obj_names'][i]}")
        for j in sorted(state.get("part_names", {}).keys()):
            opts.append(f"part: {state['part_names'][j]}")
        return opts

    with server.gui.add_folder("Highlight"):
        gui_select = server.gui.add_dropdown(
            "Select", options=_build_options(), initial_value="(none)",
        )

        def _rebuild_selection_options():
            gui_select.options = _build_options()
            gui_select.value = "(none)"

        def _apply_current_highlight():
            val = gui_select.value
            if val == "(none)":
                _highlight_selection(server, state, "none", -1)
            elif val.startswith("obj: "):
                # Parse index from "[i] label"
                bracket = val.find("[")
                close = val.find("]")
                if bracket >= 0 and close > bracket:
                    idx = int(val[bracket + 1:close])
                    _highlight_selection(server, state, "object", idx)
            elif val.startswith("part: "):
                bracket = val.find("[")
                close = val.find("]")
                if bracket >= 0 and close > bracket:
                    idx = int(val[bracket + 1:close])
                    _highlight_selection(server, state, "part", idx)

        @gui_select.on_update
        def _on_select_change(_):
            _apply_current_highlight()

        gui_reset = server.gui.add_button("Reset highlight")

        @gui_reset.on_click
        def _on_reset(_):
            gui_select.value = "(none)"
            _highlight_selection(server, state, "none", -1)

    # === GUI: Camera ===
    with server.gui.add_folder("Camera"):
        n_cams = len(state.get("cam_poses", []))
        if n_cams > 0:
            gui_cam_idx = server.gui.add_slider(
                "Frame", min=0, max=max(n_cams - 1, 1), step=1, initial_value=0,
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
