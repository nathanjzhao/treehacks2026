"""
Viser-based 3D viewer for MapAnything GLB point clouds.

Usage: python viewer.py examples/        # browse all GLBs in directory
       python viewer.py examples/out.glb  # open a specific file
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
import uvicorn
import viser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from jinja2 import Template


def _cone_apex(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract the cone apex (camera position) from a camera cone mesh.

    The apex is the vertex shared by the most faces (the tip of the cone)."""
    faces = np.array(mesh.faces)
    counts = Counter(faces.flatten().tolist())
    apex_idx = max(counts, key=counts.get)
    return np.array(mesh.vertices[apex_idx])


def _estimate_up(camera_meshes: list[trimesh.Trimesh], scene_centroid: np.ndarray) -> np.ndarray:
    """Estimate scene up direction from camera positions using PCA.

    Cameras are typically at roughly the same height, so the thinnest
    axis of the camera position distribution points "up"."""
    if len(camera_meshes) < 3:
        return np.array([0.0, -1.0, 0.0])  # fallback: OpenCV -Y is up

    positions = np.array([_cone_apex(m) for m in camera_meshes])
    centered = positions - positions.mean(axis=0)
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Smallest eigenvalue = thinnest spread = up direction
    up = eigenvectors[:, 0]

    # Make sure "up" points away from scene centroid relative to cameras
    cam_mean = positions.mean(axis=0)
    if np.dot(up, cam_mean - scene_centroid) < 0:
        up = -up

    return up / (np.linalg.norm(up) + 1e-8)


def load_glb(path: str) -> tuple[np.ndarray, np.ndarray, list[trimesh.Trimesh]]:
    """Load a MapAnything GLB and return (points, colors, camera_meshes)."""
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

    print(f"Loaded {len(points):,} points, {len(camera_meshes)} camera cones from {Path(path).name}")
    return points, colors, camera_meshes


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
            reply = f"I received your message: \"{msg}\". (LLM not connected yet.)"
            await websocket.send_json({"role": "assistant", "content": reply})
    except WebSocketDisconnect:
        pass


def _load_and_display(server: viser.ViserServer, glb_path: str, downsample: int):
    """Load a GLB, downsample, and display it. Returns scene state dict."""
    points, colors, camera_meshes = load_glb(glb_path)

    if downsample > 1:
        idx = np.arange(0, len(points), downsample)
        points = points[idx]
        colors = colors[idx]
        print(f"Downsampled to {len(points):,} points")

    centroid = points.mean(axis=0)
    up = _estimate_up(camera_meshes, centroid)
    print(f"Estimated up direction: {up}")

    if camera_meshes:
        first_cam_pos = _cone_apex(camera_meshes[0])
        look_dir = centroid - first_cam_pos
        look_dir /= np.linalg.norm(look_dir) + 1e-8
        init_cam_pos = first_cam_pos - look_dir * 0.3
    else:
        bbox_extent = points.max(axis=0) - points.min(axis=0)
        cam_distance = float(np.linalg.norm(bbox_extent)) * 0.8
        init_cam_pos = centroid + up * cam_distance

    base_point_size = 0.005 * (downsample ** 0.5)

    server.scene.add_point_cloud(
        name="/point_cloud",
        points=points,
        colors=colors,
        point_size=base_point_size,
        point_shape="rounded",
    )

    # Move all connected clients to the new camera position
    for client in server.get_clients().values():
        client.camera.position = tuple(init_cam_pos)
        client.camera.look_at = tuple(centroid)
        client.camera.up_direction = tuple(up)

    return points, colors, centroid, init_cam_pos, base_point_size, up


# ---------- main ----------
def main():
    global VISER_PORT

    parser = argparse.ArgumentParser(description="View MapAnything GLB point clouds")
    parser.add_argument("glb", help="Path to .glb file or directory of .glb files")
    parser.add_argument("--port", type=int, default=8080, help="Main web UI port")
    parser.add_argument("--viser-port", type=int, default=8081, help="Internal viser port")
    parser.add_argument("--downsample", type=int, default=20,
                        help="Keep every Nth point (default: 20)")
    args = parser.parse_args()
    VISER_PORT = args.viser_port

    # Discover all GLB files
    glb_input = Path(args.glb)
    if glb_input.is_dir():
        glb_files = sorted(glb_input.glob("*.glb"))
    else:
        glb_files = [glb_input]
        # Also find siblings in the same directory
        siblings = sorted(glb_input.parent.glob("*.glb"))
        if len(siblings) > 1:
            glb_files = siblings

    if not glb_files:
        raise ValueError(f"No .glb files found at {args.glb}")

    glb_names = [f.stem for f in glb_files]
    glb_map = {f.stem: str(f) for f in glb_files}
    initial_name = glb_input.stem if glb_input.is_file() else glb_names[0]

    print(f"Found {len(glb_files)} GLB files: {', '.join(glb_names)}")

    # --- Start viser ---
    server = viser.ViserServer(host="0.0.0.0", port=args.viser_port)

    # Load initial scene
    points, colors, centroid, init_cam_pos, base_point_size, up = _load_and_display(
        server, glb_map[initial_name], args.downsample
    )

    # State shared between GUI callbacks
    state = {
        "points": points,
        "colors": colors,
        "point_size": base_point_size,
        "centroid": centroid,
        "init_cam_pos": init_cam_pos,
        "up": up,
    }

    with server.gui.add_folder("Viewer"):
        # File picker dropdown
        if len(glb_files) > 1:
            gui_file = server.gui.add_dropdown(
                "Scene",
                options=glb_names,
                initial_value=initial_name,
            )

            @gui_file.on_update
            def _on_file_change(_):
                name = gui_file.value
                print(f"\nSwitching to {name}...")
                pts, cols, cent, cam_pos, pt_size, up_dir = _load_and_display(
                    server, glb_map[name], args.downsample
                )
                state["points"] = pts
                state["colors"] = cols
                state["point_size"] = pt_size
                state["centroid"] = cent
                state["init_cam_pos"] = cam_pos
                state["up"] = up_dir
                gui_point_size.value = pt_size

        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.1, step=0.001, initial_value=base_point_size
        )

        @gui_point_size.on_update
        def _on_size_change(_):
            server.scene.add_point_cloud(
                name="/point_cloud",
                points=state["points"],
                colors=state["colors"],
                point_size=gui_point_size.value,
                point_shape="rounded",
            )

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = tuple(state["init_cam_pos"])
        client.camera.look_at = tuple(state["centroid"])
        client.camera.up_direction = tuple(state["up"])

    print(f"\nViser running on http://localhost:{args.viser_port}")
    print(f"Web UI at http://localhost:{args.port}")

    # --- Run FastAPI on main thread ---
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
