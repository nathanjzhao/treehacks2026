"""
Viser-based 3D viewer with object finder.

Usage: python viewer.py /path/to/scene.glb
"""

import argparse
import asyncio
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
import uvicorn
import viser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from jinja2 import Template

from object_finder import find_and_fly_to_object
from gemini_client import detect_up_direction
from camera_utils import look_at_wxyz


def _camera_apex(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract the cone apex (camera position) from a VGGT camera cone mesh."""
    faces = np.array(mesh.faces)
    verts = np.array(mesh.vertices)
    counts = Counter(faces.flatten().tolist())
    apex_idx = max(counts, key=counts.get)
    return verts[apex_idx]


def _camera_pose(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Extract camera position AND look direction from a VGGT cone mesh.

    The cone apex = camera position. The direction from apex to base center
    = the original camera look direction from the photo.
    """
    faces = np.array(mesh.faces)
    verts = np.array(mesh.vertices)
    counts = Counter(faces.flatten().tolist())
    apex_idx = max(counts, key=counts.get)
    apex = verts[apex_idx]

    # Base vertices = all except apex
    base_mask = np.ones(len(verts), dtype=bool)
    base_mask[apex_idx] = False
    base_center = verts[base_mask].mean(axis=0)

    # Look direction: apex → base center
    look_dir = base_center - apex
    look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)

    return apex, look_dir


def load_glb(path: str) -> tuple[np.ndarray, np.ndarray, list[trimesh.Trimesh]]:
    """Load a VGGT GLB and return (points, colors, camera_meshes)."""
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


# ---------- Shared state ----------
class AppState:
    """Shared state between viser callbacks and FastAPI."""

    def __init__(self):
        self.points: np.ndarray | None = None
        self.colors: np.ndarray | None = None
        self.centroid: np.ndarray | None = None
        self.active_client: viser.ClientHandle | None = None
        self.camera_positions: list[np.ndarray] | None = None
        self.camera_look_dirs: list[np.ndarray] | None = None
        self.viser_server: viser.ViserServer | None = None


state = AppState()


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

            client = state.active_client
            if client is None:
                await websocket.send_json({
                    "role": "assistant",
                    "content": "No 3D viewer connected. Please open the viewer first.",
                })
                continue

            if state.points is None:
                await websocket.send_json({
                    "role": "assistant",
                    "content": "No point cloud loaded.",
                })
                continue

            async def send_status(text: str):
                await websocket.send_json({"role": "status", "content": text})

            async def send_result(text: str):
                await websocket.send_json({"role": "assistant", "content": text})

            try:
                await find_and_fly_to_object(
                    query=msg,
                    client=client,
                    points=state.points,
                    colors=state.colors,
                    centroid=state.centroid,
                    send_status=send_status,
                    send_result=send_result,
                    camera_positions=state.camera_positions,
                    camera_look_dirs=state.camera_look_dirs,
                    server=state.viser_server,
                )
            except Exception as e:
                await websocket.send_json({
                    "role": "assistant",
                    "content": f"Error finding object: {e}",
                })
    except WebSocketDisconnect:
        pass


# ---------- main ----------
def main():
    global VISER_PORT

    parser = argparse.ArgumentParser(description="3D Object Finder")
    parser.add_argument("glb", help="Path to .glb file")
    parser.add_argument("--port", type=int, default=8080, help="Main web UI port")
    parser.add_argument("--viser-port", type=int, default=8081, help="Internal viser port")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Keep every Nth point (default: 4)")
    args = parser.parse_args()
    VISER_PORT = args.viser_port

    points, colors, camera_meshes = load_glb(args.glb)

    # Downsample if requested
    if args.downsample > 1:
        idx = np.arange(0, len(points), args.downsample)
        points = points[idx]
        colors = colors[idx]
        print(f"Downsampled to {len(points):,} points")

    # Compute scene centroid
    centroid = points.mean(axis=0)

    # Extract camera positions AND orientations from VGGT cones
    cam_positions = []
    cam_look_dirs = []
    for m in camera_meshes:
        pos, look = _camera_pose(m)
        cam_positions.append(pos)
        cam_look_dirs.append(look)

    # Store in shared state
    state.points = points
    state.colors = colors
    state.centroid = centroid
    state.camera_positions = cam_positions
    state.camera_look_dirs = cam_look_dirs

    # Use first VGGT camera as the initial viewer position
    if camera_meshes:
        first_cam_pos = _camera_apex(camera_meshes[0])
        look_dir = centroid - first_cam_pos
        look_dir /= np.linalg.norm(look_dir) + 1e-8
        init_cam_pos = first_cam_pos - look_dir * 0.3
    else:
        bbox_extent = points.max(axis=0) - points.min(axis=0)
        cam_distance = float(np.linalg.norm(bbox_extent)) * 0.8
        init_cam_pos = centroid + np.array([0, -cam_distance, cam_distance * 0.5])

    # Scene up will be detected on first client connect via Gemini
    scene_up = [0.0, 0.0, 1.0]  # default, updated after detection
    up_detected = [False]  # mutable flag for closure

    # --- Start viser ---
    server = viser.ViserServer(host="0.0.0.0", port=args.viser_port)
    state.viser_server = server

    server.scene.add_point_cloud(
        name="/point_cloud",
        points=points,
        colors=colors,
        point_size=0.005,
        point_shape="rounded",
    )

    with server.gui.add_folder("Viewer"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.005
        )

        @gui_point_size.on_update
        def _(_):
            server.scene.add_point_cloud(
                name="/point_cloud",
                points=points,
                colors=colors,
                point_size=gui_point_size.value,
                point_shape="rounded",
            )

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        import object_finder as _of

        state.active_client = client
        client.camera.position = tuple(init_cam_pos)
        client.camera.look_at = tuple(centroid)
        client.camera.up_direction = tuple(scene_up)

        if not up_detected[0] and cam_positions:
            up_detected[0] = True
            print("Detecting scene up direction via Gemini...")

            import threading

            def _detect():
                try:
                    import time
                    time.sleep(0.5)  # let client finish connecting

                    test_pos = cam_positions[0]
                    up_candidates = [
                        (np.array([0.0, 0.0, 1.0]), "A (+Z up)"),
                        (np.array([0.0, 1.0, 0.0]), "B (+Y up)"),
                        (np.array([0.0, -1.0, 0.0]), "C (-Y up)"),
                    ]
                    up_renders = []
                    up_labels = []
                    for up_vec, label in up_candidates:
                        wxyz = look_at_wxyz(test_pos, centroid, up=up_vec)
                        render = client.get_render(
                            height=480, width=640,
                            wxyz=tuple(wxyz),
                            position=tuple(test_pos),
                            fov=1.2,
                        )
                        up_renders.append(np.array(render))
                        up_labels.append(label)

                    detected = asyncio.run(detect_up_direction(up_renders, up_labels))
                    scene_up[:] = detected.tolist()
                    _of._scene_up = detected
                    print(f"Detected up direction: {detected}")

                    # Update camera with correct orientation
                    client.camera.up_direction = tuple(scene_up)
                    client.camera.position = tuple(init_cam_pos)
                    client.camera.look_at = tuple(centroid)
                except Exception as e:
                    print(f"Up detection failed: {e}")

            threading.Thread(target=_detect, daemon=True).start()

    @server.on_client_disconnect
    def _(client: viser.ClientHandle):
        if state.active_client is client:
            state.active_client = None

    print(f"\nViser running on http://localhost:{args.viser_port} (internal)")
    print(f"Web UI at http://localhost:{args.port}")
    print(f"Scene center: {centroid}, init camera: {init_cam_pos}")

    # --- Run FastAPI on main thread ---
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
