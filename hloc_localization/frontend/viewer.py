"""
Live 3D viewer with camera localization overlay.

Loads a GLB point cloud, connects to the localization server,
and renders live camera frustums as poses arrive.

Usage:
  python localization/viewer.py data/reconstruction/IMG_4717.glb \\
    --localization-url http://localhost:8090
"""

import argparse
import asyncio
import json
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
import uvicorn
import viser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from jinja2 import Template


def _camera_apex(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract the cone apex (camera position) from a camera cone mesh."""
    faces = np.array(mesh.faces)
    verts = np.array(mesh.vertices)
    counts = Counter(faces.flatten().tolist())
    apex_idx = max(counts, key=counts.get)
    return verts[apex_idx]


def load_glb(path: str) -> tuple[np.ndarray, np.ndarray, list[trimesh.Trimesh]]:
    """Load a GLB and return (points, colors, camera_meshes)."""
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
    """Quaternion (w, x, y, z) to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])
    return R


def pose_to_transform(qw, qx, qy, qz, tx, ty, tz):
    """Convert COLMAP pose (world-to-camera) to camera-to-world 4x4 transform."""
    R = qvec_to_rotation(qw, qx, qy, qz)
    t = np.array([tx, ty, tz])
    # COLMAP stores world-to-camera, invert to get camera-to-world
    R_inv = R.T
    t_inv = -R_inv @ t
    T = np.eye(4)
    T[:3, :3] = R_inv
    T[:3, 3] = t_inv
    return T


# ---------- Shared state ----------
class AppState:
    def __init__(self):
        self.points: np.ndarray | None = None
        self.centroid: np.ndarray | None = None
        self.active_client: viser.ClientHandle | None = None
        self.viser_server: viser.ViserServer | None = None
        self.localization_url: str | None = None
        self.pose_history: list[dict] = []
        self.frustum_count: int = 0


state = AppState()


# ---------- FastAPI app ----------
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
VISER_PORT = 8081


@app.get("/", response_class=HTMLResponse)
async def index():
    template = Template(TEMPLATE_PATH.read_text())
    return template.render(viser_port=VISER_PORT)


@app.get("/api/reference/status")
async def reference_status():
    """Proxy to localization server."""
    if not state.localization_url:
        return {"available": [], "active_reference": None, "building": False}
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{state.localization_url}/reference/status")
        return resp.json()


@app.post("/api/reference/select/{name}")
async def select_reference(name: str):
    """Proxy to localization server."""
    if not state.localization_url:
        return {"error": "No localization server configured"}
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{state.localization_url}/reference/select/{name}")
        return resp.json()


@app.websocket("/ws/localize")
async def localize_ws(websocket: WebSocket):
    """
    Client sends JPEG frames (binary), server forwards to localization server,
    returns poses, and updates the 3D viewer with camera frustums.
    """
    await websocket.accept()

    if not state.localization_url:
        await websocket.send_json({"error": "No localization server configured"})
        await websocket.close()
        return

    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                frame_bytes = await websocket.receive_bytes()

                t0 = time.time()
                # Forward to localization server
                files = {"image": ("frame.jpg", frame_bytes, "image/jpeg")}
                resp = await client.post(
                    f"{state.localization_url}/localize",
                    files=files,
                )
                pose = resp.json()
                pose["latency_ms"] = (time.time() - t0) * 1000

                # Update 3D viewer with camera frustum
                if pose.get("success"):
                    _add_camera_frustum(pose)
                    state.pose_history.append(pose)

                await websocket.send_json(pose)

    except WebSocketDisconnect:
        pass


def _add_camera_frustum(pose: dict):
    """Add a camera frustum to the viser scene."""
    server = state.viser_server
    if server is None:
        return

    T = pose_to_transform(
        pose["qw"], pose["qx"], pose["qy"], pose["qz"],
        pose["tx"], pose["ty"], pose["tz"],
    )

    cam_pos = T[:3, 3]
    state.frustum_count += 1
    name = f"/live_camera_{state.frustum_count:04d}"

    # Add frustum visualization
    server.scene.add_frame(
        name=name,
        wxyz=np.array([pose["qw"], pose["qx"], pose["qy"], pose["qz"]]),
        position=tuple(cam_pos),
        axes_length=0.05,
        axes_radius=0.003,
    )

    # Keep only last N frustums visible
    max_visible = 30
    if state.frustum_count > max_visible:
        old_name = f"/live_camera_{state.frustum_count - max_visible:04d}"
        try:
            server.scene.remove(old_name)
        except Exception:
            pass

    # Draw trajectory line from recent poses
    if len(state.pose_history) >= 2:
        recent = state.pose_history[-min(50, len(state.pose_history)):]
        positions = []
        for p in recent:
            t = pose_to_transform(p["qw"], p["qx"], p["qy"], p["qz"],
                                   p["tx"], p["ty"], p["tz"])
            positions.append(t[:3, 3])
        positions = np.array(positions)

        # Spline/line through recent positions
        colors = np.full((len(positions), 3), [255, 100, 50], dtype=np.uint8)
        server.scene.add_point_cloud(
            name="/live_trajectory",
            points=positions,
            colors=colors,
            point_size=0.008,
            point_shape="circle",
        )


# ---------- main ----------
def main():
    global VISER_PORT

    parser = argparse.ArgumentParser(description="Live Camera Localization Viewer")
    parser.add_argument("glb", help="Path to .glb file")
    parser.add_argument("--port", type=int, default=8085, help="Web UI port")
    parser.add_argument("--viser-port", type=int, default=8082, help="Viser port")
    parser.add_argument("--localization-url", default="http://localhost:8090",
                        help="URL of the hloc localization server")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Keep every Nth point (default: 4)")
    args = parser.parse_args()
    VISER_PORT = args.viser_port
    state.localization_url = args.localization_url

    points, colors, camera_meshes = load_glb(args.glb)

    if args.downsample > 1:
        idx = np.arange(0, len(points), args.downsample)
        points = points[idx]
        colors = colors[idx]
        print(f"Downsampled to {len(points):,} points")

    centroid = points.mean(axis=0)
    state.points = points
    state.centroid = centroid

    # Initial camera position
    if camera_meshes:
        first_cam_pos = _camera_apex(camera_meshes[0])
        look_dir = centroid - first_cam_pos
        look_dir /= np.linalg.norm(look_dir) + 1e-8
        init_cam_pos = first_cam_pos - look_dir * 0.3
    else:
        bbox_extent = points.max(axis=0) - points.min(axis=0)
        cam_distance = float(np.linalg.norm(bbox_extent)) * 0.8
        init_cam_pos = centroid + np.array([0, -cam_distance, cam_distance * 0.5])

    # Start viser
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
        state.active_client = client
        client.camera.position = tuple(init_cam_pos)
        client.camera.look_at = tuple(centroid)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    @server.on_client_disconnect
    def _(client: viser.ClientHandle):
        if state.active_client is client:
            state.active_client = None

    print(f"\nViser running on http://localhost:{args.viser_port}")
    print(f"Web UI at http://localhost:{args.port}")
    print(f"Localization server: {args.localization_url}")
    print(f"Scene center: {centroid}, init camera: {init_cam_pos}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
