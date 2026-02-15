"""Standalone DPVO result viewer. Run: python3 view_dpvo.py"""
import json
import time
import numpy as np
import trimesh
import viser
from collections import Counter
from pathlib import Path

GLB_PATH = "data/mapanything/IMG_4720.glb"
RESULT_PATH = "hloc_localization/data/dpvo_results/IMG_4730_dpvo.json"
PORT = 9000
DOWNSAMPLE = 4
FRUSTUM_SCALE = 0.4

# --- Load point cloud ---
scene = trimesh.load(GLB_PATH)
points = None
colors = None
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

idx = np.arange(0, len(points), DOWNSAMPLE)
points = points[idx]
colors = colors[idx]
print(f"Point cloud: {len(points):,} points")

# --- Load DPVO result ---
result = json.loads(Path(RESULT_PATH).read_text())
poses = result["poses"]
print(f"Poses: {len(poses)}, scale: {result.get('alignment_scale', 'N/A')}")

# --- Compute camera positions + orientations ---
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

cam_positions = []
cam_wxyz = []
for p in poses:
    R = qvec_to_R(p["qw"], p["qx"], p["qy"], p["qz"])
    pos = -R.T @ np.array([p["tx"], p["ty"], p["tz"]])
    wxyz = mat_to_quat(R.T)  # camera-to-world quaternion
    cam_positions.append(pos)
    cam_wxyz.append(wxyz)
cam_positions = np.array(cam_positions)

print(f"Camera range: {cam_positions.min(axis=0)} -> {cam_positions.max(axis=0)}")
print(f"Point cloud range: {points.min(axis=0)} -> {points.max(axis=0)}")

# --- Start viser ---
server = viser.ViserServer(host="0.0.0.0", port=PORT)

# Point cloud
server.scene.add_point_cloud(
    name="/pc",
    points=points,
    colors=colors,
    point_size=0.005,
    point_shape="rounded",
)

# Camera frustums
for i, p in enumerate(poses):
    is_anchor = p.get("source") == "hloc"
    color = (0, 255, 100) if is_anchor else (255, 100, 50)
    scale = FRUSTUM_SCALE * 1.5 if is_anchor else FRUSTUM_SCALE

    server.scene.add_camera_frustum(
        name=f"/cams/{i:03d}",
        fov=np.deg2rad(60),
        aspect=16/9,
        scale=scale,
        line_width=4.0,
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
    name="/traj",
    points=cam_positions,
    colors=traj_colors,
    point_size=0.025,
    point_shape="circle",
)

# Initial view
all_pts = np.vstack([points[::1000], cam_positions])
center = all_pts.mean(axis=0)
extent = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))

@server.on_client_connect
def _(client):
    client.camera.position = tuple(center + np.array([0, -extent * 0.8, extent * 0.5]))
    client.camera.look_at = tuple(center)
    client.camera.up_direction = (0.0, 0.0, 1.0)

print(f"\nViewer at http://localhost:{PORT}")
print(f"  {len(poses)} frustums ({sum(1 for p in poses if p.get('source')=='hloc')} hloc, "
      f"{sum(1 for p in poses if p.get('source')=='dpvo')} dpvo)")
print(f"  Scene center: {center}, extent: {extent:.1f}")

while True:
    time.sleep(1)
