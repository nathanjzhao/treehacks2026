"""
Convert a VGGT GLB point cloud into a demo OpenFunGraph scene graph pickle.
Keeps original colors and point density. Clusters spatially for fake "objects".

Usage: python openfungraph/make_demo.py vggt/examples/IMG_4708.glb
"""

import argparse
import pickle
from collections import Counter

import numpy as np
import trimesh


def _camera_apex(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract the cone apex (camera position) from a VGGT camera cone mesh."""
    faces = np.array(mesh.faces)
    verts = np.array(mesh.vertices)
    counts = Counter(faces.flatten().tolist())
    apex_idx = max(counts, key=counts.get)
    return verts[apex_idx]


def load_glb(path: str):
    """Extract points, colors, and camera meshes from a VGGT GLB."""
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
            colors = c[:, :3]
        elif isinstance(geom, trimesh.Trimesh):
            camera_meshes.append(geom)

    if points is None:
        raise ValueError("No PointCloud in GLB")
    return points, colors, camera_meshes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", help="Path to .glb file")
    parser.add_argument("--output", "-o", default=None, help="Output .pkl path")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of fake objects")
    parser.add_argument("--downsample", type=int, default=4, help="Keep every Nth point")
    args = parser.parse_args()

    print(f"Loading {args.glb}...")
    points, colors, camera_meshes = load_glb(args.glb)
    print(f"  {len(points):,} points, {len(camera_meshes)} cameras")

    # Downsample
    if args.downsample > 1:
        idx = np.arange(0, len(points), args.downsample)
        points = points[idx]
        colors = colors[idx]
        print(f"  Downsampled to {len(points):,} points")

    # Extract camera positions for viewer
    cam_positions = []
    for cm in camera_meshes:
        cam_positions.append(_camera_apex(cm).tolist())

    # Cluster into objects with original colors
    from scipy.cluster.vq import kmeans2
    centroids, labels = kmeans2(points, args.n_clusters, minit="points", iter=20)

    object_names = [
        "table", "chair", "lamp", "shelf", "door",
        "cabinet", "monitor", "plant", "sofa", "window",
        "desk", "box", "bed", "rug", "wall_art",
    ]

    objects = []
    for i in range(args.n_clusters):
        mask = labels == i
        if mask.sum() < 50:
            continue
        obj_pts = points[mask]
        obj_cols = colors[mask].astype(np.float32) / 255.0  # keep original RGB
        name = object_names[i % len(object_names)]
        objects.append({
            "pcd_np": obj_pts,
            "pcd_color_np": obj_cols,
            "clip_ft": np.random.randn(1024).astype(np.float32),
            "text_ft": np.random.randn(1024).astype(np.float32),
            "class_name": name,
            "refined_obj_tag": f"{name}_{i}",
            "n_detections": 10,
        })

    # Fake parts
    parts = []
    part_names = ["handle", "button", "knob", "switch"]
    for i, obj in enumerate(objects[:4]):
        center = obj["pcd_np"].mean(axis=0)
        dists = np.linalg.norm(obj["pcd_np"] - center, axis=1)
        close = dists < np.percentile(dists, 15)
        if close.sum() < 10:
            continue
        parts.append({
            "pcd_np": obj["pcd_np"][close],
            "pcd_color_np": obj["pcd_color_np"][close],
            "clip_ft": np.random.randn(1024).astype(np.float32),
            "class_name": part_names[i % len(part_names)],
            "n_detections": 5,
        })

    # Fake edges
    edges = []
    if len(objects) >= 2 and len(parts) >= 1:
        edges.append((0, -1, 0, "pulling opens door", 0.85))
        edges.append((1, -1, 1, "pressing toggles light", 0.72))
    if len(objects) >= 4:
        edges.append((2, -1, 3, "remote controls device", 0.65))

    result = {
        "objects": objects,
        "parts": parts,
        "edges": edges,
        "n_frames": 10,
        "camera_positions": cam_positions,
    }

    out_path = args.output or args.glb.replace(".glb", "_scene_graph.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Created demo scene graph: {len(objects)} objects, {len(parts)} parts, {len(edges)} edges")
    print(f"Camera positions: {len(cam_positions)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
