"""
Depth Anything V2 viewer — side-by-side video + interactive 3D point cloud.

Usage:
  python viewer.py examples/ --source-dir ~/Downloads
  python viewer.py examples/IMG_4717_depth.mp4 --source ~/Downloads/IMG_4717.MOV
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import viser


# ---------------------------------------------------------------------------
# Depth extraction from INFERNO colormap
# ---------------------------------------------------------------------------

def _build_inferno_lut():
    """Build a lookup table mapping INFERNO BGR values -> normalized depth [0,1]."""
    import matplotlib.cm as cm
    lut = np.zeros((256, 256, 256), dtype=np.float32)
    cmap = cm.get_cmap("inferno", 256)
    inferno_rgb = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    # Store as BGR -> depth mapping (nearest-neighbor via the LUT)
    # Instead of a full 256^3 LUT, we'll use vectorized nearest search
    return inferno_rgb


def colormap_to_depth(frame_bgr: np.ndarray, inferno_rgb: np.ndarray) -> np.ndarray:
    """Convert an INFERNO-colormapped BGR frame back to approximate depth [0,1].

    Uses nearest-neighbor matching against the 256-entry colormap palette."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = frame_rgb.shape[:2]

    # Reshape for broadcasting: (H*W, 1, 3) vs (1, 256, 3)
    pixels = frame_rgb.reshape(-1, 1, 3)
    palette = inferno_rgb.reshape(1, 256, 3).astype(np.float32)

    # Find nearest colormap index for each pixel
    dists = np.sum((pixels - palette) ** 2, axis=2)  # (H*W, 256)
    indices = np.argmin(dists, axis=1)  # (H*W,)

    depth = (indices.astype(np.float32) / 255.0).reshape(h, w)
    return depth


# ---------------------------------------------------------------------------
# Point cloud from depth
# ---------------------------------------------------------------------------

def depth_to_pointcloud(
    color_frame: np.ndarray,
    depth: np.ndarray,
    downsample: int = 4,
    fov_deg: float = 60.0,
    max_points: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project pixels to 3D using depth and assumed pinhole camera.

    Returns (points_Nx3, colors_Nx3_uint8)."""
    h, w = depth.shape[:2]

    # Downsample
    depth_ds = depth[::downsample, ::downsample]
    color_ds = color_frame[::downsample, ::downsample]
    h_ds, w_ds = depth_ds.shape[:2]

    # Assumed intrinsics from FOV
    fx = w / (2 * np.tan(np.radians(fov_deg / 2)))
    fy = fx
    cx, cy = w / 2.0, h / 2.0

    # Pixel grid (at original resolution coords, sampled)
    u = np.arange(0, w, downsample).astype(np.float32)
    v = np.arange(0, h, downsample).astype(np.float32)
    uu, vv = np.meshgrid(u, v)

    # Scale depth to a reasonable range
    z = depth_ds * 5.0 + 0.1  # avoid zero depth
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z

    points = np.stack([x, -y, -z], axis=-1).reshape(-1, 3)  # flip y,z for GL convention
    colors = cv2.cvtColor(color_ds, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # Filter out very close/far points
    valid = (z.reshape(-1) > 0.15) & (z.reshape(-1) < 4.5)
    points = points[valid]
    colors = colors[valid]

    # Subsample if still too many
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points.astype(np.float32), colors.astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, max_frames: int = 300) -> list[np.ndarray]:
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // max_frames)

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    print(f"  Extracted {len(frames)} frames from {Path(video_path).name} ({total} total, interval={interval})")
    return frames


def find_source_video(depth_path: Path, source_dir: Path | None, source_path: Path | None) -> Path | None:
    """Try to find the original source video for a depth video."""
    if source_path and source_path.exists():
        return source_path

    # Strip _depth suffix and try common extensions
    stem = depth_path.stem.replace("_depth", "")
    search_dirs = [depth_path.parent]
    if source_dir:
        search_dirs.insert(0, source_dir)

    for d in search_dirs:
        for ext in [".MOV", ".mov", ".mp4", ".MP4", ".avi", ".mkv", ".webm"]:
            candidate = d / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 viewer")
    parser.add_argument("path", help="Depth video file or directory of depth videos")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Directory containing original source videos")
    parser.add_argument("--source", type=str, default=None,
                        help="Explicit source video path (for single file mode)")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Point cloud spatial downsample factor (default: 4)")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Max frames to load per video (default: 200)")
    args = parser.parse_args()

    # Discover depth videos
    input_path = Path(args.path)
    if input_path.is_dir():
        depth_files = sorted(input_path.glob("*_depth.mp4"))
    else:
        depth_files = [input_path]

    if not depth_files:
        print(f"No depth videos found at {args.path}")
        return

    source_dir = Path(args.source_dir) if args.source_dir else None
    source_path = Path(args.source) if args.source else None

    # Load first video
    names = [f.stem for f in depth_files]
    file_map = {f.stem: f for f in depth_files}
    initial = depth_files[0].stem

    print(f"Found {len(depth_files)} depth videos: {', '.join(names)}")

    # Build inferno reverse LUT
    inferno_rgb = _build_inferno_lut()

    # --- Extract frames ---
    print(f"\nLoading {initial}...")
    depth_frames = extract_frames(str(file_map[initial]), args.max_frames)

    src = find_source_video(file_map[initial], source_dir, source_path)
    if src:
        print(f"  Source: {src.name}")
        color_frames = extract_frames(str(src), args.max_frames)
        # Match frame counts
        n = min(len(depth_frames), len(color_frames))
        depth_frames = depth_frames[:n]
        color_frames = color_frames[:n]
    else:
        print("  No source video found — using depth colormap as color")
        color_frames = depth_frames

    # --- Start viser ---
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    state = {
        "depth_frames": depth_frames,
        "color_frames": color_frames,
        "current_frame": 0,
        "inferno_rgb": inferno_rgb,
        "downsample": args.downsample,
    }

    def show_frame(idx: int):
        """Update the 3D point cloud for a given frame index."""
        idx = max(0, min(idx, len(state["depth_frames"]) - 1))
        state["current_frame"] = idx

        depth = colormap_to_depth(state["depth_frames"][idx], state["inferno_rgb"])
        color = state["color_frames"][idx]

        # Resize color to match depth if needed
        dh, dw = depth.shape[:2]
        ch, cw = color.shape[:2]
        if (dh, dw) != (ch, cw):
            color = cv2.resize(color, (dw, dh))

        points, colors = depth_to_pointcloud(
            color, depth,
            downsample=state["downsample"],
            max_points=250_000,
        )

        server.scene.add_point_cloud(
            name="/depth_cloud",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
            point_shape="rounded",
        )

        # Side-by-side image display
        depth_vis = state["depth_frames"][idx]
        color_vis = state["color_frames"][idx]
        # Resize both to same height
        target_h = 360
        dh, dw = depth_vis.shape[:2]
        depth_small = cv2.resize(depth_vis, (int(dw * target_h / dh), target_h))
        ch, cw = color_vis.shape[:2]
        color_small = cv2.resize(color_vis, (int(cw * target_h / ch), target_h))
        # Match widths
        target_w = min(depth_small.shape[1], color_small.shape[1])
        depth_small = cv2.resize(depth_small, (target_w, target_h))
        color_small = cv2.resize(color_small, (target_w, target_h))

        sidebyside = np.hstack([color_small, depth_small])
        sidebyside_rgb = cv2.cvtColor(sidebyside, cv2.COLOR_BGR2RGB)

        server.scene.add_image(
            name="/sidebyside",
            image=sidebyside_rgb,
            render_width=4.0,
            render_height=4.0 * target_h / (target_w * 2),
            position=(0, 2.5, -3),
        )

    # --- GUI ---
    with server.gui.add_folder("Playback"):
        if len(depth_files) > 1:
            gui_scene = server.gui.add_dropdown(
                "Video", options=names, initial_value=initial
            )

            @gui_scene.on_update
            def _on_scene_change(_):
                name = gui_scene.value
                print(f"\nSwitching to {name}...")
                df = extract_frames(str(file_map[name]), args.max_frames)
                src_v = find_source_video(file_map[name], source_dir, None)
                if src_v:
                    cf = extract_frames(str(src_v), args.max_frames)
                    n = min(len(df), len(cf))
                    df, cf = df[:n], cf[:n]
                else:
                    cf = df
                state["depth_frames"] = df
                state["color_frames"] = cf
                gui_frame.max = len(df) - 1
                gui_frame.value = 0
                show_frame(0)

        gui_frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(len(depth_frames) - 1, 1),
            step=1,
            initial_value=0,
        )

        @gui_frame.on_update
        def _on_frame_change(_):
            show_frame(int(gui_frame.value))

        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.008
        )

        @gui_point_size.on_update
        def _on_size_change(_):
            show_frame(state["current_frame"])

    # Initial display
    show_frame(0)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, 0.5, 2.0)
        client.camera.look_at = (0.0, 0.0, -1.0)
        client.camera.up_direction = (0.0, -1.0, 0.0)

    print(f"\nViewer running at http://localhost:{args.port}")
    print("Use the Frame slider to scrub through the video.")
    print("The side-by-side panel shows original + depth above the 3D view.")

    # Keep alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
