"""
Visualize camera poses from debug localization.

Extracts frames from a query video, localizes them against a reference,
and renders the trajectory as an interactive 3D plot.

Usage:
  python -m hloc_localization.backend.visualize_poses \
    --video data/IMG_4730.MOV \
    --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz \
    --fps 2
"""

import argparse
import pathlib
import time

import cv2
import modal
import numpy as np

from hloc_localization.backend.app import app, localize_frame


def quat_to_rotation(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
    return R


def main():
    parser = argparse.ArgumentParser(description="Visualize localized camera poses")
    parser.add_argument("--video", required=True, help="Query video path")
    parser.add_argument("--reference", required=True, help="Reference tar.gz path")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args()

    video_path = pathlib.Path(args.video).expanduser().resolve()
    ref_path = pathlib.Path(args.reference).expanduser().resolve()

    ref_tar = ref_path.read_bytes()
    print(f"Reference: {ref_path.name} ({len(ref_tar) / 1024 / 1024:.1f} MB)")

    # Extract frames
    vs = cv2.VideoCapture(str(video_path))
    source_fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / args.fps))

    frames = []
    count = 0
    while True:
        ok, frame = vs.read()
        if not ok:
            break
        count += 1
        if count % frame_interval == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frames.append(buf.tobytes())
            if len(frames) >= args.max_frames:
                break
    vs.release()
    print(f"Extracted {len(frames)} query frames")

    # Localize
    results = []
    with app.run():
        for i, fb in enumerate(frames):
            t0 = time.time()
            r = localize_frame.remote(fb, ref_tar)
            dt = time.time() - t0
            results.append(r)
            status = f"inliers={r['num_inliers']}" if r["success"] else f"FAIL: {r.get('error','?')}"
            print(f"  Frame {i:3d}: {status} ({dt:.1f}s)")

    # Extract successful poses
    positions = []  # camera positions in world coords
    forwards = []   # camera forward directions
    frame_ids = []

    for i, r in enumerate(results):
        if not r["success"]:
            continue
        # cam_from_world: R @ world_pt + t = cam_pt
        # camera position in world = -R^T @ t
        R = quat_to_rotation(r["qw"], r["qx"], r["qy"], r["qz"])
        t = np.array([r["tx"], r["ty"], r["tz"]])
        cam_pos = -R.T @ t
        cam_fwd = R.T @ np.array([0, 0, 1])  # z-axis in camera space → world
        positions.append(cam_pos)
        forwards.append(cam_fwd)
        frame_ids.append(i)

    if not positions:
        print("No frames localized successfully!")
        return

    positions = np.array(positions)
    forwards = np.array(forwards)

    print(f"\n{len(positions)} poses recovered")
    print(f"Position range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}] "
          f"y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}] "
          f"z=[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")

    # Plot with matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 6))

    # 3D trajectory
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-o", markersize=4, linewidth=1.5)
    ax1.scatter(*positions[0], c="green", s=100, marker="^", label="Start", zorder=5)
    ax1.scatter(*positions[-1], c="red", s=100, marker="v", label="End", zorder=5)

    # Draw forward direction arrows
    scale = 0.3
    for p, f in zip(positions[::2], forwards[::2]):
        ax1.quiver(p[0], p[1], p[2], f[0]*scale, f[1]*scale, f[2]*scale,
                   color="orange", arrow_length_ratio=0.3, linewidth=1)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"Camera Trajectory ({len(positions)} poses)")
    ax1.legend()

    # Top-down view (XZ plane)
    ax2 = fig.add_subplot(122)
    ax2.plot(positions[:, 0], positions[:, 2], "b-o", markersize=4, linewidth=1.5)
    ax2.scatter(positions[0, 0], positions[0, 2], c="green", s=100, marker="^", label="Start", zorder=5)
    ax2.scatter(positions[-1, 0], positions[-1, 2], c="red", s=100, marker="v", label="End", zorder=5)

    for p, f in zip(positions[::2], forwards[::2]):
        ax2.annotate("", xy=(p[0]+f[0]*scale, p[2]+f[2]*scale), xytext=(p[0], p[2]),
                     arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))

    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("Top-Down View (XZ)")
    ax2.legend()
    ax2.set_aspect("equal")

    fig.tight_layout()
    out_path = pathlib.Path(__file__).parent.parent / "data" / "trajectory_plot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved trajectory plot to {out_path}")


if __name__ == "__main__":
    main()
