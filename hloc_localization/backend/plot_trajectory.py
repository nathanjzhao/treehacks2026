"""Quick trajectory plot from already-computed poses."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib


def quat_to_rotation(qw, qx, qy, qz):
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
    return R


# Results from debug_localize run
results = [
    {"success": True, "tx": -1.356, "ty": -0.480, "tz": 6.893, "qw": 0.988, "qx": -0.090, "qy": 0.126, "qz": 0.022, "num_inliers": 8380},
    {"success": True, "tx": -0.753, "ty": -0.636, "tz": 6.195, "qw": 0.983, "qx": -0.063, "qy": 0.170, "qz": 0.038, "num_inliers": 5786},
    {"success": True, "tx": -0.267, "ty": -0.522, "tz": 4.587, "qw": 0.973, "qx": -0.029, "qy": 0.225, "qz": 0.050, "num_inliers": 7941},
    {"success": True, "tx": 0.403, "ty": -0.283, "tz": 3.377, "qw": 0.970, "qx": -0.013, "qy": 0.238, "qz": 0.050, "num_inliers": 7474},
    {"success": True, "tx": 1.351, "ty": -0.013, "tz": 1.723, "qw": 0.976, "qx": -0.005, "qy": 0.216, "qz": 0.041, "num_inliers": 8651},
    {"success": True, "tx": 1.864, "ty": 0.116, "tz": -0.059, "qw": 0.985, "qx": -0.002, "qy": 0.172, "qz": 0.024, "num_inliers": 8776},
    {"success": True, "tx": 2.440, "ty": -0.543, "tz": -1.577, "qw": 0.991, "qx": -0.034, "qy": 0.129, "qz": 0.007, "num_inliers": 4045},
    {"success": True, "tx": 2.781, "ty": -1.547, "tz": -1.993, "qw": 0.989, "qx": -0.099, "qy": 0.114, "qz": 0.000, "num_inliers": 3549},
    {"success": True, "tx": 2.986, "ty": -1.735, "tz": -1.792, "qw": 0.988, "qx": -0.118, "qy": 0.098, "qz": 0.004, "num_inliers": 3597},
    {"success": True, "tx": 2.993, "ty": -1.829, "tz": -1.613, "qw": 0.987, "qx": -0.125, "qy": 0.097, "qz": 0.005, "num_inliers": 4290},
]

positions = []
forwards = []

for r in results:
    R = quat_to_rotation(r["qw"], r["qx"], r["qy"], r["qz"])
    t = np.array([r["tx"], r["ty"], r["tz"]])
    cam_pos = -R.T @ t  # camera center in world coordinates
    cam_fwd = R.T @ np.array([0, 0, 1])
    positions.append(cam_pos)
    forwards.append(cam_fwd)

positions = np.array(positions)
forwards = np.array(forwards)

print(f"Camera positions (world coords):")
for i, p in enumerate(positions):
    print(f"  Frame {i}: ({p[0]:7.3f}, {p[1]:7.3f}, {p[2]:7.3f})")

fig = plt.figure(figsize=(16, 6))

# 3D trajectory
ax1 = fig.add_subplot(131, projection="3d")
ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-o", markersize=5, linewidth=2)
ax1.scatter(*positions[0], c="green", s=120, marker="^", label="Start", zorder=5)
ax1.scatter(*positions[-1], c="red", s=120, marker="v", label="End", zorder=5)

scale = 0.4
for i, (p, f) in enumerate(zip(positions, forwards)):
    ax1.quiver(p[0], p[1], p[2], f[0]*scale, f[1]*scale, f[2]*scale,
               color="orange", arrow_length_ratio=0.3, linewidth=1.2)

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title("3D Camera Trajectory")
ax1.legend()

# Top-down (XZ)
ax2 = fig.add_subplot(132)
ax2.plot(positions[:, 0], positions[:, 2], "b-o", markersize=5, linewidth=2)
ax2.scatter(positions[0, 0], positions[0, 2], c="green", s=120, marker="^", label="Start", zorder=5)
ax2.scatter(positions[-1, 0], positions[-1, 2], c="red", s=120, marker="v", label="End", zorder=5)
for p, f in zip(positions, forwards):
    ax2.annotate("", xy=(p[0]+f[0]*scale, p[2]+f[2]*scale), xytext=(p[0], p[2]),
                 arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
ax2.set_title("Top-Down (XZ)")
ax2.legend()
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.3)

# Inlier count over time
ax3 = fig.add_subplot(133)
inliers = [r["num_inliers"] for r in results]
ax3.bar(range(len(inliers)), inliers, color="steelblue")
ax3.set_xlabel("Frame")
ax3.set_ylabel("Inliers")
ax3.set_title("PnP Inlier Count")
ax3.grid(True, alpha=0.3, axis="y")

fig.suptitle("IMG_4730 localized against IMG_4720 reference (hloc)", fontsize=14, fontweight="bold")
fig.tight_layout()

out_path = pathlib.Path(__file__).parent.parent / "data" / "trajectory_plot.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")
