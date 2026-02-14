"""Camera math utilities for the 3D object finder."""

import numpy as np


def rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def wxyz_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def look_at_wxyz(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """Compute a viser wxyz quaternion for a camera at `position` looking at `target`.

    Viser uses OpenCV convention: camera looks along +Z, -Y is up in camera space.
    R_world_camera columns are [right, -up, forward].
    """
    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    up_ortho = np.cross(right, forward)
    up_ortho = up_ortho / (np.linalg.norm(up_ortho) + 1e-8)

    # Rotation matrix: columns are right, -up, forward (OpenCV convention)
    R = np.stack([right, -up_ortho, forward], axis=1)
    return rotation_matrix_to_wxyz(R)


def generate_orbit_viewpoints(centroid: np.ndarray, radius: float) -> list[dict]:
    """Generate 6 viewpoints on a sphere around the centroid.

    3 azimuth angles x 2 elevations = 6 viewpoints.
    Returns list of dicts with 'position', 'wxyz', 'label' keys.
    """
    viewpoints = []
    azimuths = [0, 120, 240]
    elevations = [25, 55]

    for elev_deg in elevations:
        for az_deg in azimuths:
            az = np.radians(az_deg)
            el = np.radians(elev_deg)

            x = radius * np.cos(el) * np.cos(az)
            y = radius * np.cos(el) * np.sin(az)
            z = radius * np.sin(el)

            position = centroid + np.array([x, y, z])
            wxyz = look_at_wxyz(position, centroid)

            viewpoints.append({
                "position": position,
                "wxyz": wxyz,
                "label": f"az{az_deg}_el{elev_deg}",
            })

    return viewpoints


def unproject_pixel_to_3d(
    px: float,
    py: float,
    img_width: int,
    img_height: int,
    cam_position: np.ndarray,
    cam_wxyz: np.ndarray,
    fov_y: float,
    points: np.ndarray,
) -> np.ndarray:
    """Find the 3D point that projects closest to (px, py) in the given camera.

    Projects all points into the camera's image plane and finds the nearest
    projected point to the LLM's pixel coordinates.
    """
    R = wxyz_to_rotation_matrix(cam_wxyz)

    # Transform points to camera space: p_cam = R^T @ (p_world - cam_position)
    points_cam = (points - cam_position) @ R  # (N, 3)

    # Filter points in front of camera (positive Z in OpenCV = in front)
    in_front = points_cam[:, 2] > 0
    if not np.any(in_front):
        return points.mean(axis=0)

    points_front = points_cam[in_front]
    original_indices = np.where(in_front)[0]

    # Perspective projection (OpenCV convention: Y down, Z forward)
    f_y = img_height / (2.0 * np.tan(fov_y / 2.0))
    f_x = f_y
    cx, cy = img_width / 2.0, img_height / 2.0

    depth = points_front[:, 2]
    proj_x = f_x * points_front[:, 0] / depth + cx
    proj_y = f_y * points_front[:, 1] / depth + cy

    dist_sq = (proj_x - px) ** 2 + (proj_y - py) ** 2
    best_idx = np.argmin(dist_sq)

    return points[original_indices[best_idx]]


def interpolate_camera(
    start_pos: np.ndarray,
    start_look_at: np.ndarray,
    end_pos: np.ndarray,
    end_look_at: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate camera position and look_at with cubic ease-in-out.

    t: normalized time [0, 1]
    Returns (position, look_at) at time t.
    """
    if t < 0.5:
        eased = 4 * t * t * t
    else:
        eased = 1 - (-2 * t + 2) ** 3 / 2

    pos = start_pos + (end_pos - start_pos) * eased
    look = start_look_at + (end_look_at - start_look_at) * eased
    return pos, look
