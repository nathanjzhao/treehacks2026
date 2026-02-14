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


def points_in_bbox(
    bbox: list[float],
    img_width: int,
    img_height: int,
    cam_position: np.ndarray,
    cam_wxyz: np.ndarray,
    fov_y: float,
    points: np.ndarray,
) -> np.ndarray:
    """Return indices of points that project inside bbox [x1, y1, x2, y2]."""
    R = wxyz_to_rotation_matrix(cam_wxyz)
    points_cam = (points - cam_position) @ R

    in_front = points_cam[:, 2] > 0
    if not np.any(in_front):
        return np.array([], dtype=int)

    points_front = points_cam[in_front]
    original_indices = np.where(in_front)[0]

    f_y = img_height / (2.0 * np.tan(fov_y / 2.0))
    f_x = f_y
    cx, cy = img_width / 2.0, img_height / 2.0

    depth = points_front[:, 2]
    proj_x = f_x * points_front[:, 0] / depth + cx
    proj_y = f_y * points_front[:, 1] / depth + cy

    x1, y1, x2, y2 = bbox
    inside = (proj_x >= x1) & (proj_x <= x2) & (proj_y >= y1) & (proj_y <= y2)

    return original_indices[inside]


def localize_by_ray(
    detections: list[dict],
    viewpoints: list[dict],
    points: np.ndarray,
    img_width: int,
    img_height: int,
    fov_y: float,
) -> np.ndarray:
    """Localize by raycasting through bbox center and finding the first surface.

    Uses ONLY the best detection (highest confidence). Casts a ray from the
    camera through the center of the bbox, finds all points near that ray,
    then uses a depth histogram to find the FIRST (nearest) surface.

    This is geometrically correct and doesn't suffer from multi-view
    averaging artifacts.
    """
    # Pick the single best detection
    best_det = max(detections, key=lambda d: d.get("confidence", 0))
    vi = best_det["view_index"]
    if vi >= len(viewpoints):
        vi = 0
    vp = viewpoints[vi]
    bbox = best_det["bbox"]

    # Bbox center = target pixel
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    # Use half the smaller bbox dimension as search radius
    bbox_half = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.0
    search_radius = max(bbox_half, 20.0)

    print(f"[Raycast] Best detection: view {vi}, bbox={bbox}, center=({cx:.0f},{cy:.0f}), "
          f"search_radius={search_radius:.0f}px", flush=True)

    R = wxyz_to_rotation_matrix(vp["wxyz"])
    points_cam = (points - vp["position"]) @ R

    # Only points in front of camera
    in_front = points_cam[:, 2] > 0.01
    pts = points_cam[in_front]
    orig_idx = np.where(in_front)[0]

    if len(pts) == 0:
        print("[Raycast] WARNING: no points in front of camera", flush=True)
        return points.mean(axis=0)

    # Project all points to image plane
    f = img_height / (2.0 * np.tan(fov_y / 2.0))
    depth = pts[:, 2]
    proj_x = f * pts[:, 0] / depth + img_width / 2.0
    proj_y = f * pts[:, 1] / depth + img_height / 2.0

    # Find points projecting near the bbox center (within search radius)
    dist_sq = (proj_x - cx) ** 2 + (proj_y - cy) ** 2
    in_cone = dist_sq < search_radius ** 2

    if in_cone.sum() < 5:
        # Expand search
        in_cone = dist_sq < (search_radius * 3) ** 2
        print(f"[Raycast] Expanded search: {in_cone.sum()} points", flush=True)

    if in_cone.sum() < 3:
        # Total fallback: nearest projected point
        best = np.argmin(dist_sq)
        print(f"[Raycast] Fallback to nearest projected point", flush=True)
        return points[orig_idx[best]]

    cone_depths = depth[in_cone]
    cone_world_idx = orig_idx[in_cone]

    # Build depth histogram to find the FIRST (nearest) surface
    n_bins = 40
    hist, bin_edges = np.histogram(cone_depths, bins=n_bins)

    # Find the first significant peak (nearest surface)
    # A peak = a bin with at least 15% of the max count
    peak_threshold = max(hist.max() * 0.15, 3)
    peak_bin = None
    for i in range(len(hist)):
        if hist[i] >= peak_threshold:
            peak_bin = i
            break

    if peak_bin is not None:
        # Expand the peak to adjacent bins that are also significant
        peak_start = peak_bin
        peak_end = peak_bin
        while peak_end + 1 < len(hist) and hist[peak_end + 1] >= peak_threshold * 0.5:
            peak_end += 1
            if peak_end - peak_start > 4:  # don't expand too much
                break

        surface_depth_min = bin_edges[peak_start]
        surface_depth_max = bin_edges[peak_end + 1]
        print(f"[Raycast] First surface at depth [{surface_depth_min:.3f}, {surface_depth_max:.3f}]", flush=True)
    else:
        # No clear peak — use nearest 20% of points
        surface_depth_max = np.percentile(cone_depths, 20)
        surface_depth_min = cone_depths.min()
        print(f"[Raycast] No clear peak, using nearest 20%", flush=True)

    # Keep only points at the first surface depth
    surface_mask = (cone_depths >= surface_depth_min) & (cone_depths <= surface_depth_max)
    surface_idx = cone_world_idx[surface_mask]

    if len(surface_idx) == 0:
        surface_idx = cone_world_idx[:10]

    target = points[surface_idx].mean(axis=0)
    print(f"[Raycast] {len(surface_idx)} surface points -> target {target}", flush=True)
    return target


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
