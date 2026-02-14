"""Pipeline: render -> LLM (multi-view bbox) -> 3D consensus -> animate."""

import asyncio
import time
from pathlib import Path
from typing import Callable, Awaitable

import numpy as np
import viser
from PIL import Image

from camera_utils import (
    look_at_wxyz,
    localize_by_ray,
)
from gemini_client import query_object_location, detect_up_direction

# Cached scene up direction (detected once via Gemini)
_scene_up: np.ndarray | None = None

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_FOV = 1.2  # radians (~69 degrees) - wider for indoor scenes
FLY_DURATION = 3.0  # seconds — slower for elderly users
FLY_FPS = 60
NUM_VIEWS = 6  # number of diverse camera views to use
MAX_LLM_RETRIES = 2  # retry LLM if no detections
PULSE_DURATION = 3.0  # seconds of pulsing highlight after landing
PULSE_FPS = 30
MARKER_COLOR_PRIMARY = (255, 50, 50)  # bright red — high contrast
MARKER_COLOR_RING = (255, 180, 60)  # warm orange ring
MARKER_SCALE = 0.003  # relative to scene size


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two wxyz quaternions."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(np.clip(dot, -1, 1))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def _select_diverse_cameras(
    cam_positions: list[np.ndarray],
    n: int,
) -> list[int]:
    """Select n diverse camera indices using farthest point sampling."""
    positions = np.array(cam_positions)
    selected = [0]  # start with first camera
    for _ in range(n - 1):
        dists = np.min(
            [np.linalg.norm(positions - positions[s], axis=1) for s in selected],
            axis=0,
        )
        # Don't re-select already selected
        for s in selected:
            dists[s] = -1
        selected.append(int(np.argmax(dists)))
    return selected


async def find_and_fly_to_object(
    query: str,
    client: viser.ClientHandle,
    points: np.ndarray,
    colors: np.ndarray | None,
    centroid: np.ndarray,
    send_status: Callable[[str], Awaitable[None]],
    send_result: Callable[[str], Awaitable[None]],
    camera_positions: list[np.ndarray] | None = None,
    camera_look_dirs: list[np.ndarray] | None = None,
    server: viser.ViserServer | None = None,
) -> None:
    """Full object-finding pipeline with multi-view consensus localization.

    1. Render views from VGGT camera positions using ORIGINAL orientations
    2. LLM identifies object with bounding boxes in EACH view
    3. Depth-filtered multi-view localization
    4. Fly camera + marker animation
    """
    global _scene_up
    loop = asyncio.get_event_loop()

    # --- Detect scene "up" direction on first run ---
    if _scene_up is None and camera_positions and len(camera_positions) > 0:
        await send_status("Detecting scene orientation...")
        try:
            # Render the same view with 3 candidate up vectors
            test_pos = camera_positions[0]
            test_look = centroid
            up_candidates = [
                (np.array([0.0, 0.0, 1.0]), "A (+Z up)"),
                (np.array([0.0, 1.0, 0.0]), "B (+Y up)"),
                (np.array([0.0, -1.0, 0.0]), "C (-Y up)"),
            ]
            up_renders = []
            up_labels = []
            for up_vec, label in up_candidates:
                wxyz = look_at_wxyz(test_pos, test_look, up=up_vec)
                render = await loop.run_in_executor(
                    None,
                    lambda wxyz=wxyz: client.get_render(
                        height=RENDER_HEIGHT,
                        width=RENDER_WIDTH,
                        wxyz=tuple(wxyz),
                        position=tuple(test_pos),
                        fov=RENDER_FOV,
                    ),
                )
                arr = np.array(render)
                up_renders.append(arr)
                up_labels.append(label)
                # Save debug renders
                debug_dir = Path(__file__).parent / "debug_renders"
                debug_dir.mkdir(exist_ok=True)
                img = Image.fromarray(arr)
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(debug_dir / f"up_test_{label[0]}.jpg")

            _scene_up = await detect_up_direction(up_renders, up_labels)
            print(f"[Pipeline] Scene up direction: {_scene_up}", flush=True)
        except Exception as e:
            print(f"[Pipeline] Up detection failed: {e}, using +Z", flush=True)
            _scene_up = np.array([0.0, 0.0, 1.0])

    up = _scene_up if _scene_up is not None else np.array([0.0, 0.0, 1.0])

    # Clean up any previous search markers
    if server is not None:
        try:
            server.scene.remove_by_name("/finder/marker")
            server.scene.remove_by_name("/finder/ring")
            server.scene.remove_by_name("/finder/label")
        except Exception:
            pass

    # --- Step 1: Capture multi-view renders ---
    await send_status("Scanning the scene...")

    debug_dir = Path(__file__).parent / "debug_renders"
    debug_dir.mkdir(exist_ok=True)

    # Build viewpoints from VGGT camera positions with ORIGINAL orientations
    if camera_positions and len(camera_positions) >= NUM_VIEWS:
        indices = _select_diverse_cameras(camera_positions, NUM_VIEWS)
        viewpoints = []
        for i, idx in enumerate(indices):
            pos = camera_positions[idx]
            # Use original camera orientation if available, else look at centroid
            if camera_look_dirs and idx < len(camera_look_dirs):
                look_target = pos + camera_look_dirs[idx] * 10.0
                wxyz = look_at_wxyz(pos, look_target)
            else:
                wxyz = look_at_wxyz(pos, centroid)
            viewpoints.append({
                "position": pos,
                "wxyz": wxyz,
                "label": f"cam{idx}",
            })
        print(f"[Pipeline] Using {NUM_VIEWS} VGGT cameras with original orientation: {indices}", flush=True)
    else:
        print("[Pipeline] No VGGT cameras available, using current view only", flush=True)
        viewpoints = [{
            "position": np.array(client.camera.position),
            "wxyz": np.array(client.camera.wxyz),
            "label": "current",
        }]

    # Temporarily add a denser point cloud for better LLM renders
    dense_added = False
    if server is not None and colors is not None:
        try:
            server.scene.add_point_cloud(
                name="/temp_dense_render",
                points=points,
                colors=colors,
                point_size=0.015,  # 3x default — fills gaps for clearer images
                point_shape="rounded",
            )
            dense_added = True
            await asyncio.sleep(0.2)  # let client receive the update
        except Exception as e:
            print(f"[Pipeline] Failed to add dense render cloud: {e}", flush=True)

    renders = []
    render_labels = []
    valid_viewpoints = []  # track which viewpoints produced valid renders
    ts = int(time.time())

    for vp in viewpoints:
        print(f"[Render] {vp['label']}: pos={vp['position']}", flush=True)
        await asyncio.sleep(0.05)

        render = await loop.run_in_executor(
            None,
            lambda vp=vp: client.get_render(
                height=RENDER_HEIGHT,
                width=RENDER_WIDTH,
                wxyz=tuple(vp["wxyz"]),
                position=tuple(vp["position"]),
                fov=RENDER_FOV,
            ),
        )
        arr = np.array(render)

        # Validate render isn't blank/corrupt
        if arr.ndim < 2:
            print(f"[Render] WARNING: {vp['label']} invalid shape {arr.shape}", flush=True)
            continue
        mean_val = arr.mean()
        print(f"[Render] {vp['label']}: shape={arr.shape}, mean={mean_val:.1f}", flush=True)
        if mean_val < 5 or mean_val > 250:
            print(f"[Render] WARNING: {vp['label']} looks blank, skipping", flush=True)
            continue

        renders.append(arr)
        render_labels.append(vp["label"])
        valid_viewpoints.append(vp)

        img = Image.fromarray(arr)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(debug_dir / f"{ts}_{vp['label']}.jpg")

    # Remove temporary dense point cloud
    if dense_added:
        try:
            server.scene.remove_by_name("/temp_dense_render")
        except Exception:
            pass

    if not renders:
        await send_result("Failed to capture scene renders. Try again.")
        return

    print(f"[Pipeline] Rendered {len(renders)} valid views, sending to LLM...", flush=True)

    # --- Step 2: Query LLM for multi-view bounding boxes ---
    await send_status(f"Asking AI to find '{query}'...")

    llm_result = None
    for attempt in range(MAX_LLM_RETRIES):
        result = await query_object_location(query, renders, render_labels)
        detections = result.get("detections", [])
        target_3d_direct = result.get("target_3d")
        description = result.get("description", "No description")
        print(f"[Pipeline] LLM attempt {attempt+1}: {len(detections)} detections, "
              f"box_3d={'yes' if target_3d_direct else 'no'}, desc={description}", flush=True)

        if detections or target_3d_direct:
            llm_result = result
            break
        elif attempt < MAX_LLM_RETRIES - 1:
            print(f"[Pipeline] No detections, retrying...", flush=True)
            await send_status(f"Looking more carefully for '{query}'...")
            await asyncio.sleep(0.5)

    if llm_result is None or (not llm_result.get("detections") and not llm_result.get("target_3d")):
        await send_result(
            f"Sorry, I couldn't find '{query}' in the scene. Try a different description."
        )
        return

    description = llm_result.get("description", "Found object")

    # --- Step 3: Localize in 3D ---
    if llm_result.get("target_3d"):
        # box_3d path — Gemini gave us 3D coords directly
        target_3d = np.array(llm_result["target_3d"])
        n_views_found = 0
        await send_status("Found it via 3D spatial detection!")
        print(f"[Pipeline] box_3d target: {target_3d}", flush=True)
    else:
        # box_2d path — raycast to find 3D position
        detections = llm_result["detections"]
        await send_status("Triangulating 3D position...")
        n_views_found = len(detections)
        print(f"[Pipeline] Object detected in {n_views_found} views, running raycast", flush=True)

        target_3d = localize_by_ray(
            detections=detections,
            viewpoints=valid_viewpoints,
            points=points,
            img_width=RENDER_WIDTH,
            img_height=RENDER_HEIGHT,
            fov_y=RENDER_FOV,
        )
    print(f"[Pipeline] Target 3D position: {target_3d}", flush=True)

    # --- Step 4: Place 3D marker at target ---
    await send_status("Found it! Flying to object...")

    bbox_extent = points.max(axis=0) - points.min(axis=0)
    scene_scale = float(np.linalg.norm(bbox_extent))
    marker_radius = scene_scale * MARKER_SCALE

    if server is not None:
        server.scene.add_icosphere(
            name="/finder/marker",
            radius=marker_radius,
            color=MARKER_COLOR_PRIMARY,
            position=tuple(target_3d),
        )
        # Ring around marker — shape (N, 2, 3)
        n_ring = 48
        angles = np.linspace(0, 2 * np.pi, n_ring + 1)
        ring_r = marker_radius * 4
        ring_pts = np.zeros((n_ring, 2, 3))
        for j in range(n_ring):
            ring_pts[j, 0] = target_3d + np.array([
                ring_r * np.cos(angles[j]), ring_r * np.sin(angles[j]), 0
            ])
            ring_pts[j, 1] = target_3d + np.array([
                ring_r * np.cos(angles[j + 1]), ring_r * np.sin(angles[j + 1]), 0
            ])
        server.scene.add_line_segments(
            name="/finder/ring",
            points=ring_pts.astype(np.float32),
            colors=MARKER_COLOR_RING,
            line_width=3.0,
        )
        label_offset = target_3d + np.array([0, 0, marker_radius * 8])
        server.scene.add_label(
            name="/finder/label",
            text=description,
            position=tuple(label_offset),
        )

    # --- Step 5: Fly camera to the best viewing position ---
    # Use original VGGT camera positions — guaranteed to be in free space.
    # Pick the one that saw the object with highest confidence, or nearest.
    if detections:
        best_det = max(detections, key=lambda d: d.get("confidence", 0))
        best_vp = valid_viewpoints[best_det["view_index"]]
    else:
        dists = [np.linalg.norm(vp["position"] - target_3d) for vp in valid_viewpoints]
        best_vp = valid_viewpoints[int(np.argmin(dists))]

    # Fly to the original camera position but look at the target
    dest_pos = np.array(best_vp["position"])
    dest_look_at = target_3d

    start_pos = np.array(client.camera.position)
    start_wxyz = np.array(client.camera.wxyz)
    dest_wxyz = look_at_wxyz(dest_pos, dest_look_at, up=up)

    n_frames = int(FLY_DURATION * FLY_FPS)
    frame_dt = FLY_DURATION / n_frames

    for i in range(n_frames + 1):
        t = i / n_frames
        if t < 0.5:
            eased = 4 * t * t * t
        else:
            eased = 1 - (-2 * t + 2) ** 3 / 2
        pos = start_pos + (dest_pos - start_pos) * eased
        wxyz = _slerp(start_wxyz, dest_wxyz, eased)
        with client.atomic():
            client.camera.position = tuple(pos)
            client.camera.wxyz = tuple(wxyz)
        await asyncio.sleep(frame_dt)

    with client.atomic():
        client.camera.wxyz = tuple(dest_wxyz)
        client.camera.up_direction = tuple(up)

    # --- Step 6: Pulse the marker ---
    if server is not None:
        n_pulse_frames = int(PULSE_DURATION * PULSE_FPS)
        pulse_dt = PULSE_DURATION / n_pulse_frames
        for i in range(n_pulse_frames):
            phase = (i / n_pulse_frames) * PULSE_DURATION * 2.5
            scale = 1.0 + 0.6 * abs(np.sin(phase * np.pi))
            server.scene.add_icosphere(
                name="/finder/marker",
                radius=marker_radius * scale,
                color=MARKER_COLOR_PRIMARY,
                position=tuple(target_3d),
            )
            await asyncio.sleep(pulse_dt)

        server.scene.add_icosphere(
            name="/finder/marker",
            radius=marker_radius * 1.2,
            color=MARKER_COLOR_PRIMARY,
            position=tuple(target_3d),
        )

    # --- Done ---
    views_msg = f" (seen in {n_views_found} views)" if n_views_found > 1 else ""
    await send_result(f"Found: {description}{views_msg}")
