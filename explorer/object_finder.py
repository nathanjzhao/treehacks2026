"""Pipeline: render -> LLM -> unproject -> animate."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Callable, Awaitable

import numpy as np
import viser
from PIL import Image

from camera_utils import (
    look_at_wxyz,
    interpolate_camera,
    unproject_pixel_to_3d,
)
from openrouter_client import query_object_location

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_FOV = 1.2  # radians (~69 degrees) - wider for indoor scenes
FLY_DURATION = 2.0  # seconds
FLY_FPS = 60
NUM_VIEWS = 6  # number of diverse camera views to use
MAX_LLM_RETRIES = 2  # retry LLM if confidence is very low


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
    centroid: np.ndarray,
    send_status: Callable[[str], Awaitable[None]],
    send_result: Callable[[str], Awaitable[None]],
    camera_positions: list[np.ndarray] | None = None,
) -> None:
    """Full object-finding pipeline.

    1. Render views from VGGT camera positions (inside the scene)
    2. Send to LLM for identification
    3. Unproject pixel to 3D
    4. Smoothly fly camera to target
    """
    loop = asyncio.get_event_loop()

    # --- Step 1: Capture multi-view renders ---
    await send_status("Scanning the scene...")

    debug_dir = Path(__file__).parent / "debug_renders"
    debug_dir.mkdir(exist_ok=True)

    # Build viewpoints from VGGT camera positions
    if camera_positions and len(camera_positions) >= NUM_VIEWS:
        indices = _select_diverse_cameras(camera_positions, NUM_VIEWS)
        viewpoints = []
        for i, idx in enumerate(indices):
            pos = camera_positions[idx]
            wxyz = look_at_wxyz(pos, centroid)
            viewpoints.append({
                "position": pos,
                "wxyz": wxyz,
                "label": f"cam{idx}",
            })
        print(f"[Pipeline] Using {NUM_VIEWS} diverse VGGT cameras: {indices}", flush=True)
    else:
        # Fallback: use current camera view only
        print("[Pipeline] No VGGT cameras available, using current view only", flush=True)
        viewpoints = [{
            "position": np.array(client.camera.position),
            "wxyz": np.array(client.camera.wxyz),
            "label": "current",
        }]

    renders = []
    labels = []
    ts = int(time.time())

    for vp in viewpoints:
        print(f"[Render] {vp['label']}: pos={vp['position']}", flush=True)

        # Small delay between renders to let viser client settle
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
            print(f"[Render] WARNING: {vp['label']} returned invalid array shape {arr.shape}", flush=True)
            continue
        mean_val = arr.mean()
        print(f"[Render] {vp['label']}: shape={arr.shape}, mean={mean_val:.1f}", flush=True)
        if mean_val < 5 or mean_val > 250:
            print(f"[Render] WARNING: {vp['label']} looks blank (mean={mean_val:.1f}), skipping", flush=True)
            continue

        renders.append(arr)
        labels.append(vp["label"])

        # Save debug render with timestamp
        img = Image.fromarray(arr)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(debug_dir / f"{ts}_{vp['label']}.jpg")

    if not renders:
        await send_result("Failed to capture scene renders. Try again.")
        return

    print(f"[Pipeline] Rendered {len(renders)} valid views, sending to LLM...", flush=True)

    # --- Step 2: Query LLM (with retry) ---
    await send_status(f"Asking AI to find '{query}'...")

    llm_result = None
    for attempt in range(MAX_LLM_RETRIES):
        result = await query_object_location(query, renders, labels)
        confidence = result.get("confidence", 0.0)
        description = result.get("description", "No description")
        print(f"[Pipeline] LLM attempt {attempt+1}: confidence={confidence}, desc={description}", flush=True)

        if confidence >= 0.1:
            llm_result = result
            break
        elif attempt < MAX_LLM_RETRIES - 1:
            print(f"[Pipeline] Low confidence, retrying...", flush=True)
            await send_status(f"Looking more carefully for '{query}'...")
            await asyncio.sleep(0.5)

    if llm_result is None:
        await send_result(
            f"Sorry, I couldn't find '{query}' in the scene. Try a different description."
        )
        return

    # --- Step 3: Unproject pixel to 3D ---
    await send_status("Found it! Flying to object...")

    view_idx = llm_result["view_index"]
    px = llm_result["pixel_x"]
    py = llm_result["pixel_y"]
    matched_vp = viewpoints[view_idx]

    target_3d = unproject_pixel_to_3d(
        px, py,
        RENDER_WIDTH, RENDER_HEIGHT,
        matched_vp["position"],
        matched_vp["wxyz"],
        RENDER_FOV,
        points,
    )

    # --- Step 4: Smooth camera fly-to ---
    bbox_extent = points.max(axis=0) - points.min(axis=0)
    radius = float(np.linalg.norm(bbox_extent)) * 0.8

    view_dir = matched_vp["position"] - target_3d
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
    fly_distance = radius * 0.15  # closer for indoor scenes
    dest_pos = target_3d + view_dir * fly_distance
    dest_look_at = target_3d

    start_pos = np.array(client.camera.position)
    start_look_at = np.array(client.camera.look_at)

    n_frames = int(FLY_DURATION * FLY_FPS)
    frame_dt = FLY_DURATION / n_frames

    for i in range(n_frames + 1):
        t = i / n_frames
        pos, look = interpolate_camera(start_pos, start_look_at, dest_pos, dest_look_at, t)

        with client.atomic():
            client.camera.position = tuple(pos)
            client.camera.look_at = tuple(look)

        await asyncio.sleep(frame_dt)

    # --- Done ---
    await send_result(f"Found: {llm_result.get('description', description)}")
