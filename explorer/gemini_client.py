"""Gemini vision client — uses native Gemini API for box_3d spatial detection,
falls back to box_2d + raycasting via OpenRouter."""

import asyncio
import base64
import io
import json
import os
import re

import httpx
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

# --- OpenRouter fallback config ---
OR_API_KEY = "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"
OR_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_MODEL = "google/gemini-3-flash-preview"

# --- Native Gemini config ---
GEMINI_API_KEY = "AIzaSyBCVFzvkIxQ4K0l5ZxudjLyuJBTful6lv4"
GEMINI_MODEL = "gemini-3.0-flash-preview"  # Gemini 3 Flash


def _numpy_to_jpeg_bytes(img_array: np.ndarray, quality: int = 85) -> bytes:
    img = Image.fromarray(img_array)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _numpy_to_base64_jpeg(img_array: np.ndarray, quality: int = 85) -> str:
    return base64.b64encode(_numpy_to_jpeg_bytes(img_array, quality)).decode("utf-8")


def _parse_json(content: str) -> dict | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Primary: Native Gemini API with box_3d
# ---------------------------------------------------------------------------

async def _query_box_3d(
    query: str,
    renders: list[np.ndarray],
    labels: list[str],
) -> dict | None:
    """Try native Gemini API for box_3d spatial detection.

    Returns dict with 'box_3d' key if successful, None otherwise.
    box_3d = [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Send the best 3 views (less noise for 3D reasoning)
    n_views = min(3, len(renders))
    content_parts = []
    for i in range(n_views):
        content_parts.append(f"View {i} ({labels[i]}):")
        jpeg_bytes = _numpy_to_jpeg_bytes(renders[i])
        content_parts.append(
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        )

    content_parts.append(
        f"Detect the 3D bounding box of this object: {query}\n"
        "Output a JSON object with 'label' containing the object name, "
        "'description' with a brief description, "
        "and 'box_3d' as [x_center, y_center, z_center, x_size, y_size, z_size, "
        "roll, pitch, yaw] in the scene's coordinate system."
    )

    config = types.GenerateContentConfig(
        temperature=0.5,
        response_mime_type="application/json",
    )

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=content_parts,
            config=config,
        )
        raw = response.text
        print(f"[Gemini-3D] Raw box_3d response: {raw[:600]}", flush=True)

        parsed = json.loads(raw)
        box_3d = parsed.get("box_3d")
        if box_3d and len(box_3d) >= 3:
            print(f"[Gemini-3D] Got box_3d: {box_3d}", flush=True)
            return parsed
        else:
            print(f"[Gemini-3D] No valid box_3d in response", flush=True)
            return None
    except Exception as e:
        print(f"[Gemini-3D] box_3d failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Fallback: OpenRouter with box_2d
# ---------------------------------------------------------------------------

async def _query_box_2d(
    query: str,
    renders: list[np.ndarray],
    labels: list[str],
) -> dict:
    """OpenRouter fallback using box_2d format."""
    img_h, img_w = renders[0].shape[:2]

    image_content = []
    for i, (render, label) in enumerate(zip(renders, labels)):
        b64 = _numpy_to_base64_jpeg(render)
        image_content.append({"type": "text", "text": f"View {i} ({label}):"})
        image_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    system_prompt = (
        "You are an expert at finding objects in 3D point cloud renders of indoor scenes. "
        "These are rendered from a sparse point cloud, so objects may look rough, have gaps, "
        "or appear as scattered dots. Use shape, color, and spatial context.\n\n"
        "You are given multiple camera views of the SAME scene. The object may be visible "
        "in several views from different angles.\n\n"
        "For EACH view where the object is visible, provide a bounding box using the "
        "box_2d format: [y_min, x_min, y_max, x_max] with coordinates normalized to "
        "a 0-1000 scale (as if the image were 1000x1000 pixels).\n\n"
        "Reply with ONLY a JSON object:\n"
        "{\n"
        '  "description": "<what you found>",\n'
        '  "detections": [\n'
        '    {"view_index": 0, "box_2d": [y_min, x_min, y_max, x_max], "confidence": 0.9},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "IMPORTANT: box_2d uses [y_min, x_min, y_max, x_max] order, normalized 0-1000.\n"
        "Provide detections for EVERY view where you can see the object — "
        "more views = better 3D localization.\n"
        "If not found in any view: empty detections list and description \"not found\"."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                *image_content,
                {"type": "text", "text": f"\nFind this object: {query}"},
            ],
        },
    ]

    payload = {
        "model": OR_MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 2000,
    }

    content = None
    last_error = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                response = await http_client.post(
                    OR_API_URL,
                    headers={
                        "Authorization": f"Bearer {OR_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

            content = result["choices"][0]["message"]["content"].strip()
            print(f"[Gemini-2D] Raw response: {content[:800]}", flush=True)
            break
        except Exception as e:
            last_error = e
            print(f"[Gemini-2D] Attempt {attempt + 1} failed: {e}", flush=True)
            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))

    if content is None:
        return {"description": f"API error: {last_error}", "detections": []}

    parsed = _parse_json(content)
    if parsed is None:
        return {"description": "Failed to parse response", "detections": []}

    detections = parsed.get("detections", [])
    valid = []
    for det in detections:
        box_2d = det.get("box_2d")
        vi = det.get("view_index")
        if box_2d and len(box_2d) == 4 and vi is not None:
            y_min, x_min, y_max, x_max = box_2d
            pixel_bbox = [
                x_min / 1000.0 * img_w,
                y_min / 1000.0 * img_h,
                x_max / 1000.0 * img_w,
                y_max / 1000.0 * img_h,
            ]
            valid.append({
                "view_index": int(vi),
                "bbox": pixel_bbox,
                "confidence": float(det.get("confidence", 0.5)),
            })

    parsed["detections"] = valid
    print(f"[Gemini-2D] {len(valid)} valid detections", flush=True)
    return parsed


# ---------------------------------------------------------------------------
# Public API — tries box_3d first, falls back to box_2d
# ---------------------------------------------------------------------------

async def query_object_location(
    query: str,
    renders: list[np.ndarray],
    labels: list[str],
) -> dict:
    """Query Gemini for object location. Tries box_3d first, falls back to box_2d.

    Returns dict with:
      - description: str
      - detections: list of {view_index, bbox, confidence} (for box_2d path)
      - box_3d: [x,y,z, sx,sy,sz, roll,pitch,yaw] (if box_3d succeeded)
      - target_3d: [x,y,z] center position (if box_3d succeeded)
    """
    # box_3d returns coordinates in Gemini's own frame (meters), not our
    # point cloud's world frame — no reliable way to map them.  Use box_2d
    # via OpenRouter (Gemini 3 Flash) which works well with raycasting.
    return await _query_box_2d(query, renders, labels)


# ---------------------------------------------------------------------------
# Up direction detection — ask Gemini which orientation looks right-side up
# ---------------------------------------------------------------------------

async def detect_up_direction(
    renders: list[np.ndarray],
    labels: list[str],
) -> np.ndarray:
    """Use Gemini to detect the scene's 'up' direction.

    Sends 3 renders to Gemini (rendered with different up-vector candidates)
    and asks which looks right-side up.

    Primary: OpenRouter. Fallback: native Gemini API.

    Args:
        renders: list of 3 images, one per candidate up vector
        labels: ["A (+Z up)", "B (+Y up)", "C (-Y up)"]

    Returns:
        The up vector as np.ndarray (e.g. [0,0,1] or [0,1,0])
    """
    UP_MAP = {"A": np.array([0.0, 0.0, 1.0]),
              "B": np.array([0.0, 1.0, 0.0]),
              "C": np.array([0.0, -1.0, 0.0])}

    prompt = (
        "These are 3 renders of the same indoor 3D scene from the same camera position, "
        "but with different 'up' orientations. Which option shows the scene RIGHT-SIDE UP "
        "(gravity pointing down, floor at bottom, ceiling at top)?\n"
        "Reply with ONLY the letter: A, B, or C."
    )

    def _parse_up(answer: str) -> np.ndarray | None:
        for key in UP_MAP:
            if key in answer.upper():
                print(f"[Gemini-Up] Detected up direction: {key} = {UP_MAP[key]}", flush=True)
                return UP_MAP[key]
        return None

    # --- Primary: OpenRouter ---
    try:
        image_content = []
        for render, label in zip(renders, labels):
            b64 = _numpy_to_base64_jpeg(render)
            image_content.append({"type": "text", "text": f"Option {label}:"})
            image_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        image_content.append({"type": "text", "text": prompt})

        payload = {
            "model": OR_MODEL,
            "messages": [{"role": "user", "content": image_content}],
            "temperature": 0.0,
            "max_tokens": 10,
        }

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                OR_API_URL,
                headers={
                    "Authorization": f"Bearer {OR_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()

        print(f"[Gemini-Up] OpenRouter response: {answer}", flush=True)
        result = _parse_up(answer)
        if result is not None:
            return result
    except Exception as e:
        print(f"[Gemini-Up] OpenRouter failed: {e}, trying native API...", flush=True)

    # --- Fallback: Native Gemini API ---
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        content_parts = []
        for render, label in zip(renders, labels):
            content_parts.append(f"Option {label}:")
            jpeg_bytes = _numpy_to_jpeg_bytes(render)
            content_parts.append(
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
            )
        content_parts.append(prompt)
        config = types.GenerateContentConfig(temperature=0.0)

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=content_parts,
            config=config,
        )
        answer = response.text.strip()
        print(f"[Gemini-Up] Native API response: {answer}", flush=True)
        result = _parse_up(answer)
        if result is not None:
            return result
    except Exception as e:
        print(f"[Gemini-Up] Native API also failed: {e}", flush=True)

    print(f"[Gemini-Up] All attempts failed, defaulting to +Z", flush=True)
    return np.array([0.0, 0.0, 1.0])
