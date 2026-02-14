"""OpenRouter vision API wrapper for object finding."""

import asyncio
import base64
import io
import json
import re

import httpx
import numpy as np
from PIL import Image

API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4"


def numpy_to_base64_jpeg(img_array: np.ndarray, quality: int = 85) -> str:
    """Convert a numpy RGBA/RGB array to a base64-encoded JPEG string."""
    img = Image.fromarray(img_array)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def query_object_location(
    query: str,
    renders: list[np.ndarray],
    labels: list[str],
) -> dict:
    """Send rendered views to a vision LLM and ask it to locate an object.

    Returns dict with keys: description, detections (list of per-view bboxes).
    Each detection has: view_index, bbox [x1, y1, x2, y2], confidence.
    """
    image_content = []
    for i, (render, label) in enumerate(zip(renders, labels)):
        b64 = numpy_to_base64_jpeg(render)
        image_content.append({
            "type": "text",
            "text": f"View {i} ({label}):",
        })
        image_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
            },
        })

    img_h, img_w = renders[0].shape[:2]

    system_prompt = (
        "You are an expert at finding objects in 3D point cloud renders of indoor scenes. "
        "These are rendered from a sparse point cloud, so objects may look rough, have gaps, "
        "or appear as scattered dots. Use shape, color, and spatial context.\n\n"
        "You are given multiple camera views of the SAME scene. The object may be visible "
        "in several views from different angles.\n\n"
        "For EACH view where the object is visible, provide a bounding box around it.\n\n"
        "Reply with ONLY a JSON object:\n"
        "{\n"
        '  "description": "<what you found>",\n'
        '  "detections": [\n'
        f'    {{"view_index": 0, "bbox": [x1, y1, x2, y2], "confidence": 0.9}},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        f"bbox coordinates: x in [0, {img_w}], y in [0, {img_h}]. "
        "Draw a tight box around the object.\n"
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
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1000,
    }

    content = None
    last_error = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                response = await http_client.post(
                    API_URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

            content = result["choices"][0]["message"]["content"].strip()
            print(f"[OpenRouter] Raw LLM response: {content[:800]}", flush=True)
            break
        except Exception as e:
            last_error = e
            print(f"[OpenRouter] Attempt {attempt+1} failed: {e}", flush=True)
            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))

    if content is None:
        print(f"[OpenRouter] All attempts failed: {last_error}", flush=True)
        return {"description": f"API error: {last_error}", "detections": []}

    # Parse JSON response
    parsed = _parse_json(content)
    if parsed is None:
        print(f"[OpenRouter] Failed to parse JSON: {content[:300]}", flush=True)
        return {"description": "Failed to parse LLM response", "detections": []}

    # Validate detections
    detections = parsed.get("detections", [])
    valid = []
    for det in detections:
        bbox = det.get("bbox")
        vi = det.get("view_index")
        if bbox and len(bbox) == 4 and vi is not None:
            valid.append({
                "view_index": int(vi),
                "bbox": [float(b) for b in bbox],
                "confidence": float(det.get("confidence", 0.5)),
            })

    parsed["detections"] = valid
    print(f"[OpenRouter] {len(valid)} valid detections", flush=True)
    return parsed


def _parse_json(content: str) -> dict | None:
    """Try multiple strategies to extract JSON from LLM output."""
    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strip markdown code blocks
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Find outermost JSON object
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
