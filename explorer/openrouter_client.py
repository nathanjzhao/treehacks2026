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

    Args:
        query: User's object query (e.g., "show me the chair")
        renders: List of numpy RGB arrays (H, W, 3)
        labels: Labels for each view (e.g., "az0_el25")

    Returns:
        Dict with keys: view_index, pixel_x, pixel_y, description, confidence
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

    system_prompt = (
        "You are an expert at finding objects in 3D point cloud renders of indoor scenes. "
        "These are rendered from a sparse point cloud, so objects may look rough, have gaps, "
        "or appear as scattered dots — this is normal. Use shape, color, and context clues.\n\n"
        "Reply with ONLY a JSON object, no other text.\n\n"
        "Format: "
        '{"view_index": <int>, "pixel_x": <int 0-640>, "pixel_y": <int 0-480>, '
        '"description": "<brief>", "confidence": <float 0-1>}\n\n'
        "Pick the view where the object is most visible. Point to its approximate center.\n"
        "Be generous with confidence — if you can see anything that could be the object, "
        "set confidence >= 0.3. Only use confidence 0.0 if the object is truly absent from ALL views."
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
        "max_tokens": 500,
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
            print(f"[OpenRouter] Raw LLM response: {content[:500]}", flush=True)
            break
        except Exception as e:
            last_error = e
            print(f"[OpenRouter] Attempt {attempt+1} failed: {e}", flush=True)
            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))

    if content is None:
        print(f"[OpenRouter] All attempts failed: {last_error}", flush=True)
        return {
            "view_index": 0,
            "pixel_x": 320,
            "pixel_y": 240,
            "description": f"API error: {last_error}",
            "confidence": 0.0,
        }

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strip markdown code blocks
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Extract first JSON object from anywhere in the response
    match = re.search(r"\{[^{}]*\"view_index\"[^{}]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"[OpenRouter] Failed to parse JSON: {content[:300]}", flush=True)
    return {
        "view_index": 0,
        "pixel_x": 320,
        "pixel_y": 240,
        "description": f"LLM response was not valid JSON: {content[:100]}",
        "confidence": 0.0,
    }
