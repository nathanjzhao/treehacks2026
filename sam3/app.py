"""
SAM3 on Modal — Segment Anything with Concepts (Meta, 2025).

Runs SAM3 video/image segmentation on an A100. Supports:
  - Batch video processing with text/visual prompts
  - Live WebSocket streaming for real-time segmentation

Deploy:  modal deploy sam3/app.py
Dev:     modal serve sam3/app.py
Video:   modal run sam3/app.py --video-path ~/video.mp4 --prompt "person"
"""

import pathlib

import modal

app = modal.App("sam3")

cuda_version = "12.6.3"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

_hf_secret = modal.Secret.from_name("huggingface")

sam3_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.7.0",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "numpy<2",
        "Pillow",
        "huggingface_hub",
        "opencv-python",
        "tqdm",
        "timm>=1.0.17",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "typing_extensions",
        "scipy",
        "matplotlib",
        "psutil",
        "decord",
        "einops",
        "cython",
        "pycocotools",
        "fastapi",
        "uvicorn",
        "websockets",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /opt/sam3",
        "cd /opt/sam3 && pip install -e .",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
    })
    # Pre-download model weights into the image
    .run_commands(
        "python -c \""
        "from sam3.model_builder import build_sam3_image_model; "
        "model = build_sam3_image_model(); "
        "print('SAM3 image model downloaded')\"",
        gpu="any",
        secrets=[_hf_secret],
    )
)

with sam3_image.imports():
    import os
    import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_frames(video_bytes: bytes, tmpdir: str, target_fps: int):
    """Extract frames from video bytes at target FPS. Returns (image_dir, image_paths, source_fps)."""
    import cv2

    video_path = os.path.join(tmpdir, "input.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    images_dir = os.path.join(tmpdir, "images")
    os.makedirs(images_dir)

    vs = cv2.VideoCapture(video_path)
    source_fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / target_fps))

    image_paths = []
    count = 0
    frame_num = 0
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            path = os.path.join(images_dir, f"{frame_num:06d}.jpg")
            cv2.imwrite(path, frame)
            image_paths.append(path)
            frame_num += 1
    vs.release()
    print(f"Extracted {len(image_paths)} frames from {count} total (interval={frame_interval})")
    return images_dir, image_paths, source_fps


def _overlay_masks(frame, masks, obj_ids, alpha=0.45):
    """Overlay colored segmentation masks onto a BGR frame."""
    import cv2
    import numpy as np

    # Distinct colors for up to 20 objects
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
        (200, 200, 200), (100, 100, 100), (50, 200, 50), (200, 50, 200),
    ]

    overlay = frame.copy()
    for mask, obj_id in zip(masks, obj_ids):
        color = COLORS[obj_id % len(COLORS)]
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask

        # Resize mask to frame size if needed
        if mask_bool.shape[:2] != frame.shape[:2]:
            mask_bool = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0])) > 0

        # Color overlay
        for c in range(3):
            overlay[:, :, c] = np.where(mask_bool, color[c], overlay[:, :, c])

    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw contours
    for mask, obj_id in zip(masks, obj_ids):
        color = COLORS[obj_id % len(COLORS)]
        mask_u8 = mask.astype(np.uint8) if mask.dtype == bool else mask
        if mask_u8.shape[:2] != frame.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (frame.shape[1], frame.shape[0]))
        mask_u8 = (mask_u8 * 255).astype(np.uint8) if mask_u8.max() <= 1 else mask_u8
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, color, 2)

    return blended


# ---------------------------------------------------------------------------
# Batch video segmentation
# ---------------------------------------------------------------------------

@app.function(
    image=sam3_image,
    gpu="A100",
    timeout=1800,
    memory=32768,
    secrets=[_hf_secret],
)
def predict_video(
    video_bytes: bytes,
    text_prompt: str = "object",
    target_fps: int = 2,
    points_json: str | None = None,
    boxes_json: str | None = None,
) -> dict:
    """
    Run SAM3 on a video. Segments and tracks objects, returns annotated MP4.

    Args:
        video_bytes: Raw video file content.
        text_prompt: Text description of objects to segment (e.g. "person in red").
        target_fps: Frames to extract per second (default: 2).
        points_json: Optional JSON — [[x, y], ...] normalized coords with labels.
        boxes_json: Optional JSON — [[x, y, w, h], ...] normalized bounding boxes.
    """
    import json
    import tempfile

    import cv2
    import numpy as np

    sys.path.insert(0, "/opt/sam3")
    from sam3.model_builder import build_sam3_video_predictor

    print(f"Building SAM3 video predictor...")
    predictor = build_sam3_video_predictor(gpus_to_use=[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir, image_paths, source_fps = _extract_frames(video_bytes, tmpdir, target_fps)

        if len(image_paths) == 0:
            raise ValueError("No frames extracted from video")

        # Start video session with the JPEG folder
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": images_dir,
        })
        session_id = response["session_id"]
        print(f"Session started: {session_id}")

        # Add text prompt on frame 0
        prompt_request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt,
        }

        # Add optional geometric prompts
        if points_json:
            data = json.loads(points_json)
            prompt_request["points"] = data.get("points", [])
            prompt_request["point_labels"] = data.get("labels", [1] * len(prompt_request["points"]))

        if boxes_json:
            data = json.loads(boxes_json)
            prompt_request["bounding_boxes"] = data if isinstance(data, list) else data.get("boxes", [])
            prompt_request["bounding_box_labels"] = [1] * len(prompt_request["bounding_boxes"])

        response = predictor.handle_request(prompt_request)
        print(f"Prompt added. Initial detections: {response.get('outputs', {})}")

        # Propagate through video
        print("Propagating segmentation through video...")
        frame_results = {}
        for result in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
        }):
            frame_idx = result["frame_index"]
            output = result["outputs"]
            frame_results[frame_idx] = output
            n_objs = len(output.get("out_obj_ids", []))
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}: {n_objs} objects")

        print(f"Propagation done. {len(frame_results)} frames processed.")

        # Build annotated video
        sample_frame = cv2.imread(image_paths[0])
        h, w = sample_frame.shape[:2]
        out_fps = min(target_fps, source_fps)

        out_path = os.path.join(tmpdir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))

        total_objects = set()
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)
            if i in frame_results:
                output = frame_results[i]
                obj_ids = output.get("out_obj_ids", np.array([]))
                masks = output.get("out_binary_masks", np.array([]))
                if len(obj_ids) > 0:
                    total_objects.update(obj_ids.tolist())
                if len(masks) > 0 and len(obj_ids) > 0:
                    frame = _overlay_masks(frame, masks, obj_ids.tolist())
            writer.write(frame)

        writer.release()

        # Re-encode with ffmpeg for broader compatibility
        final_path = os.path.join(tmpdir, "output.mp4")
        os.system(f'ffmpeg -y -i {out_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p {final_path} 2>/dev/null')
        if not os.path.exists(final_path):
            final_path = out_path  # fallback

        with open(final_path, "rb") as f:
            video_out = f.read()

        # Close session
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    print(f"Output: {len(video_out) / 1024 / 1024:.1f} MB, {len(total_objects)} unique objects tracked")
    return {
        "video": video_out,
        "num_frames": len(image_paths),
        "num_objects": len(total_objects),
        "source_fps": source_fps,
    }


# ---------------------------------------------------------------------------
# Live WebSocket streaming
# ---------------------------------------------------------------------------

@app.function(
    image=sam3_image,
    gpu="A100",
    timeout=3600,
    memory=32768,
    secrets=[_hf_secret],
    scaledown_window=120,
)
@modal.asgi_app()
def stream_segment():
    """FastAPI app with WebSocket for live frame-by-frame segmentation."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    sys.path.insert(0, "/opt/sam3")

    web = FastAPI()

    # Simple test page
    TEST_HTML = """<!DOCTYPE html>
<html><head><title>SAM3 Live</title>
<style>
  body { background: #111; color: #eee; font-family: system-ui; display: flex;
         flex-direction: column; align-items: center; padding: 20px; }
  .container { display: flex; gap: 20px; margin-top: 20px; }
  video, canvas { border: 2px solid #444; border-radius: 8px; }
  input, button { padding: 8px 16px; font-size: 16px; border-radius: 4px; border: 1px solid #555;
                  background: #222; color: #eee; }
  button { background: #2563eb; border: none; cursor: pointer; }
  button:hover { background: #1d4ed8; }
  #status { margin-top: 10px; color: #888; }
</style>
</head><body>
<h1>SAM3 Live Segmentation</h1>
<div>
  <input id="prompt" type="text" value="person" placeholder="Text prompt..." />
  <button onclick="start()">Start</button>
  <button onclick="stop()">Stop</button>
</div>
<div class="container">
  <div><h3>Camera</h3><video id="cam" width="640" height="480" autoplay muted></video></div>
  <div><h3>Segmented</h3><canvas id="out" width="640" height="480"></canvas></div>
</div>
<div id="status">Click Start to begin</div>
<script>
let ws, running = false, stream;
const cam = document.getElementById('cam');
const out = document.getElementById('out');
const ctx = out.getContext('2d');
const status = document.getElementById('status');

async function start() {
  if (running) return;
  stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
  cam.srcObject = stream;
  const prompt = document.getElementById('prompt').value;
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws?prompt=${encodeURIComponent(prompt)}`);
  ws.binaryType = 'arraybuffer';
  running = true;
  status.textContent = 'Connecting...';

  ws.onopen = () => { status.textContent = 'Connected — streaming'; sendFrame(); };
  ws.onmessage = (e) => {
    const blob = new Blob([e.data], {type: 'image/jpeg'});
    const img = new Image();
    img.onload = () => { ctx.drawImage(img, 0, 0, 640, 480); URL.revokeObjectURL(img.src); if (running) sendFrame(); };
    img.src = URL.createObjectURL(blob);
  };
  ws.onclose = () => { status.textContent = 'Disconnected'; running = false; };
  ws.onerror = () => { status.textContent = 'Error'; running = false; };
}

function sendFrame() {
  if (!running || ws.readyState !== 1) return;
  const c = document.createElement('canvas');
  c.width = 640; c.height = 480;
  c.getContext('2d').drawImage(cam, 0, 0, 640, 480);
  c.toBlob(b => { if (b && ws.readyState === 1) ws.send(b); }, 'image/jpeg', 0.7);
}

function stop() {
  running = false;
  if (ws) ws.close();
  if (stream) stream.getTracks().forEach(t => t.stop());
  status.textContent = 'Stopped';
}
</script>
</body></html>"""

    @web.get("/")
    async def index():
        return HTMLResponse(TEST_HTML)

    # Load model once at container startup (shared across connections)
    import cv2
    import numpy as np
    from PIL import Image

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3 image model for streaming...")
    _model = build_sam3_image_model()
    _processor = Sam3Processor(_model, confidence_threshold=0.3)
    print("Model loaded. Ready for WebSocket connections.")

    @web.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, prompt: str = "object"):
        await websocket.accept()
        print(f"WebSocket connected. Prompt: '{prompt}'")

        try:
            while True:
                data = await websocket.receive_bytes()

                # Decode JPEG frame
                arr = np.frombuffer(data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Run SAM3 image segmentation
                state = _processor.set_image(pil_image)
                state = _processor.set_text_prompt(state=state, prompt=prompt)

                masks = state.get("masks")
                if masks is not None and len(masks) > 0:
                    import torch  # noqa: already loaded
                    if isinstance(masks, torch.Tensor):
                        masks_np = masks.cpu().numpy()
                    else:
                        masks_np = np.array(masks)

                    obj_ids = list(range(len(masks_np)))
                    frame_bgr = _overlay_masks(frame_bgr, masks_np, obj_ids)

                # Encode annotated frame as JPEG
                _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                await websocket.send_bytes(buf.tobytes())

        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            print(f"WebSocket error: {e}")
            import traceback
            traceback.print_exc()

    return web


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    video_path: str,
    prompt: str = "object",
    fps: int = 2,
    output_dir: str = "",
    points: str = "",
    boxes: str = "",
):
    """
    Run SAM3 on local video file(s). Supports glob patterns for parallel batch processing.

    Args:
        video_path: Path to video file, or glob pattern (e.g. "~/Downloads/IMG_47*.MOV").
        prompt: Text prompt for segmentation (e.g. "person in red shirt").
        fps: Frames to extract per second.
        output_dir: Directory to save outputs (default: next to input).
        points: Path to JSON file with points and labels.
        boxes: Path to JSON file with bounding boxes.
    """
    import glob as globmod

    # Resolve glob pattern to list of files
    expanded = str(pathlib.Path(video_path).expanduser())
    paths = sorted(globmod.glob(expanded))
    if not paths:
        paths = [expanded]  # treat as single file

    video_files = [pathlib.Path(p).resolve() for p in paths]
    print(f"Found {len(video_files)} video(s)")

    out_dir = None
    if output_dir:
        out_dir = pathlib.Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {out_dir}")

    points_json = None
    if points:
        points_json = pathlib.Path(points).expanduser().resolve().read_text()

    boxes_json = None
    if boxes:
        boxes_json = pathlib.Path(boxes).expanduser().resolve().read_text()

    # Launch all videos in parallel using starmap
    print(f"Launching {len(video_files)} parallel jobs (fps={fps}, prompt='{prompt}')...")

    inputs = []
    for p in video_files:
        print(f"  -> {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
        inputs.append((p.read_bytes(), prompt, fps, points_json, boxes_json))

    for p, result in zip(video_files, predict_video.starmap(inputs)):
        if out_dir:
            out = out_dir / (p.stem + "_sam3.mp4")
        else:
            out = p.with_name(p.stem + "_sam3.mp4")
        out.write_bytes(result["video"])
        print(f"Wrote {out.name} ({len(result['video']) / 1024 / 1024:.1f} MB) — {result['num_frames']} frames, {result['num_objects']} objects")
