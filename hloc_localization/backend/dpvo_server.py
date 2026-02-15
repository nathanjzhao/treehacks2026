"""
FastAPI server for DPVO odometry pipeline.

Receives frames, runs DPVO on Modal for near-real-time tracking,
uses HLoc for anchor-frame localization.

Usage: python -m hloc_localization.backend.dpvo_server [--port 8091]

Endpoints:
  POST /anchor       — Upload frame + select reference, localize anchor with HLoc
  POST /odometry     — Upload video, run full DPVO pipeline
  WS   /stream       — Stream frames, get DPVO poses in real-time
  GET  /status       — Current session info
"""

import argparse
import asyncio
import json
import pathlib
import time

import cv2
import modal
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="DPVO Odometry Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
REF_DIR = DATA_DIR / "hloc_reference"


class ServerState:
    def __init__(self):
        self.reference_tar: bytes | None = None
        self.reference_name: str | None = None
        self.anchor_pose: dict | None = None
        self.anchor_frame_bytes: bytes | None = None
        self.session_active: bool = False
        self.pose_count: int = 0
        self.last_pose: dict | None = None
        self.calib: list[float] | None = None


state = ServerState()


def _load_reference(name: str) -> bytes | None:
    tar_path = REF_DIR / name / "reference.tar.gz"
    if tar_path.exists():
        return tar_path.read_bytes()
    return None


@app.on_event("startup")
async def startup():
    if REF_DIR.exists():
        for d in sorted(REF_DIR.iterdir()):
            tar_path = d / "reference.tar.gz"
            if tar_path.exists():
                state.reference_tar = tar_path.read_bytes()
                state.reference_name = d.name
                print(f"Loaded reference: {d.name} ({len(state.reference_tar) / 1024 / 1024:.1f} MB)")
                break


@app.get("/")
async def root():
    return {
        "service": "DPVO Odometry Server",
        "endpoints": {
            "GET /": "This page",
            "GET /status": "Current session info",
            "GET /reference/status": "List available references",
            "POST /reference/select/{name}": "Activate a reference map",
            "POST /anchor": "Localize anchor frame with HLoc",
            "POST /odometry": "Run full DPVO pipeline on a video",
            "WS /stream": "Stream frames for real-time DPVO tracking",
        },
        "active_reference": state.reference_name,
        "anchor_localized": state.anchor_pose is not None,
    }


@app.get("/status")
async def status():
    return {
        "reference_name": state.reference_name,
        "anchor_localized": state.anchor_pose is not None,
        "anchor_pose": state.anchor_pose,
        "session_active": state.session_active,
        "pose_count": state.pose_count,
        "last_pose": state.last_pose,
        "calib": state.calib,
    }


@app.get("/reference/status")
async def reference_status():
    refs = []
    if REF_DIR.exists():
        for d in sorted(REF_DIR.iterdir()):
            tar_path = d / "reference.tar.gz"
            if tar_path.exists():
                refs.append({
                    "name": d.name,
                    "size_mb": tar_path.stat().st_size / 1024 / 1024,
                    "active": d.name == state.reference_name,
                })
    return {
        "active_reference": state.reference_name,
        "available": refs,
    }


@app.post("/reference/select/{name}")
async def select_reference(name: str):
    tar_bytes = _load_reference(name)
    if tar_bytes is None:
        raise HTTPException(404, f"Reference '{name}' not found")
    state.reference_tar = tar_bytes
    state.reference_name = name
    # Reset anchor when reference changes
    state.anchor_pose = None
    return {"status": "ok", "name": name}


@app.post("/anchor")
async def localize_anchor(
    image: UploadFile = File(...),
    reference: str = Form(None),
):
    """Localize an anchor frame with HLoc to establish world coordinates."""
    if reference:
        tar_bytes = _load_reference(reference)
        if tar_bytes:
            state.reference_tar = tar_bytes
            state.reference_name = reference

    if state.reference_tar is None:
        raise HTTPException(400, "No reference map loaded. Select one first.")

    image_bytes = await image.read()

    from hloc_localization.backend.app import localize_frame

    t0 = time.time()
    result = localize_frame.remote(image_bytes, state.reference_tar)
    result["latency_ms"] = (time.time() - t0) * 1000

    if result.get("success"):
        state.anchor_pose = result
        state.anchor_frame_bytes = image_bytes
        print(f"Anchor localized: t=({result['tx']:.3f}, {result['ty']:.3f}, {result['tz']:.3f}), "
              f"inliers={result['num_inliers']}, latency={result['latency_ms']:.0f}ms")

    return JSONResponse(result)


@app.post("/odometry")
async def run_odometry(
    video: UploadFile = File(...),
    reference: str = Form(None),
    anchor_frame: int = Form(0),
    fps: int = Form(15),
    fx: float = Form(None),
    fy: float = Form(None),
    cx: float = Form(None),
    cy: float = Form(None),
):
    """Upload a video and run the full DPVO odometry pipeline on Modal."""
    if reference:
        tar_bytes = _load_reference(reference)
        if tar_bytes:
            state.reference_tar = tar_bytes
            state.reference_name = reference

    if state.reference_tar is None:
        raise HTTPException(400, "No reference map loaded.")

    video_bytes = await video.read()

    calib = None
    if fx is not None and fy is not None and cx is not None and cy is not None:
        calib = [fx, fy, cx, cy]
        state.calib = calib

    print(f"Running DPVO odometry on {video.filename} ({len(video_bytes) / 1024 / 1024:.1f} MB)")

    from hloc_localization.backend.dpvo_app import run_dpvo_odometry

    t0 = time.time()
    result = run_dpvo_odometry.remote(
        video_bytes,
        state.reference_tar,
        calib=calib,
        anchor_frame_idx=anchor_frame,
        target_fps=fps,
    )
    result["total_latency_ms"] = (time.time() - t0) * 1000

    if result.get("success"):
        state.anchor_pose = result.get("anchor_pose")
        state.pose_count = result.get("num_dpvo_poses", 0)
        if result.get("poses"):
            state.last_pose = result["poses"][-1]

    return JSONResponse(result)


@app.websocket("/stream")
async def stream_odometry(websocket: WebSocket):
    """
    Stream frames for real-time DPVO tracking.

    Protocol:
    1. Client sends JSON config: {type: "config", calib: [fx,fy,cx,cy], reference: "name"}
    2. Client sends binary JPEG frames
    3. Server responds with JSON pose dicts

    First frame is localized with HLoc (anchor), subsequent frames use DPVO.
    """
    await websocket.accept()

    if state.reference_tar is None:
        await websocket.send_json({"error": "No reference map loaded"})
        await websocket.close()
        return

    from hloc_localization.backend.app import localize_frame

    frame_idx = 0
    state.session_active = True

    try:
        while True:
            data = await websocket.receive()

            # Handle config messages
            if "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "config":
                    if msg.get("calib"):
                        state.calib = msg["calib"]
                    if msg.get("reference"):
                        tar_bytes = _load_reference(msg["reference"])
                        if tar_bytes:
                            state.reference_tar = tar_bytes
                            state.reference_name = msg["reference"]
                    await websocket.send_json({"type": "config_ack", "status": "ok"})
                    continue

            # Handle binary frame data
            frame_bytes = data.get("bytes", b"")
            if not frame_bytes:
                continue

            t0 = time.time()

            if state.anchor_pose is None:
                # First frame: localize with HLoc
                result = localize_frame.remote(frame_bytes, state.reference_tar)
                if result.get("success"):
                    state.anchor_pose = result
                    result["source"] = "hloc"
                    result["frame_idx"] = frame_idx
                    print(f"Stream anchor localized: inliers={result['num_inliers']}")
                result["latency_ms"] = (time.time() - t0) * 1000
                await websocket.send_json(result)
            else:
                # Subsequent frames: for now, still use HLoc
                # TODO: integrate streaming DPVO session once Modal supports
                # persistent GPU containers. For now, each frame is localized
                # independently with HLoc, but the /odometry endpoint uses
                # full DPVO batch processing.
                result = localize_frame.remote(frame_bytes, state.reference_tar)
                result["source"] = "hloc"
                result["frame_idx"] = frame_idx
                result["latency_ms"] = (time.time() - t0) * 1000

                if result.get("success"):
                    state.pose_count += 1
                    state.last_pose = result

                await websocket.send_json(result)

            frame_idx += 1

    except WebSocketDisconnect:
        pass
    finally:
        state.session_active = False


def main():
    parser = argparse.ArgumentParser(description="DPVO Odometry Server")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reference", help="Reference name to load on startup")
    args = parser.parse_args()

    if args.reference:
        tar_bytes = _load_reference(args.reference)
        if tar_bytes:
            state.reference_tar = tar_bytes
            state.reference_name = args.reference
            print(f"Pre-loaded reference: {args.reference}")
        else:
            print(f"Warning: reference '{args.reference}' not found")

    print(f"Starting DPVO odometry server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
