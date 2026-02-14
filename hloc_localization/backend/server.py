"""
FastAPI server for hloc localization.

Receives frames from clients, calls Modal for localization, returns 6DoF poses.

Usage: python hloc_localization/server.py [--port 8090]

Endpoints:
  POST /reference/build   — Upload video, build reference map
  GET  /reference/status   — Check reference map status
  POST /localize           — Upload single frame, get 6DoF pose
  WS   /stream             — Stream frames in, stream poses out
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
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="hloc Localization Server")
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
        self.building: bool = False


state = ServerState()


def _load_reference(name: str) -> bytes | None:
    """Load a reference tar from disk."""
    tar_path = REF_DIR / name / "reference.tar.gz"
    if tar_path.exists():
        return tar_path.read_bytes()
    return None


@app.on_event("startup")
async def startup():
    # Auto-load first available reference
    if REF_DIR.exists():
        for d in sorted(REF_DIR.iterdir()):
            tar_path = d / "reference.tar.gz"
            if tar_path.exists():
                state.reference_tar = tar_path.read_bytes()
                state.reference_name = d.name
                print(f"Loaded reference: {d.name} ({len(state.reference_tar) / 1024 / 1024:.1f} MB)")
                break


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
        "building": state.building,
        "available": refs,
    }


@app.post("/reference/select/{name}")
async def select_reference(name: str):
    tar_bytes = _load_reference(name)
    if tar_bytes is None:
        raise HTTPException(404, f"Reference '{name}' not found")
    state.reference_tar = tar_bytes
    state.reference_name = name
    return {"status": "ok", "name": name}


@app.post("/reference/build")
async def build_reference_endpoint(video: UploadFile = File(...), fps: int = 3):
    """Upload a video and build a reference map on Modal."""
    if state.building:
        raise HTTPException(409, "Already building a reference")

    state.building = True
    try:
        video_bytes = await video.read()
        video_name = pathlib.Path(video.filename).stem if video.filename else "upload"

        print(f"Building reference from {video.filename} ({len(video_bytes) / 1024 / 1024:.1f} MB)")

        # Import and call Modal function
        from hloc_localization.backend.app import build_reference

        with modal.app.run():
            result = build_reference.remote(video_bytes, target_fps=fps)

        # Save reference
        out_dir = REF_DIR / video_name
        out_dir.mkdir(parents=True, exist_ok=True)
        tar_path = out_dir / "reference.tar.gz"
        tar_path.write_bytes(result["tar"])

        # Activate it
        state.reference_tar = result["tar"]
        state.reference_name = video_name

        return {
            "status": "ok",
            "name": video_name,
            "num_frames": result["num_frames"],
            "num_registered": result["num_registered"],
            "num_points3d": result["num_points3d"],
            "size_mb": len(result["tar"]) / 1024 / 1024,
        }
    finally:
        state.building = False


@app.post("/localize")
async def localize_endpoint(image: UploadFile = File(...)):
    """Localize a single frame against the active reference map."""
    if state.reference_tar is None:
        raise HTTPException(400, "No reference map loaded. Build or select one first.")

    image_bytes = await image.read()

    from hloc_localization.backend.app import localize_frame

    with modal.app.run():
        result = localize_frame.remote(image_bytes, state.reference_tar)

    return JSONResponse(result)


@app.websocket("/stream")
async def stream_localize(websocket: WebSocket):
    """
    Stream frames in, stream poses out.

    Client sends binary JPEG frames.
    Server responds with JSON pose dicts.
    """
    await websocket.accept()

    if state.reference_tar is None:
        await websocket.send_json({"error": "No reference map loaded"})
        await websocket.close()
        return

    from hloc_localization.backend.app import localize_frame

    try:
        while True:
            # Receive binary frame
            data = await websocket.receive_bytes()

            t0 = time.time()
            with modal.app.run():
                result = localize_frame.remote(data, state.reference_tar)
            result["latency_ms"] = (time.time() - t0) * 1000

            await websocket.send_json(result)

    except WebSocketDisconnect:
        pass


def main():
    parser = argparse.ArgumentParser(description="hloc Localization Server")
    parser.add_argument("--port", type=int, default=8090)
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

    print(f"Starting hloc localization server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
