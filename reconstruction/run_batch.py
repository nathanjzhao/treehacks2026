"""
Run MapAnything on multiple videos in parallel via Modal.

Usage: modal run reconstruction/run_batch.py
"""

import pathlib
import time

import modal

# Re-use the existing app and its image/function
from reconstruction.app import app, predict_video

_data_dir = pathlib.Path(__file__).parent.parent / "data"

VIDEOS = sorted(_data_dir.glob("*.MOV")) + sorted(_data_dir.glob("*.mov"))


@app.local_entrypoint()
def batch(fps: int = 5, conf: float = 20.0):
    """Run MapAnything on 7 videos in parallel. Saves GLBs to examples/."""
    out_dir = pathlib.Path(__file__).parent / "examples"
    out_dir.mkdir(exist_ok=True)

    print(f"Launching {len(VIDEOS)} parallel MapAnything jobs (fps={fps}, conf={conf})")
    for v in VIDEOS:
        sz = v.stat().st_size / 1024 / 1024
        print(f"  {v.name}: {sz:.1f} MB")

    t0 = time.time()

    # starmap launches all calls in parallel across separate containers
    inputs = [(v.read_bytes(), fps, conf) for v in VIDEOS]
    results = list(predict_video.starmap(inputs))

    elapsed = time.time() - t0
    print(f"\nAll {len(results)} jobs completed in {elapsed:.1f}s\n")

    for video_path, result in zip(VIDEOS, results):
        out = out_dir / video_path.with_suffix(".glb").name
        out.write_bytes(result["glb"])
        print(
            f"  {video_path.name} -> {out.name} "
            f"({len(result['glb']) / 1024 / 1024:.1f} MB, "
            f"{result['num_frames']} frames, "
            f"{result['num_points']:,} pts)"
        )
