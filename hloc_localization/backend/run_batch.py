"""
Build hloc reference maps for all videos in data/ via Modal.

Usage: modal run hloc_localization/run_batch.py
"""

import pathlib
import time

import modal

from hloc_localization.backend.app import app, build_reference

_data_dir = pathlib.Path(__file__).parent.parent / "data"

VIDEOS = sorted(_data_dir.glob("*.MOV")) + sorted(_data_dir.glob("*.mov"))


@app.local_entrypoint()
def batch(fps: int = 3):
    """Build hloc reference maps for all videos in parallel."""
    out_base = _data_dir / "hloc_reference"
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Launching {len(VIDEOS)} parallel hloc reference builds (fps={fps})")
    for v in VIDEOS:
        sz = v.stat().st_size / 1024 / 1024
        print(f"  {v.name}: {sz:.1f} MB")

    t0 = time.time()

    inputs = [(v.read_bytes(), fps) for v in VIDEOS]
    results = list(build_reference.starmap(inputs))

    elapsed = time.time() - t0
    print(f"\nAll {len(results)} jobs completed in {elapsed:.1f}s\n")

    for video_path, result in zip(VIDEOS, results):
        out_dir = out_base / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        tar_path = out_dir / "reference.tar.gz"
        tar_path.write_bytes(result["tar"])
        print(
            f"  {video_path.name} -> {tar_path.relative_to(_data_dir)} "
            f"({len(result['tar']) / 1024 / 1024:.1f} MB, "
            f"{result['num_frames']} frames, "
            f"{result['num_registered']} registered, "
            f"{result['num_points3d']:,} pts)"
        )
