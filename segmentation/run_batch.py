"""
Run Grounded SAM 2 tracking on multiple videos in parallel via Modal.

Usage: modal run segmentation/run_batch.py --text-prompt "person. car."
"""

import pathlib
import time

import modal

from segmentation.app import app, track_objects

_data_dir = pathlib.Path(__file__).parent.parent / "data"

VIDEOS = sorted(_data_dir.glob("*.MOV")) + sorted(_data_dir.glob("*.mov"))


@app.local_entrypoint()
def batch(
    text_prompt: str = "person.",
    prompt_type: str = "mask",
    box_threshold: float = 0.4,
):
    """Run Grounded SAM 2 tracking on all data/ videos in parallel."""
    out_dir = pathlib.Path(__file__).parent.parent / "data" / "segmentation"
    out_dir.mkdir(exist_ok=True)

    print(f"Launching {len(VIDEOS)} parallel tracking jobs")
    print(f"  text_prompt={text_prompt!r}, prompt_type={prompt_type}")
    for v in VIDEOS:
        sz = v.stat().st_size / 1024 / 1024
        print(f"  {v.name}: {sz:.1f} MB")

    t0 = time.time()

    inputs = [
        (v.read_bytes(), text_prompt, prompt_type, box_threshold)
        for v in VIDEOS
    ]
    results = list(track_objects.starmap(inputs))

    elapsed = time.time() - t0
    print(f"\nAll {len(results)} jobs completed in {elapsed:.1f}s\n")

    for video_path, result in zip(VIDEOS, results):
        stem = video_path.stem

        out_video = out_dir / f"{stem}_tracked.mp4"
        out_video.write_bytes(result["video"])

        out_json = out_dir / f"{stem}_detections.json"
        out_json.write_text(result["detections_json"])

        print(
            f"  {video_path.name} -> {out_video.name} "
            f"({len(result['video']) / 1024 / 1024:.1f} MB, "
            f"{result['num_frames']} frames, "
            f"objects: {result['objects_detected']})"
        )
