"""
Debug localization: extract frames from a video and localize each against a reference map.

Usage:
  python hloc_localization/debug_localize.py \\
    --video data/IMG_4718.MOV \\
    --reference data/hloc_reference/IMG_4717/reference.tar.gz \\
    --fps 1
"""

import argparse
import pathlib
import time

import cv2
import modal

from hloc_localization.backend.app import app, localize_frame


def main():
    parser = argparse.ArgumentParser(description="Debug hloc localization")
    parser.add_argument("--video", required=True, help="Query video path")
    parser.add_argument("--reference", required=True, help="Reference tar.gz path")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to localize")
    args = parser.parse_args()

    video_path = pathlib.Path(args.video).expanduser().resolve()
    ref_path = pathlib.Path(args.reference).expanduser().resolve()

    print(f"Video: {video_path.name}")
    print(f"Reference: {ref_path}")

    ref_tar = ref_path.read_bytes()
    print(f"Reference tar: {len(ref_tar) / 1024 / 1024:.1f} MB")

    # Extract frames from query video
    vs = cv2.VideoCapture(str(video_path))
    source_fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / args.fps))

    frames = []
    count = 0
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frames.append(buf.tobytes())
            if len(frames) >= args.max_frames:
                break
    vs.release()

    print(f"Extracted {len(frames)} query frames (interval={frame_interval})")

    # Localize each frame via Modal
    with app.run():
        t0 = time.time()
        results = []
        for i, frame_bytes in enumerate(frames):
            t_frame = time.time()
            result = localize_frame.remote(frame_bytes, ref_tar)
            elapsed = time.time() - t_frame
            results.append(result)

            if result["success"]:
                print(
                    f"  Frame {i:3d}: "
                    f"t=({result['tx']:7.3f}, {result['ty']:7.3f}, {result['tz']:7.3f}) "
                    f"q=({result['qw']:6.3f}, {result['qx']:6.3f}, {result['qy']:6.3f}, {result['qz']:6.3f}) "
                    f"inliers={result['num_inliers']:3d} "
                    f"({elapsed:.1f}s)"
                )
            else:
                print(f"  Frame {i:3d}: FAILED — {result.get('error', 'unknown')} ({elapsed:.1f}s)")

        total = time.time() - t0
        n_success = sum(1 for r in results if r["success"])
        print(f"\n{n_success}/{len(results)} frames localized in {total:.1f}s")


if __name__ == "__main__":
    main()
