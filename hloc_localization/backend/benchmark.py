"""
Benchmark hloc localization per-frame timing with component breakdown.

Usage:
  modal run hloc_localization/backend/benchmark.py \
    --video data/IMG_4724.mov \
    --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz \
    --fps 2 --max-frames 10
"""

import pathlib
import time

import modal

bench_app = modal.App("hloc-benchmark")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

hloc_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "cmake", "build-essential")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "opencv-python",
        "h5py",
        "scipy",
        "tqdm",
        "matplotlib",
        "plotly",
        "kornia>=0.6.11",
        "pycolmap>=3.10.0",
        "Pillow",
    )
    .run_commands(
        "pip install git+https://github.com/cvg/LightGlue.git",
        "git clone --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc",
        "cd /opt/hloc && pip install -e .",
    )
    .env({"TORCH_HOME": "/opt/torch_cache"})
    .run_commands(
        "python -c \""
        "import numpy as np, cv2, os, tempfile; "
        "d = tempfile.mkdtemp(); "
        "img = np.zeros((480,640,3), dtype=np.uint8); "
        "cv2.imwrite(os.path.join(d, 'dummy.jpg'), img); "
        "from pathlib import Path; "
        "from hloc import extract_features; "
        "extract_features.main(extract_features.confs['superpoint_max'], Path(d), feature_path=Path(d)/'sp.h5'); "
        "print('SuperPoint weights downloaded'); "
        "extract_features.main(extract_features.confs['netvlad'], Path(d), feature_path=Path(d)/'nv.h5'); "
        "print('NetVLAD weights downloaded'); "
        "\"",
        gpu="any",
    )
)


@bench_app.function(
    image=hloc_image,
    gpu="A100",
    timeout=600,
    memory=16384,
)
def benchmark_localize(
    video_bytes: bytes,
    reference_tar: bytes,
    target_fps: float = 2.0,
    max_frames: int = 10,
) -> dict:
    """Localize frames with detailed per-component timing."""
    import io
    import os
    import shutil
    import sys
    import tarfile
    import tempfile

    import cv2
    import h5py
    import numpy as np
    import pycolmap

    sys.path.insert(0, "/opt/hloc")
    from pathlib import Path
    from hloc import extract_features, match_features, pairs_from_retrieval

    # --- Extract frames from video ---
    with tempfile.NamedTemporaryFile(suffix=".mov") as tmp_vid:
        tmp_vid.write(video_bytes)
        tmp_vid.flush()
        vs = cv2.VideoCapture(tmp_vid.name)
        source_fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(source_fps / target_fps))
        frames_bytes = []
        count = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frames_bytes.append(buf.tobytes())
                if len(frames_bytes) >= max_frames:
                    break
        vs.release()
    print(f"Extracted {len(frames_bytes)} query frames (interval={frame_interval}, source={source_fps:.0f}fps)")

    # --- Untar reference ---
    t_setup_start = time.time()
    ref_dir = Path(tempfile.mkdtemp(prefix="hloc_bench_ref_"))
    buf = io.BytesIO(reference_tar)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(ref_dir)

    sfm_dir = ref_dir / "sfm"
    db_features_path = ref_dir / "features.h5"
    db_descriptors_path = ref_dir / "global_descriptors.h5"

    rec = pycolmap.Reconstruction(str(sfm_dir))
    name_to_id = {img.name: img_id for img_id, img in rec.images.items()}
    t_setup = time.time() - t_setup_start
    print(f"Setup (untar + load reconstruction): {t_setup:.3f}s")
    print(f"Reference: {rec.num_reg_images()} images, {rec.num_points3D()} 3D points")
    print(f"\nBenchmarking {len(frames_bytes)} frames...\n")

    sp_conf = extract_features.confs["superpoint_max"]
    nv_conf = extract_features.confs["netvlad"]
    lg_conf = match_features.confs["superpoint+lightglue"]

    frame_results = []

    for idx, img_bytes in enumerate(frames_bytes):
        timings = {}
        t_total_start = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. Decode image
            t = time.time()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            query_dir = tmpdir / "query"
            query_dir.mkdir()
            cv2.imwrite(str(query_dir / "query.jpg"), img)
            timings["decode_save"] = time.time() - t

            # 2. SuperPoint feature extraction
            t = time.time()
            query_features_path = tmpdir / "query_features.h5"
            extract_features.main(sp_conf, query_dir, feature_path=query_features_path)
            timings["superpoint"] = time.time() - t

            # 3. NetVLAD descriptor extraction
            t = time.time()
            query_descriptors_path = tmpdir / "query_descriptors.h5"
            extract_features.main(nv_conf, query_dir, feature_path=query_descriptors_path)
            timings["netvlad"] = time.time() - t

            # 4. Retrieval (top-K similar images)
            t = time.time()
            retrieval_pairs_path = tmpdir / "retrieval_pairs.txt"
            pairs_from_retrieval.main(
                query_descriptors_path,
                retrieval_pairs_path,
                num_matched=10,
                db_descriptors=db_descriptors_path,
            )
            timings["retrieval"] = time.time() - t

            # 5. Merge features + LightGlue matching
            t = time.time()
            merged_features_path = tmpdir / "merged_features.h5"
            shutil.copy2(db_features_path, merged_features_path)
            with h5py.File(query_features_path, "r") as src, \
                 h5py.File(merged_features_path, "a") as dst:
                for key in src:
                    if key not in dst:
                        src.copy(src[key], dst, name=key)
            timings["merge_features"] = time.time() - t

            t = time.time()
            loc_matches_path = tmpdir / "loc_matches.h5"
            match_features.main(
                lg_conf,
                retrieval_pairs_path,
                features=merged_features_path,
                matches=loc_matches_path,
            )
            timings["lightglue"] = time.time() - t

            # 6. Build 2D-3D correspondences
            t = time.time()
            with h5py.File(query_features_path, "r") as f:
                query_key = list(f.keys())[0]
                query_kpts = f[query_key]["keypoints"][:]

            pairs = []
            with open(retrieval_pairs_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        pairs.append((parts[0], parts[1]))

            h_img, w_img = img.shape[:2]
            focal_length = max(h_img, w_img) * 1.2
            camera = pycolmap.Camera(
                model="SIMPLE_PINHOLE",
                width=w_img,
                height=h_img,
                params=[focal_length, w_img / 2, h_img / 2],
            )

            p2d_all = []
            p3d_all = []

            with h5py.File(loc_matches_path, "r") as f_matches, \
                 h5py.File(merged_features_path, "r") as f_feat:
                for q_name, db_name in pairs:
                    key_a = f"{q_name}/{db_name}"
                    key_b = f"{db_name}/{q_name}"
                    if key_a in f_matches:
                        grp = f_matches[key_a]
                        matches0 = grp["matches0"][:]
                        is_query_first = True
                    elif key_b in f_matches:
                        grp = f_matches[key_b]
                        matches0 = grp["matches0"][:]
                        is_query_first = False
                    else:
                        continue

                    if db_name not in name_to_id:
                        continue
                    db_img_id = name_to_id[db_name]
                    rec_img = rec.images[db_img_id]
                    points2D = rec_img.points2D
                    if db_name not in f_feat:
                        continue
                    db_kpts = f_feat[db_name]["keypoints"][:]

                    p2d_to_p3d = {}
                    for i, p2d in enumerate(points2D):
                        if p2d.has_point3D() and i < len(db_kpts):
                            p2d_to_p3d[i] = p2d.point3D_id

                    for i, m in enumerate(matches0):
                        if m < 0:
                            continue
                        m = int(m)
                        if is_query_first:
                            qi, di = i, m
                        else:
                            qi, di = m, i
                        if qi >= len(query_kpts) or di >= len(db_kpts):
                            continue
                        if di in p2d_to_p3d:
                            p3d_id = p2d_to_p3d[di]
                            if p3d_id in rec.points3D:
                                p2d_all.append(query_kpts[qi])
                                p3d_all.append(rec.points3D[p3d_id].xyz)

            timings["correspondences"] = time.time() - t

            # 7. PnP + RANSAC
            t = time.time()
            success = False
            num_inliers = 0
            num_correspondences = len(p2d_all)

            if len(p2d_all) >= 4:
                p2d_arr = np.array(p2d_all, dtype=np.float64)
                p3d_arr = np.array(p3d_all, dtype=np.float64)
                answer = pycolmap.estimate_and_refine_absolute_pose(
                    p2d_arr, p3d_arr, camera,
                    estimation_options={"ransac": {"max_error": 12.0}},
                )
                if answer is not None:
                    success = True
                    num_inliers = int(answer["num_inliers"])
            timings["pnp_ransac"] = time.time() - t

        timings["total"] = time.time() - t_total_start

        frame_results.append({
            "frame_idx": idx,
            "success": success,
            "num_inliers": num_inliers,
            "num_correspondences": num_correspondences,
            "timings": timings,
        })

        status = f"inliers={num_inliers}" if success else "FAILED"
        print(
            f"Frame {idx:3d} | "
            f"total={timings['total']:.3f}s | "
            f"SP={timings['superpoint']:.3f}s  "
            f"NV={timings['netvlad']:.3f}s  "
            f"ret={timings['retrieval']:.3f}s  "
            f"LG={timings['lightglue']:.3f}s  "
            f"corr={timings['correspondences']:.3f}s  "
            f"PnP={timings['pnp_ransac']:.3f}s | "
            f"{status}"
        )

    # --- Summary statistics ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n = len(frame_results)
    n_success = sum(1 for r in frame_results if r["success"])

    all_timings = {}
    for r in frame_results:
        for k, v in r["timings"].items():
            all_timings.setdefault(k, []).append(v)

    print(f"\nFrames: {n_success}/{n} localized successfully")
    print(f"\nComponent timing (seconds) over {n} frames:")
    print(f"{'Component':<20s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s} {'% of total':>10s}")
    print("-" * 60)

    import statistics
    mean_total = statistics.mean(all_timings["total"])
    component_order = [
        "decode_save", "superpoint", "netvlad", "retrieval",
        "merge_features", "lightglue", "correspondences", "pnp_ransac", "total",
    ]
    for comp in component_order:
        vals = all_timings[comp]
        mn = statistics.mean(vals)
        lo = min(vals)
        hi = max(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        pct = (mn / mean_total * 100) if comp != "total" else 100.0
        print(f"{comp:<20s} {mn:8.3f} {lo:8.3f} {hi:8.3f} {sd:8.3f} {pct:9.1f}%")

    return {
        "setup_time": t_setup,
        "num_frames": n,
        "num_success": n_success,
        "frame_results": frame_results,
        "mean_total": mean_total,
    }


@bench_app.local_entrypoint()
def main(
    video: str,
    reference: str,
    fps: float = 2.0,
    max_frames: int = 10,
):
    """
    Benchmark hloc localization per-frame.

    Usage:
      modal run hloc_localization/backend/benchmark.py \
        --video data/IMG_4724.mov \
        --reference hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz
    """
    video_path = pathlib.Path(video).expanduser().resolve()
    ref_path = pathlib.Path(reference).expanduser().resolve()

    print(f"Video: {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Reference: {ref_path.name} ({ref_path.stat().st_size / 1024 / 1024:.1f} MB)")

    video_bytes = video_path.read_bytes()
    ref_tar = ref_path.read_bytes()

    t0 = time.time()
    result = benchmark_localize.remote(video_bytes, ref_tar, target_fps=fps, max_frames=max_frames)
    wall_time = time.time() - t0

    print(f"\nWall time (including Modal overhead): {wall_time:.1f}s")
    print(f"Mean per-frame (GPU compute only): {result['mean_total']:.3f}s")
    print(f"Theoretical max FPS: {1.0 / result['mean_total']:.1f}")
