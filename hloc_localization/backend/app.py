"""
hloc-based visual localization on Modal — SuperPoint + LightGlue + PnP.

Builds a reference SfM model from video, then localizes query frames against it.

Deploy:  modal deploy hloc_localization/backend/app.py
Dev:     modal serve hloc_localization/backend/app.py
Build:   modal run hloc_localization/backend/app.py --video-path data/IMG_4717.MOV
"""

import pathlib

import modal

app = modal.App("hloc-localization")

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
    .env({
        "TORCH_HOME": "/opt/torch_cache",
    })
    # Pre-download SuperPoint + NetVLAD + LightGlue weights
    # Use extract_features.main on a dummy image to trigger weight downloads
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

with hloc_image.imports():
    import io
    import os
    import sys
    import tarfile
    import tempfile


@app.function(
    image=hloc_image,
    gpu="A100",
    timeout=1800,
    memory=32768,
)
def build_reference(video_bytes: bytes, target_fps: int = 3) -> dict:
    """
    Build an SfM reference model from a video using hloc.

    Extracts frames, computes SuperPoint features + NetVLAD descriptors,
    matches with LightGlue, and runs pycolmap SfM reconstruction.

    Returns dict with 'tar' (bytes of reference model) and metadata.
    """
    import cv2
    import numpy as np

    sys.path.insert(0, "/opt/hloc")
    from pathlib import Path

    from hloc import (
        extract_features,
        match_features,
        pairs_from_retrieval,
        reconstruction,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # --- Save video and extract frames ---
        video_path = tmpdir / "input.mov"
        video_path.write_bytes(video_bytes)

        images_dir = tmpdir / "images"
        images_dir.mkdir()

        vs = cv2.VideoCapture(str(video_path))
        source_fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(source_fps / target_fps))

        frame_count = 0
        extracted = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            frame_count += 1
            if frame_count % frame_interval == 0:
                fname = f"frame_{extracted:06d}.jpg"
                cv2.imwrite(str(images_dir / fname), frame)
                extracted += 1
        vs.release()
        print(f"Extracted {extracted} frames from {frame_count} total "
              f"(interval={frame_interval}, source_fps={source_fps:.1f})")

        if extracted < 3:
            raise ValueError(f"Only extracted {extracted} frames — need at least 3")

        # --- hloc pipeline ---
        sfm_dir = tmpdir / "sfm"
        features_path = tmpdir / "features.h5"
        descriptors_path = tmpdir / "global_descriptors.h5"
        pairs_path = tmpdir / "pairs.txt"
        matches_path = tmpdir / "matches.h5"

        # 1. Extract SuperPoint features
        print("Extracting SuperPoint features...")
        sp_conf = extract_features.confs["superpoint_max"]
        extract_features.main(
            sp_conf,
            images_dir,
            feature_path=features_path,
        )

        # 2. Extract NetVLAD global descriptors
        print("Extracting NetVLAD descriptors...")
        nv_conf = extract_features.confs["netvlad"]
        extract_features.main(
            nv_conf,
            images_dir,
            feature_path=descriptors_path,
        )

        # 3. Generate pairs from retrieval
        print("Generating retrieval pairs...")
        pairs_from_retrieval.main(
            descriptors_path,
            pairs_path,
            num_matched=min(20, extracted - 1),
        )

        # 4. Match features with LightGlue
        print("Matching features with LightGlue...")
        lg_conf = match_features.confs["superpoint+lightglue"]
        match_features.main(
            lg_conf,
            pairs_path,
            features=features_path,
            matches=matches_path,
        )

        # 5. Run SfM reconstruction
        print("Running SfM reconstruction...")
        model = reconstruction.main(
            sfm_dir,
            images_dir,
            pairs_path,
            features_path,
            matches_path,
        )

        num_images = model.num_reg_images() if model is not None else 0
        num_points = model.num_points3D() if model is not None else 0
        print(f"SfM model: {num_images} registered images, {num_points} 3D points")

        if num_images == 0:
            raise ValueError("SfM reconstruction failed — no images registered")

        # --- Pack everything into a tar ---
        print("Packing reference model...")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            # SfM model
            for f in sfm_dir.rglob("*"):
                if f.is_file():
                    tar.add(str(f), arcname=f"sfm/{f.relative_to(sfm_dir)}")
            # Features + descriptors + matches
            for f in [features_path, descriptors_path, matches_path, pairs_path]:
                if f.exists():
                    tar.add(str(f), arcname=f.name)
            # Images
            for f in images_dir.iterdir():
                if f.is_file():
                    tar.add(str(f), arcname=f"images/{f.name}")

        tar_bytes = buf.getvalue()
        print(f"Reference tar: {len(tar_bytes) / 1024 / 1024:.1f} MB")

    return {
        "tar": tar_bytes,
        "num_frames": extracted,
        "source_fps": source_fps,
        "num_registered": num_images,
        "num_points3d": num_points,
    }


# Cached reference model for localize_frame
_cached_ref = {"path": None, "lock": None}


def _untar_reference(tar_bytes: bytes) -> str:
    """Untar reference model to /tmp and cache it."""
    import hashlib
    import threading

    # Use full content hash to avoid collisions
    ref_hash = hashlib.sha256(tar_bytes).hexdigest()[:16]
    ref_dir = f"/tmp/hloc_ref_{ref_hash}"

    if os.path.exists(ref_dir) and _cached_ref["path"] == ref_dir:
        return ref_dir

    # Simple lock to prevent concurrent unpacking
    if _cached_ref["lock"] is None:
        _cached_ref["lock"] = threading.Lock()

    with _cached_ref["lock"]:
        # Double-check after acquiring lock
        if os.path.exists(ref_dir) and _cached_ref["path"] == ref_dir:
            return ref_dir

        os.makedirs(ref_dir, exist_ok=True)
        buf = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(ref_dir)

        _cached_ref["path"] = ref_dir
        print(f"Untarred reference to {ref_dir}")

    return ref_dir


@app.function(
    image=hloc_image,
    gpu="A100",
    timeout=300,
    memory=16384,
)
def localize_frame(image_bytes: bytes, reference_tar: bytes) -> dict:
    """
    Localize a single query frame against the reference SfM model.

    Uses hloc's localize_sfm pipeline for robust 2D-3D matching and PnP.

    Returns 6DoF pose as {qw, qx, qy, qz, tx, ty, tz, num_inliers, success}.
    """
    import cv2
    import h5py
    import numpy as np
    import pycolmap
    import shutil

    sys.path.insert(0, "/opt/hloc")
    from pathlib import Path

    from hloc import extract_features, match_features, pairs_from_retrieval

    ref_dir = Path(_untar_reference(reference_tar))
    sfm_dir = ref_dir / "sfm"
    db_features_path = ref_dir / "features.h5"
    db_descriptors_path = ref_dir / "global_descriptors.h5"

    # Validate reconstruction exists
    try:
        rec = pycolmap.Reconstruction(str(sfm_dir))
        if rec.num_reg_images() == 0:
            return {"success": False, "error": "Reconstruction has no registered images"}
    except Exception as e:
        return {"success": False, "error": f"Failed to load reconstruction: {e}"}

    # Build a name→image_id lookup and a name→point3D_ids lookup
    name_to_id = {img.name: img_id for img_id, img in rec.images.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save query image
        query_dir = tmpdir / "query"
        query_dir.mkdir()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "error": "Failed to decode image"}
        cv2.imwrite(str(query_dir / "query.jpg"), img)

        # Extract SuperPoint features for query
        query_features_path = tmpdir / "query_features.h5"
        sp_conf = extract_features.confs["superpoint_max"]
        extract_features.main(sp_conf, query_dir, feature_path=query_features_path)

        # Extract NetVLAD descriptor for query
        query_descriptors_path = tmpdir / "query_descriptors.h5"
        nv_conf = extract_features.confs["netvlad"]
        extract_features.main(nv_conf, query_dir, feature_path=query_descriptors_path)

        # Find top-K similar DB images via retrieval
        retrieval_pairs_path = tmpdir / "retrieval_pairs.txt"
        pairs_from_retrieval.main(
            query_descriptors_path,
            retrieval_pairs_path,
            num_matched=10,
            db_descriptors=db_descriptors_path,
        )

        # Merge query features into a copy of DB features for matching
        merged_features_path = tmpdir / "merged_features.h5"
        shutil.copy2(db_features_path, merged_features_path)
        with h5py.File(query_features_path, "r") as src, \
             h5py.File(merged_features_path, "a") as dst:
            for key in src:
                if key not in dst:
                    src.copy(src[key], dst, name=key)

        # Match query features against retrieved DB images
        loc_matches_path = tmpdir / "loc_matches.h5"
        lg_conf = match_features.confs["superpoint+lightglue"]
        match_features.main(
            lg_conf,
            retrieval_pairs_path,
            features=merged_features_path,
            matches=loc_matches_path,
        )

        # --- Build 2D-3D correspondences ---
        # hloc stores features with image name as key. The keypoint order in the
        # features file matches the point2D order in the COLMAP reconstruction
        # (since hloc builds the reconstruction from these features).

        # Identify the query key in the features file
        with h5py.File(query_features_path, "r") as f:
            query_key = list(f.keys())[0]  # Should be "query.jpg"
            query_kpts = f[query_key]["keypoints"][:]

        # Read pairs to know which DB images were matched
        pairs = []
        with open(retrieval_pairs_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))

        # Estimate camera intrinsics from image dimensions
        h_img, w_img = img.shape[:2]
        focal_length = max(h_img, w_img) * 1.2  # Typical iPhone estimate
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
                # hloc stores match keys as "name0/name1" with names sorted
                key_a = f"{q_name}/{db_name}"
                key_b = f"{db_name}/{q_name}"

                if key_a in f_matches:
                    grp = f_matches[key_a]
                    # matches0[i] = j means keypoint i in name0 matches keypoint j in name1
                    matches0 = grp["matches0"][:]
                    is_query_first = True
                elif key_b in f_matches:
                    grp = f_matches[key_b]
                    matches0 = grp["matches0"][:]
                    is_query_first = False
                else:
                    continue

                # Find the DB image in the reconstruction
                if db_name not in name_to_id:
                    continue
                db_img_id = name_to_id[db_name]
                rec_img = rec.images[db_img_id]
                points2D = rec_img.points2D

                # Read DB keypoints to build index from keypoint → point2D
                if db_name not in f_feat:
                    continue
                db_kpts = f_feat[db_name]["keypoints"][:]

                # Build a KD-tree-like index: for each point2D with a 3D point,
                # find the closest DB keypoint. Since hloc built the reconstruction,
                # feature indices should map 1:1 to point2D indices.
                # point2D index i corresponds to keypoint index i.
                p2d_to_p3d = {}
                for idx, p2d in enumerate(points2D):
                    if p2d.has_point3D() and idx < len(db_kpts):
                        p2d_to_p3d[idx] = p2d.point3D_id

                # Extract correspondences from matches
                for i, m in enumerate(matches0):
                    if m < 0:
                        continue
                    m = int(m)

                    if is_query_first:
                        qi, di = i, m  # query kpt index, db kpt index
                    else:
                        qi, di = m, i

                    if qi >= len(query_kpts) or di >= len(db_kpts):
                        continue

                    if di in p2d_to_p3d:
                        p3d_id = p2d_to_p3d[di]
                        if p3d_id in rec.points3D:
                            p2d_all.append(query_kpts[qi])
                            p3d_all.append(rec.points3D[p3d_id].xyz)

        if len(p2d_all) < 4:
            return {
                "success": False,
                "error": f"Not enough 2D-3D correspondences: {len(p2d_all)}",
                "num_correspondences": len(p2d_all),
            }

        p2d_arr = np.array(p2d_all, dtype=np.float64)
        p3d_arr = np.array(p3d_all, dtype=np.float64)
        print(f"Found {len(p2d_arr)} 2D-3D correspondences")

        # Run PnP+RANSAC (pycolmap >= 3.11 API)
        answer = pycolmap.estimate_and_refine_absolute_pose(
            p2d_arr,
            p3d_arr,
            camera,
            estimation_options={"ransac": {"max_error": 12.0}},
        )

        if answer is None:
            return {
                "success": False,
                "error": "PnP+RANSAC failed",
                "num_correspondences": len(p2d_arr),
            }

        # Extract pose from Rigid3d object
        cam_from_world = answer["cam_from_world"]
        quat_xyzw = cam_from_world.rotation.quat  # pycolmap returns (x, y, z, w)
        tvec = cam_from_world.translation
        num_inliers = answer["num_inliers"]

        return {
            "success": True,
            "qw": float(quat_xyzw[3]),
            "qx": float(quat_xyzw[0]),
            "qy": float(quat_xyzw[1]),
            "qz": float(quat_xyzw[2]),
            "tx": float(tvec[0]),
            "ty": float(tvec[1]),
            "tz": float(tvec[2]),
            "num_inliers": int(num_inliers),
            "num_correspondences": len(p2d_arr),
        }


@app.function(
    image=hloc_image,
    gpu="A100",
    timeout=600,
    memory=16384,
)
def localize_batch(images_bytes_list: list[bytes], reference_tar: bytes) -> list[dict]:
    """Localize a batch of query frames against the reference SfM model."""
    results = []
    for i, img_bytes in enumerate(images_bytes_list):
        print(f"Localizing frame {i + 1}/{len(images_bytes_list)}...")
        result = localize_frame.local(img_bytes, reference_tar)
        results.append(result)
    return results


@app.local_entrypoint()
def main(video_path: str, fps: int = 3):
    """
    Build reference SfM model from a video.

    Usage: modal run hloc_localization/app.py --video-path data/IMG_4717.MOV
    """
    p = pathlib.Path(video_path).expanduser().resolve()
    print(f"Reading {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Settings: fps={fps}")

    result = build_reference.remote(p.read_bytes(), target_fps=fps)

    out_dir = pathlib.Path(__file__).parent.parent / "data" / "hloc_reference" / p.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "reference.tar.gz"
    tar_path.write_bytes(result["tar"])

    print(f"\nWrote {tar_path} ({len(result['tar']) / 1024 / 1024:.1f} MB)")
    print(f"Frames extracted: {result['num_frames']}")
    print(f"Registered images: {result['num_registered']}")
    print(f"3D points: {result['num_points3d']:,}")
