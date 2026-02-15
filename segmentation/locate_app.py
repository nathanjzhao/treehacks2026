"""
Object 3D Localization on Modal — Grounded SAM 2 + Depth Anything V2 + HLoc.

Detects objects via text prompt, estimates depth, localizes camera poses,
then backprojects object centroids into 3D world coordinates.

Requires a pre-built HLoc reference map (from hloc_localization/backend/app.py).

Run:
  modal run segmentation/locate_app.py \
    --video-path data/IMG_4730.MOV \
    --text-prompt "painting. chair." \
    --reference-path hloc_localization/data/hloc_reference/IMG_4720/reference.tar.gz
"""

import json
import pathlib

import modal

app = modal.App("segmentation-locate")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

locate_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "curl", "wget", "cmake", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "Pillow",
        "opencv-python",
        "supervision>=0.22.0",
        "transformers",
        "hydra-core>=1.3.2",
        "iopath>=0.1.10",
        "tqdm",
        "pycocotools",
        "huggingface_hub",
        "matplotlib",
        "scipy",
        "h5py",
        "kornia>=0.6.11",
        "pycolmap>=3.10.0",
    )
    # SAM 2
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /opt/sam2",
        "cd /opt/sam2 && SAM2_BUILD_CUDA=0 pip install -e .",
    )
    # Depth Anything V2
    .run_commands(
        "git clone https://github.com/DepthAnything/Depth-Anything-V2.git /opt/DepthAnythingV2",
    )
    # HLoc
    .run_commands(
        "pip install git+https://github.com/cvg/LightGlue.git",
        "git clone --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc",
        "cd /opt/hloc && pip install -e .",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
    })
    # Pre-download all model weights
    .run_commands(
        # SAM 2.1 large
        "mkdir -p /opt/sam2_ckpts && "
        "curl -L -o /opt/sam2_ckpts/sam2.1_hiera_large.pt "
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        # Grounding DINO + Depth Anything V2
        "python -c \""
        "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; "
        "AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
        "AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download(repo_id='depth-anything/Depth-Anything-V2-Large', "
        "  filename='depth_anything_v2_vitl.pth', "
        "  local_dir='/opt/DepthAnythingV2/checkpoints'); "
        "print('All detector weights downloaded')\"",
        # HLoc weights (SuperPoint + NetVLAD)
        "python -c \""
        "import numpy as np, cv2, os, tempfile; "
        "d = tempfile.mkdtemp(); "
        "img = np.zeros((480,640,3), dtype=np.uint8); "
        "cv2.imwrite(os.path.join(d, 'dummy.jpg'), img); "
        "from pathlib import Path; "
        "from hloc import extract_features; "
        "extract_features.main(extract_features.confs['superpoint_max'], Path(d), feature_path=Path(d)/'sp.h5'); "
        "extract_features.main(extract_features.confs['netvlad'], Path(d), feature_path=Path(d)/'nv.h5'); "
        "print('HLoc weights downloaded')\"",
        gpu="any",
    )
)


def _extract_frames(video_path: str, output_dir: str) -> tuple[list[str], float]:
    import cv2
    import os
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    names = []
    idx = 0
    while True:
        ok, frame = vs.read()
        if not ok:
            break
        name = f"{idx:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, name), frame)
        names.append(name)
        idx += 1
    vs.release()
    return names, fps


def _extract_frames_subsampled(video_path: str, output_dir: str, target_fps: float) -> tuple[list[str], list[int], float]:
    """Extract frames at target_fps. Returns (names, original_indices, source_fps)."""
    import cv2
    import os
    vs = cv2.VideoCapture(video_path)
    source_fps = vs.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(source_fps / target_fps))
    names = []
    orig_indices = []
    count = 0
    out_idx = 0
    while True:
        ok, frame = vs.read()
        if not ok:
            break
        if count % interval == 0:
            name = f"{out_idx:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, name), frame)
            names.append(name)
            orig_indices.append(count)
            out_idx += 1
        count += 1
    vs.release()
    return names, orig_indices, source_fps


@app.function(
    image=locate_image,
    gpu="A100",
    timeout=1800,
    memory=32768,
)
def locate_objects(
    video_bytes: bytes,
    text_prompt: str,
    reference_tar: bytes,
    prompt_type: str = "mask",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    localize_fps: float = 2.0,
    depth_input_size: int = 518,
) -> dict:
    """
    Full pipeline: detect objects → estimate depth → localize camera → 3D positions.

    Args:
        video_bytes: Raw video file.
        text_prompt: Objects to detect (e.g. "painting. chair.").
        reference_tar: Pre-built HLoc reference tar.gz bytes.
        prompt_type: SAM 2 prompt type.
        box_threshold: Grounding DINO confidence threshold.
        text_threshold: Grounding DINO text threshold.
        localize_fps: FPS for localization (lower = faster, fewer poses).
        depth_input_size: Resolution for depth estimation.

    Returns dict with:
        - object_positions: per-object list of {frame, position_3d, depth, label, ...}
        - camera_poses: per-frame camera pose (for viewer alignment)
        - objects_detected: list of class names
        - num_frames: total frame count
    """
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
    import supervision as sv
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda"

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save video
        video_path = os.path.join(tmpdir, "input.mov")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # ================================================================
        # PHASE 1: Object tracking with Grounded SAM 2
        # ================================================================
        print("\n========== PHASE 1: Object Tracking ==========")

        # Extract ALL frames for tracking
        all_frames_dir = os.path.join(tmpdir, "all_frames")
        os.makedirs(all_frames_dir)
        all_frame_names, source_fps = _extract_frames(video_path, all_frames_dir)
        print(f"Extracted {len(all_frame_names)} frames at {source_fps:.1f} fps")

        # Load SAM 2
        print("Loading SAM 2.1...")
        sam2_ckpt = "/opt/sam2_ckpts/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        video_predictor = build_sam2_video_predictor(model_cfg, sam2_ckpt)
        sam2_image_model = build_sam2(model_cfg, sam2_ckpt)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        # Load Grounding DINO
        print("Loading Grounding DINO...")
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(device)

        # Init video predictor
        inference_state = video_predictor.init_state(video_path=all_frames_dir)

        # Detect on frame 0
        img_path = os.path.join(all_frames_dir, all_frame_names[0])
        image = Image.open(img_path)
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0].get("text_labels", results[0].get("labels", []))
        class_names = [str(n) for n in class_names]

        valid_indices = [i for i, name in enumerate(class_names) if name.strip()]
        input_boxes = input_boxes[valid_indices] if valid_indices else np.empty((0, 4))
        confidences = [confidences[i] for i in valid_indices]
        class_names = [class_names[i] for i in valid_indices]

        print(f"Detected {len(class_names)} objects: {class_names}")
        if len(class_names) == 0:
            raise ValueError(f"No objects detected for prompt: {text_prompt!r}")

        # Get masks + register
        image_predictor.set_image(np.array(image.convert("RGB")))
        OBJECTS = class_names
        all_masks = []
        for box in input_boxes:
            mask, _, _ = image_predictor.predict(
                point_coords=None, point_labels=None,
                box=box[None, :], multimask_output=False,
            )
            if mask.ndim == 4:
                mask = mask.squeeze(0).squeeze(0)
            elif mask.ndim == 3:
                mask = mask.squeeze(0)
            all_masks.append(mask)
        masks = np.stack(all_masks, axis=0)

        for obj_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0, obj_id=obj_id, mask=mask,
            )

        # Propagate
        print("Propagating tracking...")
        video_segments = {}
        for out_fi, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_fi] = {
                oid: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, oid in enumerate(out_obj_ids)
            }
        print(f"Tracking done: {len(video_segments)} frames")

        # Free SAM 2 memory
        del video_predictor, sam2_image_model, image_predictor, grounding_model
        torch.cuda.empty_cache()

        # ================================================================
        # PHASE 2: Depth estimation
        # ================================================================
        print("\n========== PHASE 2: Depth Estimation ==========")

        sys.path.insert(0, "/opt/DepthAnythingV2")
        from depth_anything_v2.dpt import DepthAnythingV2

        depth_model = DepthAnythingV2(
            encoder="vitl", features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        depth_model.load_state_dict(torch.load(
            "/opt/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth",
            map_location="cpu",
        ))
        depth_model = depth_model.to(device).eval()

        # Compute depth for frames we'll localize (subsampled)
        interval = max(1, int(source_fps / localize_fps))
        localize_frame_indices = list(range(0, len(all_frame_names), interval))
        print(f"Computing depth for {len(localize_frame_indices)} frames (every {interval}th)")

        first_frame = cv2.imread(os.path.join(all_frames_dir, all_frame_names[0]))
        frame_h, frame_w = first_frame.shape[:2]

        depth_maps = {}  # frame_idx -> depth array (H, W) float32
        for fi in localize_frame_indices:
            frame_bgr = cv2.imread(os.path.join(all_frames_dir, all_frame_names[fi]))
            with torch.amp.autocast("cuda"):
                depth = depth_model.infer_image(frame_bgr, depth_input_size)
            # Resize to frame size
            depth = cv2.resize(depth.astype(np.float32), (frame_w, frame_h))
            depth_maps[fi] = depth

        del depth_model
        torch.cuda.empty_cache()
        print(f"Depth estimation done for {len(depth_maps)} frames")

        # ================================================================
        # PHASE 3: Camera localization via HLoc
        # ================================================================
        print("\n========== PHASE 3: Camera Localization ==========")

        sys.path.insert(0, "/opt/hloc")
        from pathlib import Path

        from hloc import extract_features, match_features, pairs_from_retrieval

        # Untar reference
        ref_dir = Path(tmpdir) / "reference"
        ref_dir.mkdir()
        buf = io.BytesIO(reference_tar)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(str(ref_dir))

        sfm_dir = ref_dir / "sfm"
        db_features_path = ref_dir / "features.h5"
        db_descriptors_path = ref_dir / "global_descriptors.h5"

        rec = pycolmap.Reconstruction(str(sfm_dir))
        print(f"Reference: {rec.num_reg_images()} images, {rec.num_points3D()} 3D points")
        name_to_id = {img.name: img_id for img_id, img in rec.images.items()}

        # Camera intrinsics estimate
        focal_length = max(frame_h, frame_w) * 1.2
        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=frame_w, height=frame_h,
            params=[focal_length, frame_w / 2, frame_h / 2],
        )

        # Localize each subsampled frame
        camera_poses = {}  # frame_idx -> {qw, qx, qy, qz, tx, ty, tz, success, ...}

        for fi in localize_frame_indices:
            frame_path = os.path.join(all_frames_dir, all_frame_names[fi])

            # Prepare query
            query_dir = Path(tmpdir) / f"query_{fi}"
            query_dir.mkdir()
            shutil.copy2(frame_path, query_dir / "query.jpg")

            # Extract features
            qf_path = query_dir / "qf.h5"
            qd_path = query_dir / "qd.h5"
            extract_features.main(
                extract_features.confs["superpoint_max"],
                query_dir, feature_path=qf_path,
            )
            extract_features.main(
                extract_features.confs["netvlad"],
                query_dir, feature_path=qd_path,
            )

            # Retrieval
            pairs_path = query_dir / "pairs.txt"
            pairs_from_retrieval.main(
                qd_path, pairs_path, num_matched=10,
                db_descriptors=db_descriptors_path,
            )

            # Merge features
            merged_path = query_dir / "merged.h5"
            shutil.copy2(db_features_path, merged_path)
            with h5py.File(qf_path, "r") as src, h5py.File(merged_path, "a") as dst:
                for key in src:
                    if key not in dst:
                        src.copy(src[key], dst, name=key)

            # Match
            matches_path = query_dir / "matches.h5"
            match_features.main(
                match_features.confs["superpoint+lightglue"],
                pairs_path, features=merged_path, matches=matches_path,
            )

            # Build 2D-3D correspondences
            with h5py.File(qf_path, "r") as f:
                query_key = list(f.keys())[0]
                query_kpts = f[query_key]["keypoints"][:]

            pairs = []
            with open(pairs_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        pairs.append((parts[0], parts[1]))

            p2d_all, p3d_all = [], []
            with h5py.File(matches_path, "r") as f_m, h5py.File(merged_path, "r") as f_f:
                for q_name, db_name in pairs:
                    key_a = f"{q_name}/{db_name}"
                    key_b = f"{db_name}/{q_name}"
                    if key_a in f_m:
                        matches0 = f_m[key_a]["matches0"][:]
                        is_query_first = True
                    elif key_b in f_m:
                        matches0 = f_m[key_b]["matches0"][:]
                        is_query_first = False
                    else:
                        continue

                    if db_name not in name_to_id or db_name not in f_f:
                        continue
                    db_img_id = name_to_id[db_name]
                    rec_img = rec.images[db_img_id]
                    points2D = rec_img.points2D
                    db_kpts = f_f[db_name]["keypoints"][:]

                    p2d_to_p3d = {}
                    for idx, p2d in enumerate(points2D):
                        if p2d.has_point3D() and idx < len(db_kpts):
                            p2d_to_p3d[idx] = p2d.point3D_id

                    for i, m in enumerate(matches0):
                        if m < 0:
                            continue
                        m = int(m)
                        qi, di = (i, m) if is_query_first else (m, i)
                        if qi >= len(query_kpts) or di >= len(db_kpts):
                            continue
                        if di in p2d_to_p3d:
                            p3d_id = p2d_to_p3d[di]
                            if p3d_id in rec.points3D:
                                p2d_all.append(query_kpts[qi])
                                p3d_all.append(rec.points3D[p3d_id].xyz)

            if len(p2d_all) < 4:
                print(f"  Frame {fi}: FAILED (only {len(p2d_all)} correspondences)")
                camera_poses[fi] = {"success": False}
                continue

            p2d_arr = np.array(p2d_all, dtype=np.float64)
            p3d_arr = np.array(p3d_all, dtype=np.float64)

            answer = pycolmap.estimate_and_refine_absolute_pose(
                p2d_arr, p3d_arr, camera,
                estimation_options={"ransac": {"max_error": 12.0}},
            )

            if answer is None:
                print(f"  Frame {fi}: PnP FAILED")
                camera_poses[fi] = {"success": False}
                continue

            cam_from_world = answer["cam_from_world"]
            quat_xyzw = cam_from_world.rotation.quat
            tvec = cam_from_world.translation

            camera_poses[fi] = {
                "success": True,
                "qw": float(quat_xyzw[3]), "qx": float(quat_xyzw[0]),
                "qy": float(quat_xyzw[1]), "qz": float(quat_xyzw[2]),
                "tx": float(tvec[0]), "ty": float(tvec[1]), "tz": float(tvec[2]),
                "num_inliers": int(answer["num_inliers"]),
            }
            print(f"  Frame {fi}: OK, inliers={answer['num_inliers']}")

            # Cleanup query dir
            shutil.rmtree(query_dir)

        n_success = sum(1 for p in camera_poses.values() if p.get("success"))
        print(f"Localized {n_success}/{len(localize_frame_indices)} frames")

        # ================================================================
        # PHASE 4: Backproject object centroids into 3D
        # ================================================================
        print("\n========== PHASE 4: 3D Backprojection ==========")

        def quat_to_rotation(qw, qx, qy, qz):
            return np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
            ])

        fx = fy = focal_length
        cx, cy = frame_w / 2, frame_h / 2

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
        object_positions = []  # list of {frame, label, obj_id, position_3d, depth, pixel_centroid}

        for fi in localize_frame_indices:
            pose = camera_poses.get(fi)
            if not pose or not pose.get("success"):
                continue

            depth_map = depth_maps.get(fi)
            if depth_map is None:
                continue

            segments = video_segments.get(fi, {})

            R = quat_to_rotation(pose["qw"], pose["qx"], pose["qy"], pose["qz"])
            t = np.array([pose["tx"], pose["ty"], pose["tz"]])

            for obj_id, obj_mask in segments.items():
                m = obj_mask.squeeze()
                if m.shape != (frame_h, frame_w):
                    m = cv2.resize(m.astype(np.uint8), (frame_w, frame_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

                mask_pixels = np.argwhere(m)  # (N, 2) in (row, col) = (y, x)
                if len(mask_pixels) < 10:
                    continue

                # Centroid in pixel coords
                cy_obj = mask_pixels[:, 0].mean()
                cx_obj = mask_pixels[:, 1].mean()

                # Median depth within mask
                depth_vals = depth_map[m]
                median_depth = float(np.median(depth_vals))

                if median_depth <= 0 or not np.isfinite(median_depth):
                    continue

                # Backproject to camera coordinates
                X_cam = np.array([
                    (cx_obj - cx) * median_depth / fx,
                    (cy_obj - cy) * median_depth / fy,
                    median_depth,
                ])

                # Camera-to-world: world = R^T @ (cam - t)
                X_world = R.T @ (X_cam - t)

                label = ID_TO_OBJECTS.get(obj_id, f"obj_{obj_id}")
                object_positions.append({
                    "frame_idx": fi,
                    "obj_id": int(obj_id),
                    "label": label,
                    "position_3d": X_world.tolist(),
                    "depth": median_depth,
                    "pixel_centroid": [float(cx_obj), float(cy_obj)],
                    "mask_area": int(mask_pixels.shape[0]),
                })

        print(f"Computed {len(object_positions)} 3D object observations")

        # Aggregate: average position per unique object
        obj_aggregated = {}
        for obs in object_positions:
            key = obs["obj_id"]
            if key not in obj_aggregated:
                obj_aggregated[key] = {
                    "label": obs["label"],
                    "positions": [],
                    "depths": [],
                }
            obj_aggregated[key]["positions"].append(obs["position_3d"])
            obj_aggregated[key]["depths"].append(obs["depth"])

        object_summary = []
        for obj_id, data in obj_aggregated.items():
            positions = np.array(data["positions"])
            avg_pos = positions.mean(axis=0).tolist()
            std_pos = positions.std(axis=0).tolist()
            object_summary.append({
                "obj_id": obj_id,
                "label": data["label"],
                "mean_position_3d": avg_pos,
                "std_position_3d": std_pos,
                "mean_depth": float(np.mean(data["depths"])),
                "num_observations": len(data["positions"]),
            })
            print(f"  {data['label']} (id={obj_id}): "
                  f"pos=({avg_pos[0]:.2f}, {avg_pos[1]:.2f}, {avg_pos[2]:.2f}), "
                  f"depth={np.mean(data['depths']):.2f}, "
                  f"n={len(data['positions'])}")

    # Serialize camera poses (only successful ones)
    poses_out = {
        str(fi): p for fi, p in camera_poses.items() if p.get("success")
    }

    return {
        "object_positions": object_positions,
        "object_summary": object_summary,
        "camera_poses": poses_out,
        "objects_detected": list(OBJECTS),
        "num_frames": len(all_frame_names),
        "source_fps": source_fps,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "focal_length": focal_length,
    }


@app.local_entrypoint()
def main(
    video_path: str,
    text_prompt: str = "painting. chair.",
    reference_path: str = "",
    box_threshold: float = 0.3,
    localize_fps: float = 2.0,
):
    """
    Locate objects in 3D from video.

    Args:
        video_path: Video file (.mov, .mp4).
        text_prompt: Objects to detect.
        reference_path: Path to HLoc reference.tar.gz.
        box_threshold: Grounding DINO confidence.
        localize_fps: Frames per second for localization.
    """
    p = pathlib.Path(video_path).expanduser().resolve()
    ref = pathlib.Path(reference_path).expanduser().resolve()

    print(f"Video: {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Reference: {ref.name} ({ref.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Prompt: {text_prompt!r}")

    result = locate_objects.remote(
        p.read_bytes(),
        text_prompt=text_prompt,
        reference_tar=ref.read_bytes(),
        box_threshold=box_threshold,
        localize_fps=localize_fps,
    )

    out_dir = pathlib.Path(__file__).parent.parent / "data" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{p.stem}_objects3d.json"
    out_data = {
        "object_positions": result["object_positions"],
        "object_summary": result["object_summary"],
        "camera_poses": result["camera_poses"],
        "objects_detected": result["objects_detected"],
        "num_frames": result["num_frames"],
        "source_fps": result["source_fps"],
        "frame_w": result["frame_w"],
        "frame_h": result["frame_h"],
        "focal_length": result["focal_length"],
    }
    out_json.write_text(json.dumps(out_data, indent=2))

    print(f"\nWrote {out_json}")
    print(f"\nObject Summary:")
    for obj in result["object_summary"]:
        pos = obj["mean_position_3d"]
        print(f"  {obj['label']}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
              f"depth={obj['mean_depth']:.2f} ({obj['num_observations']} obs)")
