"""
Grounded SAM 2 + Depth Anything V2 on Modal — Object-Aware Depth.

Detects and tracks objects via text prompt, estimates per-frame depth,
then masks depth maps to show only the segmented objects' depth.

Deploy:  modal deploy segmentation/depth_app.py
Run:     modal run segmentation/depth_app.py --video-path data/IMG_4723.MOV --text-prompt "painting. chair."
"""

import json
import pathlib

import modal

app = modal.App("segmentation-depth")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

ENCODER = "vitl"
MODEL_CONFIGS = {
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

combined_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "curl", "wget")
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
        # Grounding DINO from HF
        "python -c \""
        "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; "
        "AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
        "AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
        "print('Grounding DINO downloaded')\"",
        # Depth Anything V2 large
        "python -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download("
        "  repo_id='depth-anything/Depth-Anything-V2-Large',"
        "  filename='depth_anything_v2_vitl.pth',"
        "  local_dir='/opt/DepthAnythingV2/checkpoints'"
        "); print('Depth Anything V2 downloaded')\"",
        gpu="any",
    )
)


def _extract_frames(video_path: str, output_dir: str) -> tuple[list[str], float]:
    """Extract all frames as numbered JPEGs. Returns (frame_names, fps)."""
    import cv2
    import os

    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_names = []
    idx = 0
    while True:
        ok, frame = vs.read()
        if not ok:
            break
        name = f"{idx:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, name), frame)
        frame_names.append(name)
        idx += 1
    vs.release()
    return frame_names, fps


def _sample_points_from_masks(masks, num_points=10):
    import numpy as np
    n = masks.shape[0]
    points = []
    for i in range(n):
        indices = np.argwhere(masks[i] == 1)[:, ::-1]
        if len(indices) == 0:
            points.append(np.zeros((num_points, 2), dtype=np.float32))
            continue
        replace = len(indices) < num_points
        sampled = indices[np.random.choice(len(indices), num_points, replace=replace)]
        points.append(sampled)
    return np.array(points, dtype=np.float32)


@app.function(
    image=combined_image,
    gpu="A100",
    timeout=1200,
    memory=32768,
)
def segment_depth(
    video_bytes: bytes,
    text_prompt: str,
    prompt_type: str = "mask",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    ann_frame_idx: int = 0,
    depth_input_size: int = 518,
) -> dict:
    """
    Detect/track objects + estimate depth, then produce masked depth videos.

    Returns:
        - "masked_depth_video": mp4 showing depth only where objects are segmented
        - "full_depth_video": mp4 of the full depth map (dimmed, objects highlighted)
        - "detections_json": per-frame bounding boxes
        - "objects_detected": list of class names
        - "num_frames": frame count
        - "source_fps": original fps
    """
    import os
    import sys
    import tempfile

    import cv2
    import numpy as np
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

    # ---- Load all models ----
    print("Loading SAM 2.1...")
    sam2_ckpt = "/opt/sam2_ckpts/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_ckpt)
    sam2_image_model = build_sam2(model_cfg, sam2_ckpt)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    print("Loading Grounding DINO...")
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)

    print("Loading Depth Anything V2...")
    sys.path.insert(0, "/opt/DepthAnythingV2")
    from depth_anything_v2.dpt import DepthAnythingV2
    cfg = MODEL_CONFIGS[ENCODER]
    depth_model = DepthAnythingV2(**cfg)
    depth_model.load_state_dict(torch.load(
        f"/opt/DepthAnythingV2/checkpoints/depth_anything_v2_{ENCODER}.pth",
        map_location="cpu",
    ))
    depth_model = depth_model.to(device).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save video + extract frames
        video_path = os.path.join(tmpdir, "input.mov")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir)
        print("Extracting frames...")
        frame_names, source_fps = _extract_frames(video_path, frames_dir)
        print(f"Extracted {len(frame_names)} frames at {source_fps:.1f} fps")

        if not frame_names:
            raise ValueError("No frames extracted")

        # ==== PHASE 1: Object tracking with Grounded SAM 2 ====
        print("\n--- Phase 1: Object Tracking ---")
        inference_state = video_predictor.init_state(video_path=frames_dir)

        img_path = os.path.join(frames_dir, frame_names[ann_frame_idx])
        image = Image.open(img_path)
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
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

        # Get masks per object
        image_predictor.set_image(np.array(image.convert("RGB")))
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

        # Register with video predictor
        OBJECTS = class_names
        if prompt_type == "mask":
            for obj_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )
        elif prompt_type == "box":
            for obj_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    box=box,
                )
        elif prompt_type == "point":
            all_sample_points = _sample_points_from_masks(masks, num_points=10)
            for obj_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels = np.ones(points.shape[0], dtype=np.int32)
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

        # Propagate tracking
        print("Propagating tracking...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        print(f"Tracking done: {len(video_segments)} frames")

        # Free SAM 2 memory
        del video_predictor, sam2_image_model, image_predictor, grounding_model
        torch.cuda.empty_cache()

        # ==== PHASE 2: Depth estimation + masked compositing ====
        print("\n--- Phase 2: Depth Estimation + Compositing ---")

        first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
        h, w = first_frame.shape[:2]

        masked_dir = os.path.join(tmpdir, "masked_depth")
        composite_dir = os.path.join(tmpdir, "composite")
        os.makedirs(masked_dir)
        os.makedirs(composite_dir)

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
        detections_per_frame = {}

        # Generate a stable color per object for mask outlines
        np.random.seed(42)
        obj_colors = {}
        for obj_id in ID_TO_OBJECTS:
            obj_colors[obj_id] = tuple(int(c) for c in np.random.randint(100, 255, 3))

        for fi in range(len(frame_names)):
            frame_path = os.path.join(frames_dir, frame_names[fi])
            frame_bgr = cv2.imread(frame_path)

            # Depth estimation
            with torch.cuda.amp.autocast():
                depth = depth_model.infer_image(frame_bgr, depth_input_size)

            # Normalize depth to 0-255
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)

            # Resize depth to frame size
            depth_norm = cv2.resize(depth_norm, (w, h))
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

            # Build combined segmentation mask for this frame
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            segments = video_segments.get(fi, {})
            object_ids = list(segments.keys())

            for obj_id, obj_mask in segments.items():
                # obj_mask shape: (1, mh, mw) — resize to frame size
                m = obj_mask.squeeze()
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, m)

            # --- Masked depth: black background, depth only on objects ---
            mask_3ch = np.stack([combined_mask] * 3, axis=-1)
            masked_depth = depth_color * mask_3ch
            cv2.imwrite(os.path.join(masked_dir, f"{fi:05d}.jpg"), masked_depth)

            # --- Composite: dimmed full depth + bright object depth + outlines ---
            dimmed = (depth_color * 0.2).astype(np.uint8)
            composite = np.where(mask_3ch, depth_color, dimmed)

            # Draw object outlines + labels
            for obj_id, obj_mask in segments.items():
                m = obj_mask.squeeze()
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = obj_colors.get(obj_id, (200, 200, 200))
                cv2.drawContours(composite, contours, -1, color, 2)

                # Label at top of object
                if contours:
                    all_pts = np.concatenate(contours)
                    x, y, bw, bh = cv2.boundingRect(all_pts)
                    label = ID_TO_OBJECTS.get(obj_id, f"obj_{obj_id}")
                    cv2.putText(
                        composite, label, (x, max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                    )

            cv2.imwrite(os.path.join(composite_dir, f"{fi:05d}.jpg"), composite)

            # Detection JSON
            detections_per_frame[fi] = [
                {
                    "object_id": int(oid),
                    "label": ID_TO_OBJECTS.get(oid, ""),
                }
                for oid in object_ids
            ]

            if fi % 20 == 0:
                print(f"  Frame {fi}/{len(frame_names)}")

        # Stitch videos
        def stitch(img_dir, out_path):
            files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
            first = cv2.imread(os.path.join(img_dir, files[0]))
            fh, fw = first.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (fw, fh))
            for f in files:
                writer.write(cv2.imread(os.path.join(img_dir, f)))
            writer.release()
            # Re-encode for compatibility
            final = out_path.replace(".mp4", "_final.mp4")
            os.system(f"ffmpeg -y -i {out_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p {final} 2>/dev/null")
            return final if os.path.exists(final) else out_path

        masked_path = stitch(masked_dir, os.path.join(tmpdir, "masked_depth.mp4"))
        composite_path = stitch(composite_dir, os.path.join(tmpdir, "composite.mp4"))

        with open(masked_path, "rb") as f:
            masked_video = f.read()
        with open(composite_path, "rb") as f:
            composite_video = f.read()

    torch.cuda.empty_cache()

    print(f"\nMasked depth video: {len(masked_video) / 1024 / 1024:.1f} MB")
    print(f"Composite video: {len(composite_video) / 1024 / 1024:.1f} MB")

    return {
        "masked_depth_video": masked_video,
        "composite_video": composite_video,
        "detections_json": json.dumps(detections_per_frame),
        "objects_detected": list(OBJECTS),
        "num_frames": len(frame_names),
        "source_fps": source_fps,
    }


@app.local_entrypoint()
def main(
    video_path: str,
    text_prompt: str = "person.",
    prompt_type: str = "mask",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    ann_frame_idx: int = 0,
    depth_input_size: int = 518,
):
    """
    Run object-aware depth on a video.

    Outputs to data/segmentation/:
      - {stem}_masked_depth.mp4   (depth only on segmented objects, black bg)
      - {stem}_composite.mp4      (dimmed full depth + bright object depth + outlines)
      - {stem}_seg_depth.json     (per-frame detections)
    """
    p = pathlib.Path(video_path).expanduser().resolve()
    print(f"Reading {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Prompt: {text_prompt!r}")

    result = segment_depth.remote(
        p.read_bytes(),
        text_prompt=text_prompt,
        prompt_type=prompt_type,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        ann_frame_idx=ann_frame_idx,
        depth_input_size=depth_input_size,
    )

    out_dir = pathlib.Path(__file__).parent.parent / "data" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = p.stem

    (out_dir / f"{stem}_masked_depth.mp4").write_bytes(result["masked_depth_video"])
    (out_dir / f"{stem}_composite.mp4").write_bytes(result["composite_video"])
    (out_dir / f"{stem}_seg_depth.json").write_text(result["detections_json"])

    print(f"\nWrote to {out_dir}/:")
    print(f"  {stem}_masked_depth.mp4 ({len(result['masked_depth_video']) / 1024 / 1024:.1f} MB)")
    print(f"  {stem}_composite.mp4 ({len(result['composite_video']) / 1024 / 1024:.1f} MB)")
    print(f"  {stem}_seg_depth.json")
    print(f"Objects: {result['objects_detected']}")
    print(f"Frames: {result['num_frames']} at {result['source_fps']:.1f} fps")
