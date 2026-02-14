"""
Grounded SAM 2 on Modal — Open-Set Object Tracking in Videos.

Uses Grounding DINO (HuggingFace) + SAM 2.1 to detect and track objects
in video given a text prompt. Returns annotated video + per-frame JSON masks.

Deploy:  modal deploy groundedsam2/app.py
Dev:     modal serve groundedsam2/app.py
Run:     modal run groundedsam2/app.py --video-path data/IMG_4717.MOV --text-prompt "person. car."
"""

import json
import pathlib

import modal

app = modal.App("groundedsam2")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

gsam2_image = (
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
    )
    # Install SAM 2 from source (includes configs needed by hydra)
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /opt/sam2",
        "cd /opt/sam2 && SAM2_BUILD_CUDA=0 pip install -e .",
    )
    # Clone Grounded-SAM-2 for its utility scripts
    .run_commands(
        "git clone --depth 1 https://github.com/IDEA-Research/Grounded-SAM-2.git /opt/gsam2",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
    })
    # Pre-download model weights into the image so cold starts are fast
    .run_commands(
        # SAM 2.1 large checkpoint
        "mkdir -p /opt/sam2_ckpts && "
        "curl -L -o /opt/sam2_ckpts/sam2.1_hiera_large.pt "
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        # Grounding DINO tiny from HuggingFace
        "python -c \""
        "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; "
        "AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
        "AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
        "print('Grounding DINO weights downloaded')\"",
        gpu="any",
    )
)


def _extract_frames(video_path: str, output_dir: str) -> tuple[list[str], float]:
    """Extract all frames from video as numbered JPEGs. Returns (frame_names, fps)."""
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
    """Sample points from binary masks for point-prompt mode."""
    import numpy as np

    n, h, w = masks.shape
    points = []
    for i in range(n):
        indices = np.argwhere(masks[i] == 1)
        indices = indices[:, ::-1]  # (y,x) -> (x,y)
        if len(indices) == 0:
            points.append(np.zeros((num_points, 2), dtype=np.float32))
            continue
        replace = len(indices) < num_points
        sampled = indices[np.random.choice(len(indices), num_points, replace=replace)]
        points.append(sampled)
    return np.array(points, dtype=np.float32)


def _create_video(image_dir: str, output_path: str, fps: float):
    """Stitch annotated frames into an mp4."""
    import cv2
    import os

    valid_ext = {".jpg", ".jpeg", ".png"}
    files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    )
    if not files:
        raise ValueError("No frames to stitch")
    first = cv2.imread(os.path.join(image_dir, files[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in files:
        writer.write(cv2.imread(os.path.join(image_dir, f)))
    writer.release()


@app.function(
    image=gsam2_image,
    gpu="A100",
    timeout=900,
    memory=32768,
)
def track_objects(
    video_bytes: bytes,
    text_prompt: str,
    prompt_type: str = "mask",
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    ann_frame_idx: int = 0,
) -> dict:
    """
    Detect and track objects in a video using Grounding DINO + SAM 2.

    Args:
        video_bytes: Raw video file content (.mov, .mp4, etc).
        text_prompt: Text prompt for object detection (e.g. "person. car.").
        prompt_type: How to prompt SAM 2 video predictor: "point", "box", or "mask".
        box_threshold: Confidence threshold for Grounding DINO boxes.
        text_threshold: Text similarity threshold for Grounding DINO.
        ann_frame_idx: Frame index to run initial detection on (default: 0).

    Returns:
        dict with keys:
            - "video": bytes of annotated mp4
            - "detections_json": per-frame detection metadata (JSON string)
            - "num_frames": total frames processed
            - "objects_detected": list of detected class names
            - "source_fps": original video FPS
    """
    import os
    import tempfile

    import cv2
    import numpy as np
    import supervision as sv
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    # Only use the pip-installed SAM2, NOT the one in /opt/gsam2
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda"

    # Enable bfloat16 + TF32 for speed
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Load models ---
    print("Loading SAM 2.1 models...")
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

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save video
        video_path = os.path.join(tmpdir, "input.mov")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Extract all frames
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir)
        print("Extracting frames...")
        frame_names, source_fps = _extract_frames(video_path, frames_dir)
        print(f"Extracted {len(frame_names)} frames at {source_fps:.1f} fps")

        if not frame_names:
            raise ValueError("No frames extracted from video")

        # --- Step 1: Init SAM 2 video predictor ---
        inference_state = video_predictor.init_state(video_path=frames_dir)

        # --- Step 2: Run Grounding DINO on the annotation frame ---
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
        # transformers >= 4.51 returns integer ids in "labels", use "text_labels" for strings
        class_names = results[0].get("text_labels", results[0].get("labels", []))
        # Ensure class_names are strings
        class_names = [str(n) for n in class_names]

        # Filter out empty/invalid detections
        valid_indices = [i for i, name in enumerate(class_names) if name.strip()]
        input_boxes = input_boxes[valid_indices] if valid_indices else np.empty((0, 4))
        confidences = [confidences[i] for i in valid_indices]
        class_names = [class_names[i] for i in valid_indices]

        print(f"Detected {len(class_names)} objects: {class_names}")
        print(f"Confidences: {[f'{c:.3f}' for c in confidences]}")

        if len(class_names) == 0:
            print("WARNING: No objects detected. Returning empty results.")
            with open(video_path, "rb") as f:
                raw = f.read()
            return {
                "video": raw,
                "detections_json": json.dumps({}),
                "num_frames": len(frame_names),
                "objects_detected": [],
                "source_fps": source_fps,
            }

        # --- Step 3: Get masks from SAM 2 image predictor ---
        image_predictor.set_image(np.array(image.convert("RGB")))

        # Process each box individually to avoid batch dimension issues
        all_masks = []
        for box in input_boxes:
            mask, score, logit = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],  # (1, 4)
                multimask_output=False,
            )
            # mask shape: (1, 1, H, W) or (1, H, W) — squeeze to (H, W)
            if mask.ndim == 4:
                mask = mask.squeeze(0).squeeze(0)
            elif mask.ndim == 3:
                mask = mask.squeeze(0)
            all_masks.append(mask)
        masks = np.stack(all_masks, axis=0)  # (N, H, W)

        # --- Step 4: Register prompts with video predictor ---
        OBJECTS = class_names
        if prompt_type == "point":
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
        elif prompt_type == "box":
            for obj_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    box=box,
                )
        elif prompt_type == "mask":
            for obj_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        # --- Step 5: Propagate tracking across all frames ---
        print("Propagating tracking...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        print(f"Tracking complete: {len(video_segments)} frames")

        # --- Step 6: Annotate frames and build detection JSON ---
        annotated_dir = os.path.join(tmpdir, "annotated")
        os.makedirs(annotated_dir)

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
        detections_per_frame = {}

        for frame_idx, segments in video_segments.items():
            img = cv2.imread(os.path.join(frames_dir, frame_names[frame_idx]))
            object_ids = list(segments.keys())
            frame_masks = np.concatenate(list(segments.values()), axis=0)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(frame_masks),
                mask=frame_masks,
                class_id=np.array(object_ids, dtype=np.int32),
            )

            # Annotate
            annotated = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections)
            annotated = sv.LabelAnnotator().annotate(
                annotated, detections=detections,
                labels=[ID_TO_OBJECTS[i] for i in object_ids],
            )
            annotated = sv.MaskAnnotator().annotate(scene=annotated, detections=detections)
            cv2.imwrite(os.path.join(annotated_dir, f"{frame_idx:05d}.jpg"), annotated)

            # Store detection metadata (bboxes only, masks are too large for JSON)
            detections_per_frame[frame_idx] = [
                {
                    "object_id": int(oid),
                    "label": ID_TO_OBJECTS[oid],
                    "bbox_xyxy": detections.xyxy[i].tolist(),
                }
                for i, oid in enumerate(object_ids)
            ]

        # --- Step 7: Stitch annotated frames into video ---
        output_video_path = os.path.join(tmpdir, "output.mp4")
        _create_video(annotated_dir, output_video_path, source_fps)

        with open(output_video_path, "rb") as f:
            video_out = f.read()

    print(f"Output video: {len(video_out) / 1024 / 1024:.1f} MB")
    return {
        "video": video_out,
        "detections_json": json.dumps(detections_per_frame),
        "num_frames": len(frame_names),
        "objects_detected": list(OBJECTS),
        "source_fps": source_fps,
    }


@app.local_entrypoint()
def main(
    video_path: str,
    text_prompt: str = "person.",
    prompt_type: str = "mask",
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    ann_frame_idx: int = 0,
):
    """
    Run Grounded SAM 2 tracking on a local video file.

    Args:
        video_path: Path to video file (.mov, .mp4).
        text_prompt: Objects to detect and track (e.g. "person. car.").
        prompt_type: SAM 2 prompt type: "point", "box", or "mask".
        box_threshold: Grounding DINO box confidence threshold.
        text_threshold: Grounding DINO text similarity threshold.
        ann_frame_idx: Frame index to run initial detection on.
    """
    p = pathlib.Path(video_path).expanduser().resolve()
    print(f"Reading {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Settings: text_prompt={text_prompt!r}, prompt_type={prompt_type}")

    result = track_objects.remote(
        p.read_bytes(),
        text_prompt=text_prompt,
        prompt_type=prompt_type,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        ann_frame_idx=ann_frame_idx,
    )

    # Save output video
    out_video = p.with_name(f"{p.stem}_tracked.mp4")
    out_video.write_bytes(result["video"])
    print(f"Wrote {out_video} ({len(result['video']) / 1024 / 1024:.1f} MB)")

    # Save detections JSON
    out_json = p.with_name(f"{p.stem}_detections.json")
    out_json.write_text(result["detections_json"])

    print(f"Processed {result['num_frames']} frames at {result['source_fps']:.1f} fps")
    print(f"Objects detected: {result['objects_detected']}")
    print(f"Detections saved to {out_json}")
