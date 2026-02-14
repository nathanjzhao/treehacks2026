"""
OpenFunGraph on Modal — Functional 3D Scene Graphs from Video.

Chains MapAnything (video → depth/poses) → OpenFunGraph (RGB-D → functional scene graph).

Deploy:  modal deploy openfungraph/app.py
Run:     modal run openfungraph/app.py --video-path ~/video.mov
"""

import pathlib
import modal

app = modal.App("openfungraph")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

# ---------------------------------------------------------------------------
# Image: MapAnything + OpenFunGraph + all dependencies
# ---------------------------------------------------------------------------

ofg_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git", "ffmpeg", "libgl1", "libglib2.0-0", "wget", "ninja-build",
        "libsm6", "libxext6", "libxrender-dev",
    )
    # ---- PyTorch 2.3.1 + CUDA 12.4 ----
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # ---- Common Python deps ----
    .pip_install(
        "numpy<2",
        "Pillow",
        "huggingface_hub",
        "einops",
        "safetensors",
        "opencv-python",
        "tqdm",
        "hydra-core",
        "omegaconf",
        "scipy",
        "requests",
        "trimesh",
        "matplotlib",
        "pydantic==2.10.6",
        "imageio",
        "open3d",
        "distinctipy",
        "tyro",
        "open_clip_torch",
        "openai",
        "h5py",
        "supervision",
        "transformers",
        "accelerate",
    )
    # ---- MapAnything ----
    .run_commands(
        "git clone https://github.com/facebookresearch/map-anything.git /opt/map-anything",
        "cd /opt/map-anything && pip install -e .",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    # Pre-download MapAnything weights
    .run_commands(
        "python -c \""
        "from mapanything.models import MapAnything; "
        "MapAnything.from_pretrained('facebook/map-anything'); "
        "print('MapAnything weights downloaded')\"",
        gpu="any",
    )
    # ---- PyTorch3D (build from source) ----
    .apt_install("g++")
    .run_commands(
        "pip install 'fvcore>=0.1.5' iopath",
        "git clone https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d",
        "cd /opt/pytorch3d && TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CXX=g++ pip install --no-build-isolation .",
        gpu="any",
    )
    # ---- ChamferDist ----
    .run_commands(
        "git clone https://github.com/krrish94/chamferdist.git /opt/chamferdist",
        "cd /opt/chamferdist && TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CXX=g++ pip install --no-build-isolation .",
        gpu="any",
    )
    # ---- GradSLAM (conceptfusion branch) ----
    .run_commands(
        "git clone https://github.com/gradslam/gradslam.git /opt/gradslam",
        "cd /opt/gradslam && git checkout conceptfusion && pip install -e .",
    )
    # ---- Grounded-SAM (GroundingDINO + SAM + RAM) ----
    .run_commands(
        "git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git /opt/Grounded-SAM",
        # SAM
        "pip install git+https://github.com/facebookresearch/segment-anything.git",
        # RAM
        "pip install git+https://github.com/xinyu1205/recognize-anything.git",
    )
    # GroundingDINO needs GPU for CUDA deformable attention ops
    # Build for all common architectures so kernels work on A100/H100/etc
    .run_commands(
        "cd /opt/Grounded-SAM/GroundingDINO && "
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9;9.0' CXX=g++ pip install --no-build-isolation -e .",
        gpu="any",
    )
    .env({
        "GSA_PATH": "/opt/Grounded-SAM",
    })
    # Download Grounded-SAM model weights
    .run_commands(
        "mkdir -p /opt/Grounded-SAM/weights",
        # SAM ViT-H
        "wget -q -O /opt/Grounded-SAM/weights/sam_vit_h_4b8939.pth "
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        # GroundingDINO
        "wget -q -O /opt/Grounded-SAM/weights/groundingdino_swint_ogc.pth "
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        # RAM
        "wget -q -O /opt/Grounded-SAM/weights/ram_swin_large_14m.pth "
        "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth",
    )
    # ---- LLaVA v1.6 ----
    .run_commands(
        "git clone https://github.com/haotian-liu/LLaVA.git /opt/LLaVA",
        "cd /opt/LLaVA && pip install --no-deps -e .",
        # Install LLaVA's unique deps without letting it downgrade torch/transformers
        "pip install sentencepiece shortuuid peft bitsandbytes markdown2 gradio==4.16.0",
    )
    # Pre-download LLaVA weights
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('liuhaotian/llava-v1.6-vicuna-7b'); "
        "print('LLaVA weights downloaded')\"",
    )
    # ---- OpenFunGraph ----
    .run_commands(
        "git clone https://github.com/ZhangCYG/OpenFunGraph.git /opt/OpenFunGraph",
        "cd /opt/OpenFunGraph && pip install -e .",
    )
    # ---- Repair: reinstall correct versions that LLaVA/MapAnything/gradio may have broken ----
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "fairscale",
        "einops>=0.8",
        "timm>=1.0.17",
        "open_clip_torch",
        "transformers==4.44.2",
        "accelerate",
        "huggingface_hub",
    )
    .env({
        "FG_FOLDER": "/opt/OpenFunGraph",
    })
    # Pre-download CLIP model
    .run_commands(
        "python -c \""
        "import open_clip; "
        "open_clip.create_model_and_transforms('ViT-H-14', 'laion2b_s32b_b79k'); "
        "print('CLIP downloaded')\"",
    )
)

# ---------------------------------------------------------------------------
# Helper: convert MapAnything output → OpenFunGraph-compatible files on disk
# ---------------------------------------------------------------------------

def _bridge_to_ofg(predictions: dict, workdir: str, orig_frames_dir: str):
    """
    Convert MapAnything predictions to the file layout OpenFunGraph expects.

    Creates:
      workdir/color/  — RGB PNGs
      workdir/depth/  — depth PNGs (uint16, millimeters)
      workdir/poses.txt — camera-to-world 4x4 matrices (one per line, flattened)
      workdir/intrinsics.txt — fx fy cx cy
    """
    import os
    import cv2
    import numpy as np

    color_dir = os.path.join(workdir, "color")
    depth_dir = os.path.join(workdir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    images = predictions["img_no_norm"]      # (N, H, W, 3) uint8 or float [0,255]
    depth_maps = predictions["depth_z"]      # (N, H, W, 1) or (N, H, W) float meters
    extrinsics = predictions["camera_poses"] # (N, 4, 4) camera-to-world
    intrinsics = predictions["intrinsics"]   # (N, 3, 3)

    # Squeeze depth if (N, H, W, 1)
    if depth_maps.ndim == 4 and depth_maps.shape[-1] == 1:
        depth_maps = depth_maps[..., 0]

    n_frames = images.shape[0]
    model_h, model_w = images.shape[1], images.shape[2]

    # Use original full-res frames from disk if available
    use_orig = os.path.isdir(orig_frames_dir) and len(os.listdir(orig_frames_dir)) >= n_frames

    for i in range(n_frames):
        used_orig = False
        if use_orig:
            orig_path = os.path.join(orig_frames_dir, f"{i:06d}.png")
            if os.path.exists(orig_path):
                orig_img = cv2.imread(orig_path)
                cv2.imwrite(os.path.join(color_dir, f"{i:06d}.png"), orig_img)
                target_h, target_w = orig_img.shape[0], orig_img.shape[1]
                used_orig = True
        if not used_orig:
            img = images[i]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(color_dir, f"{i:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            target_h, target_w = img.shape[0], img.shape[1]

        # Save depth as uint16 PNG (millimeters), resize to match color frame
        d = depth_maps[i]
        d_mm = (d * 1000).clip(0, 65535).astype(np.uint16)
        if d_mm.shape[0] != target_h or d_mm.shape[1] != target_w:
            d_mm = cv2.resize(d_mm, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(depth_dir, f"{i:06d}.png"), d_mm)

    # Save poses (camera-to-world, one 4x4 per frame, flattened 16 values per line)
    with open(os.path.join(workdir, "poses.txt"), "w") as f:
        for i in range(n_frames):
            ext = extrinsics[i]
            if ext.shape == (3, 4):
                ext = np.vstack([ext, [0, 0, 0, 1]])
            vals = ext.flatten().tolist()
            f.write(" ".join(f"{v:.8f}" for v in vals) + "\n")

    # Determine final output resolution
    if use_orig:
        sample = cv2.imread(os.path.join(orig_frames_dir, "000000.png"))
        h, w = sample.shape[0], sample.shape[1]
    else:
        h, w = model_h, model_w

    # Save intrinsics from first frame, scaled to output resolution
    K = intrinsics[0]
    scale_x = w / model_w
    scale_y = h / model_h
    fx = K[0, 0] * scale_x
    fy = K[1, 1] * scale_y
    cx = K[0, 2] * scale_x
    cy = K[1, 2] * scale_y
    with open(os.path.join(workdir, "intrinsics.txt"), "w") as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")
    print(f"[Bridge] {n_frames} frames ({w}x{h}) to {workdir}")
    return n_frames, h, w


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_mapanything(video_path: str, target_fps: int) -> tuple:
    """Stage 1: MapAnything inference — extract frames, run model, return predictions + frames_dir."""
    import os
    import cv2
    import torch
    import numpy as np

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda"

    print("[MapAnything] Loading model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)

    # Extract frames from video
    vs = cv2.VideoCapture(video_path)
    source_fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / target_fps))

    frames_dir = os.path.join(os.path.dirname(video_path), "extracted_frames")
    os.makedirs(frames_dir, exist_ok=True)

    image_paths = []
    count = 0
    frame_num = 0
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            path = os.path.join(frames_dir, f"{frame_num:06d}.png")
            cv2.imwrite(path, frame)
            image_paths.append(path)
            frame_num += 1
    vs.release()
    print(f"[MapAnything] Extracted {len(image_paths)} frames (interval={frame_interval})")

    # Load and run inference
    views = load_images(frames_dir)
    predictions = model.infer(
        views,
        memory_efficient_inference=True,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )

    # model.infer returns a list of per-view dicts — stack into batched arrays
    keys_to_stack = ["pts3d", "depth_z", "camera_poses", "intrinsics", "conf", "img_no_norm"]
    result = {}
    for key in keys_to_stack:
        tensors = [p[key] for p in predictions]
        stacked = torch.stack(tensors, dim=0) if isinstance(tensors[0], torch.Tensor) else np.stack(tensors, axis=0)
        if isinstance(stacked, torch.Tensor):
            arr = stacked.cpu().float().numpy()
        else:
            arr = stacked
        # Squeeze batch dim if each view had batch=1: (N, 1, ...) → (N, ...)
        if arr.ndim > 1 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        result[key] = arr
        print(f"  {key}: {arr.shape}")

    # Free GPU memory for next stages
    del model
    torch.cuda.empty_cache()

    print(f"[MapAnything] Inference complete — depth range: "
          f"{result['depth_z'].min():.3f} to {result['depth_z'].max():.3f} m")
    return result, frames_dir


def _run_detection(workdir: str, n_frames: int, h: int, w: int):
    """Stage 2: Grounded-SAM + RAM object detection on each frame."""
    import os
    import sys
    import gzip
    import pickle
    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, "/opt/Grounded-SAM")
    sys.path.insert(0, "/opt/OpenFunGraph")

    # Load models
    from segment_anything import sam_model_registry, SamPredictor
    from groundingdino.util.inference import Model as GroundingDINOModel
    import open_clip

    weights_dir = "/opt/Grounded-SAM/weights"

    print("[Detection] Loading SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=os.path.join(weights_dir, "sam_vit_h_4b8939.pth"))
    sam = sam.to("cuda")
    sam_predictor = SamPredictor(sam)

    print("[Detection] Loading GroundingDINO...")
    gdino_config = "/opt/Grounded-SAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_ckpt = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
    gdino_model = GroundingDINOModel(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_ckpt,
        device="cuda",
    )

    print("[Detection] Loading RAM...")
    from ram.models import ram
    from ram import inference_ram
    ram_model = ram(
        pretrained=os.path.join(weights_dir, "ram_swin_large_14m.pth"),
        image_size=384,
        vit="swin_l",
    )
    ram_model.eval().to("cuda")

    print("[Detection] Loading CLIP...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to("cuda").eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # Process each frame
    color_dir = os.path.join(workdir, "color")
    det_dir = os.path.join(workdir, "detections")
    os.makedirs(det_dir, exist_ok=True)

    box_threshold = 0.25
    text_threshold = 0.25
    mask_conf_threshold = 0.3      # minimum detection confidence to keep
    max_bbox_area_ratio = 0.9      # reject boxes covering >90% of image

    from torchvision import transforms
    from groundingdino.util.inference import predict as gdino_predict
    import groundingdino.datasets.transforms as T
    from PIL import Image as PILImage

    ram_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gd_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for i in range(n_frames):
        img_path = os.path.join(color_dir, f"{i:06d}.png")
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # RAM tagging
        ram_input = ram_transform(image_rgb).unsqueeze(0).to("cuda")
        with torch.no_grad():
            tags, _ = inference_ram(ram_input, ram_model)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        text_prompt = ". ".join(tag_list) + "." if tag_list else "object."

        # GroundingDINO detection
        pil_img = PILImage.fromarray(image_rgb)
        gd_image, _ = gd_transform(pil_img, None)

        boxes, logits, phrases = gdino_predict(
            model=gdino_model.model,
            image=gd_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cuda",
        )

        if len(boxes) == 0:
            result = {
                "xyxy": np.zeros((0, 4)),
                "confidence": np.array([]),
                "mask": np.zeros((0, h, w), dtype=bool),
                "classes": [],
                "image_feats": np.zeros((0, 1024)),
                "text_feats": np.zeros((0, 1024)),
                "tagging_text_prompt": text_prompt,
            }
            with gzip.open(os.path.join(det_dir, f"{i:06d}.pkl.gz"), "wb") as f:
                pickle.dump(result, f)
            continue

        # Convert boxes from cx,cy,w,h normalized → xyxy pixel
        img_h, img_w = image_rgb.shape[:2]
        img_area = img_h * img_w
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h
        boxes_np = boxes_xyxy.cpu().numpy()

        # Filter: confidence threshold + reject oversized bounding boxes
        conf_np = logits.cpu().numpy()
        box_areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        keep = (conf_np >= mask_conf_threshold) & (box_areas < max_bbox_area_ratio * img_area)
        if not keep.any():
            result = {
                "xyxy": np.zeros((0, 4)),
                "confidence": np.array([]),
                "mask": np.zeros((0, h, w), dtype=bool),
                "classes": [],
                "image_feats": np.zeros((0, 1024)),
                "text_feats": np.zeros((0, 1024)),
                "tagging_text_prompt": text_prompt,
            }
            with gzip.open(os.path.join(det_dir, f"{i:06d}.pkl.gz"), "wb") as f:
                pickle.dump(result, f)
            continue

        boxes_xyxy = boxes_xyxy[torch.from_numpy(keep)]
        boxes_np = boxes_np[keep]
        logits = logits[torch.from_numpy(keep)]
        phrases = [phrases[j] for j in range(len(phrases)) if keep[j]]

        # SAM segmentation
        sam_predictor.set_image(image_rgb)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy.to("cuda"), image_rgb.shape[:2]
        )
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks_np = masks.squeeze(1).cpu().numpy()  # (N, H, W)

        # CLIP features
        image_feats_list = []
        text_feats_list = []
        for j in range(len(boxes_np)):
            x1, y1, x2, y2 = boxes_np[j].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                image_feats_list.append(np.zeros(1024))
                text_feats_list.append(np.zeros(1024))
                continue
            crop_pil = PILImage.fromarray(crop)
            crop_tensor = clip_preprocess(crop_pil).unsqueeze(0).to("cuda")
            with torch.no_grad():
                img_feat = clip_model.encode_image(crop_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            image_feats_list.append(img_feat.cpu().numpy().flatten())

            phrase = phrases[j] if j < len(phrases) else "object"
            text_tokens = clip_tokenizer([phrase]).to("cuda")
            with torch.no_grad():
                txt_feat = clip_model.encode_text(text_tokens)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            text_feats_list.append(txt_feat.cpu().numpy().flatten())

        result = {
            "xyxy": boxes_np,
            "confidence": logits.cpu().numpy(),
            "mask": masks_np,
            "classes": list(phrases),
            "image_feats": np.stack(image_feats_list),
            "text_feats": np.stack(text_feats_list),
            "tagging_text_prompt": text_prompt,
        }

        with gzip.open(os.path.join(det_dir, f"{i:06d}.pkl.gz"), "wb") as f:
            pickle.dump(result, f)

        if (i + 1) % 5 == 0:
            print(f"[Detection] Frame {i+1}/{n_frames}: {len(boxes_np)} detections")

    # Free GPU memory
    del sam, sam_predictor, gdino_model, ram_model, clip_model
    torch.cuda.empty_cache()

    print(f"[Detection] Completed {n_frames} frames")


def _compute_overlap(min_a, max_a, min_b, max_b):
    """Compute overlap = intersection_vol / min(vol_a, vol_b). More forgiving than IoU."""
    import numpy as np
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_vol = np.prod(np.maximum(inter_max - inter_min, 0))
    vol_a = np.prod(np.maximum(max_a - min_a, 1e-8))
    vol_b = np.prod(np.maximum(max_b - min_b, 1e-8))
    return inter_vol / max(min(vol_a, vol_b), 1e-8)


def _dbscan_filter(pts, cols, eps=0.1, min_points=10):
    """Remove outlier points using DBSCAN, keep the largest cluster."""
    import numpy as np
    import open3d as o3d

    if len(pts) < min_points:
        return pts, cols

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    if len(labels) == 0 or labels.max() < 0:
        return pts, cols

    # Keep the largest cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return pts, cols
    best_label = unique_labels[counts.argmax()]
    mask = labels == best_label
    return pts[mask], cols[mask]


def _merge_objects(objects, overlap_thresh=0.9, visual_thresh=0.75, text_thresh=0.7):
    """
    Post-processing merge pass: merge duplicate objects that have high
    spatial overlap AND high visual/text similarity.
    """
    import numpy as np

    if len(objects) <= 1:
        return objects

    merged = [True] * len(objects)  # track which objects are still alive

    for i in range(len(objects)):
        if not merged[i]:
            continue
        for j in range(i + 1, len(objects)):
            if not merged[j]:
                continue

            obj_i = objects[i]
            obj_j = objects[j]

            # Spatial overlap
            min_i, max_i = obj_i["pcd_np"].min(axis=0), obj_i["pcd_np"].max(axis=0)
            min_j, max_j = obj_j["pcd_np"].min(axis=0), obj_j["pcd_np"].max(axis=0)
            overlap = _compute_overlap(min_i, max_i, min_j, max_j)

            if overlap < overlap_thresh:
                continue

            # Visual similarity (CLIP image features)
            vis_sim = np.dot(obj_i["clip_ft"], obj_j["clip_ft"]) / (
                np.linalg.norm(obj_i["clip_ft"]) * np.linalg.norm(obj_j["clip_ft"]) + 1e-8
            )
            if vis_sim < visual_thresh:
                continue

            # Text similarity (CLIP text features)
            txt_sim = np.dot(obj_i["text_ft"], obj_j["text_ft"]) / (
                np.linalg.norm(obj_i["text_ft"]) * np.linalg.norm(obj_j["text_ft"]) + 1e-8
            )
            if txt_sim < text_thresh:
                continue

            # Merge j into i
            obj_i["pcd_np"] = np.vstack([obj_i["pcd_np"], obj_j["pcd_np"]])
            obj_i["pcd_color_np"] = np.vstack([obj_i["pcd_color_np"], obj_j["pcd_color_np"]])
            obj_i["n_detections"] += obj_j["n_detections"]
            # Weighted average of features
            w_i = obj_i["n_detections"] - obj_j["n_detections"]
            w_j = obj_j["n_detections"]
            total = w_i + w_j
            obj_i["clip_ft"] = (obj_i["clip_ft"] * w_i + obj_j["clip_ft"] * w_j) / total
            obj_i["text_ft"] = (obj_i["text_ft"] * w_i + obj_j["text_ft"] * w_j) / total
            merged[j] = False
            print(f"  Merged object '{obj_j.get('class_name')}' into '{obj_i.get('class_name')}'")

    return [obj for obj, alive in zip(objects, merged) if alive]


def _run_3d_fusion(workdir: str, n_frames: int, h: int, w: int) -> list:
    """Stage 3: Fuse 2D detections into 3D objects using depth + poses."""
    import os
    import gzip
    import pickle
    import cv2
    import numpy as np
    import torch

    color_dir = os.path.join(workdir, "color")
    depth_dir = os.path.join(workdir, "depth")
    det_dir = os.path.join(workdir, "detections")

    # Load poses and intrinsics
    poses = []
    with open(os.path.join(workdir, "poses.txt")) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(4, 4))

    with open(os.path.join(workdir, "intrinsics.txt")) as f:
        fx, fy, cx, cy = map(float, f.read().strip().split())
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 3D object list: each object has points, colors, clip features, class name
    objects_3d = []
    sim_threshold = 1.2
    phys_bias = 0.5
    min_projected_points = 16     # minimum valid 3D points per detection
    denoise_interval = 20         # run intermediate denoising every N frames

    for i in range(n_frames):
        det_path = os.path.join(det_dir, f"{i:06d}.pkl.gz")
        if not os.path.exists(det_path):
            continue
        with gzip.open(det_path, "rb") as f:
            det = pickle.load(f)

        if len(det["xyxy"]) == 0:
            continue

        # Load depth
        depth_path = os.path.join(depth_dir, f"{i:06d}.png")
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            continue
        depth_m = depth_mm.astype(np.float32) / 1000.0

        # Load color
        color_path = os.path.join(color_dir, f"{i:06d}.png")
        color_bgr = cv2.imread(color_path)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        pose = poses[i]  # camera-to-world 4x4

        n_det = len(det["xyxy"])
        for j in range(n_det):
            mask = det["mask"][j]  # (H, W) bool
            if mask.sum() < min_projected_points:
                continue

            # Backproject masked pixels to 3D
            ys, xs = np.where(mask)
            zs = depth_m[ys, xs]
            valid = zs > 0.01
            if valid.sum() < min_projected_points:
                continue

            xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]

            # Unproject: pixel → camera coords
            x_cam = (xs_v - cx) * zs_v / fx
            y_cam = (ys_v - cy) * zs_v / fy
            pts_cam = np.stack([x_cam, y_cam, zs_v], axis=-1)  # (M, 3)

            # Camera → world
            pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
            pts_world = (pose @ pts_h.T).T[:, :3]

            # Colors
            cols = color_rgb[ys_v, xs_v]  # (M, 3)

            clip_feat = det["image_feats"][j]
            text_feat = det["text_feats"][j]
            class_name = det["classes"][j] if j < len(det["classes"]) else "object"

            # Try to match to existing object using OVERLAP (not IoU)
            best_score = -1
            best_idx = -1
            new_min = pts_world.min(axis=0)
            new_max = pts_world.max(axis=0)

            for k, obj in enumerate(objects_3d):
                obj_min = obj["pcd_np"].min(axis=0)
                obj_max = obj["pcd_np"].max(axis=0)

                # Spatial similarity: overlap = intersection / min(vol_a, vol_b)
                overlap = _compute_overlap(obj_min, obj_max, new_min, new_max)

                # Visual similarity: CLIP cosine similarity
                cos_sim = np.dot(clip_feat, obj["clip_ft"]) / (
                    np.linalg.norm(clip_feat) * np.linalg.norm(obj["clip_ft"]) + 1e-8
                )

                score = (1 + phys_bias) * overlap + (1 - phys_bias) * cos_sim

                if score > best_score:
                    best_score = score
                    best_idx = k

            if best_score >= sim_threshold and best_idx >= 0:
                # Merge into existing object
                obj = objects_3d[best_idx]
                obj["pcd_np"] = np.vstack([obj["pcd_np"], pts_world])
                obj["pcd_color_np"] = np.vstack([obj["pcd_color_np"], cols])
                obj["n_detections"] += 1
                # Running average of CLIP features
                n = obj["n_detections"]
                obj["clip_ft"] = obj["clip_ft"] * ((n - 1) / n) + clip_feat * (1 / n)
                obj["text_ft"] = obj["text_ft"] * ((n - 1) / n) + text_feat * (1 / n)
            else:
                # Create new object
                objects_3d.append({
                    "pcd_np": pts_world,
                    "pcd_color_np": cols,
                    "clip_ft": clip_feat.copy(),
                    "text_ft": text_feat.copy(),
                    "class_name": class_name,
                    "n_detections": 1,
                })

        # Intermediate denoising pass to keep memory manageable
        if (i + 1) % denoise_interval == 0 and len(objects_3d) > 0:
            print(f"[3D Fusion] Denoising at frame {i+1}...")
            for obj in objects_3d:
                if len(obj["pcd_np"]) > 5000:
                    voxel_size = 0.01
                    quantized = np.floor(obj["pcd_np"] / voxel_size).astype(np.int64)
                    _, unique_idx = np.unique(quantized, axis=0, return_index=True)
                    obj["pcd_np"] = obj["pcd_np"][unique_idx]
                    obj["pcd_color_np"] = obj["pcd_color_np"][unique_idx]

        if (i + 1) % 5 == 0:
            print(f"[3D Fusion] Frame {i+1}/{n_frames}: {len(objects_3d)} objects so far")

    # ---- Post-processing ----
    min_detections = 5  # require at least 5 frame observations

    # Step 1: Filter by minimum detections
    filtered = [obj for obj in objects_3d if obj["n_detections"] >= min_detections]
    print(f"[3D Fusion] After min_detections filter: {len(filtered)} (from {len(objects_3d)})")

    # Step 2: Voxel downsample
    for obj in filtered:
        pts = obj["pcd_np"]
        cols = obj["pcd_color_np"]
        voxel_size = 0.01
        quantized = np.floor(pts / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(quantized, axis=0, return_index=True)
        obj["pcd_np"] = pts[unique_idx]
        obj["pcd_color_np"] = cols[unique_idx]

    # Step 3: DBSCAN outlier removal
    print("[3D Fusion] Running DBSCAN outlier removal...")
    for obj in filtered:
        obj["pcd_np"], obj["pcd_color_np"] = _dbscan_filter(
            obj["pcd_np"], obj["pcd_color_np"], eps=0.1, min_points=10
        )

    # Step 4: Remove objects with too few points after DBSCAN
    filtered = [obj for obj in filtered if len(obj["pcd_np"]) >= 50]

    # Step 5: Merge duplicate objects
    print("[3D Fusion] Running merge pass...")
    filtered = _merge_objects(
        filtered,
        overlap_thresh=0.9,
        visual_thresh=0.75,
        text_thresh=0.7,
    )

    print(f"[3D Fusion] Final: {len(filtered)} objects")
    return filtered


def _run_part_detection(workdir: str, n_frames: int, h: int, w: int, objects_3d: list) -> list:
    """Stage 4: Detect interactive parts (handles, knobs, buttons) on frames."""
    import os
    import gzip
    import pickle
    import cv2
    import numpy as np
    import torch

    sys_path_needed = ["/opt/Grounded-SAM", "/opt/OpenFunGraph"]
    import sys
    for p in sys_path_needed:
        if p not in sys.path:
            sys.path.insert(0, p)

    from segment_anything import sam_model_registry, SamPredictor
    from groundingdino.util.inference import Model as GroundingDINOModel

    weights_dir = "/opt/Grounded-SAM/weights"

    print("[Parts] Loading SAM + GroundingDINO...")
    sam = sam_model_registry["vit_h"](checkpoint=os.path.join(weights_dir, "sam_vit_h_4b8939.pth"))
    sam = sam.to("cuda")
    sam_predictor = SamPredictor(sam)

    gdino_config = "/opt/Grounded-SAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_ckpt = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
    gdino_model = GroundingDINOModel(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_ckpt,
        device="cuda",
    )

    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to("cuda").eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    part_prompt = "handle. button. knob. drawer. switch. remote. lever."
    box_threshold = 0.15
    text_threshold = 0.15
    mask_conf_threshold = 0.15    # lower than objects — parts are subtle

    from groundingdino.util.inference import predict as gdino_predict
    import groundingdino.datasets.transforms as T
    from PIL import Image as PILImage

    gd_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    color_dir = os.path.join(workdir, "color")
    depth_dir = os.path.join(workdir, "depth")

    # Load poses and intrinsics
    poses = []
    with open(os.path.join(workdir, "poses.txt")) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(4, 4))

    with open(os.path.join(workdir, "intrinsics.txt")) as f:
        fx, fy, cx, cy = map(float, f.read().strip().split())

    parts_3d = []
    sim_threshold = 1.2
    phys_bias = 0.5

    # Process EVERY frame (not every other — quality > speed)
    for i in range(n_frames):
        img_path = os.path.join(color_dir, f"{i:06d}.png")
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        pil_img = PILImage.fromarray(image_rgb)
        gd_image, _ = gd_transform(pil_img, None)

        boxes, logits, phrases = gdino_predict(
            model=gdino_model.model,
            image=gd_image,
            caption=part_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cuda",
        )

        if len(boxes) == 0:
            continue

        img_h, img_w = image_rgb.shape[:2]
        img_area = img_h * img_w
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h
        boxes_np = boxes_xyxy.cpu().numpy()

        # Filter: parts should be small (< 10% of image area) + confidence threshold
        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        conf_np = logits.cpu().numpy()
        keep = (areas < 0.1 * img_area) & (conf_np >= mask_conf_threshold)
        if not keep.any():
            continue
        boxes_xyxy = boxes_xyxy[torch.from_numpy(keep)]
        boxes_np = boxes_np[keep]
        logits_kept = logits[torch.from_numpy(keep)]
        phrases_kept = [phrases[j] for j in range(len(phrases)) if keep[j]]

        # SAM segmentation
        sam_predictor.set_image(image_rgb)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy.to("cuda"), image_rgb.shape[:2]
        )
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks_np = masks.squeeze(1).cpu().numpy()

        # Load depth
        depth_path = os.path.join(depth_dir, f"{i:06d}.png")
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            continue
        depth_m = depth_mm.astype(np.float32) / 1000.0
        color_float = image_rgb.astype(np.float32) / 255.0
        pose = poses[i]

        for j in range(len(boxes_np)):
            mask = masks_np[j]
            if mask.sum() < 5:
                continue

            ys, xs = np.where(mask)
            zs = depth_m[ys, xs]
            valid = zs > 0.01
            if valid.sum() < 5:
                continue

            xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
            x_cam = (xs_v - cx) * zs_v / fx
            y_cam = (ys_v - cy) * zs_v / fy
            pts_cam = np.stack([x_cam, y_cam, zs_v], axis=-1)
            pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
            pts_world = (pose @ pts_h.T).T[:, :3]
            cols = color_float[ys_v, xs_v]

            # CLIP feature
            x1, y1, x2, y2 = boxes_np[j].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                clip_feat = np.zeros(1024)
            else:
                crop_pil = PILImage.fromarray(crop)
                crop_tensor = clip_preprocess(crop_pil).unsqueeze(0).to("cuda")
                with torch.no_grad():
                    img_feat = clip_model.encode_image(crop_tensor)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                clip_feat = img_feat.cpu().numpy().flatten()

            class_name = phrases_kept[j] if j < len(phrases_kept) else "part"

            # Try to match to existing part using OVERLAP (not IoU)
            best_score = -1
            best_idx = -1
            new_min = pts_world.min(axis=0)
            new_max = pts_world.max(axis=0)

            for k, part in enumerate(parts_3d):
                part_min = part["pcd_np"].min(axis=0)
                part_max = part["pcd_np"].max(axis=0)

                overlap = _compute_overlap(part_min, part_max, new_min, new_max)
                cos_sim = np.dot(clip_feat, part["clip_ft"]) / (
                    np.linalg.norm(clip_feat) * np.linalg.norm(part["clip_ft"]) + 1e-8
                )
                score = (1 + phys_bias) * overlap + (1 - phys_bias) * cos_sim
                if score > best_score:
                    best_score = score
                    best_idx = k

            if best_score >= sim_threshold and best_idx >= 0:
                part = parts_3d[best_idx]
                part["pcd_np"] = np.vstack([part["pcd_np"], pts_world])
                part["pcd_color_np"] = np.vstack([part["pcd_color_np"], cols])
                part["n_detections"] += 1
                n = part["n_detections"]
                part["clip_ft"] = part["clip_ft"] * ((n - 1) / n) + clip_feat * (1 / n)
            else:
                parts_3d.append({
                    "pcd_np": pts_world,
                    "pcd_color_np": cols,
                    "clip_ft": clip_feat.copy(),
                    "class_name": class_name,
                    "n_detections": 1,
                })

        if (i + 1) % 5 == 0:
            print(f"[Parts] Frame {i+1}/{n_frames}: {len(parts_3d)} parts so far")

    # ---- Post-processing ----
    min_part_detections = 2

    # Step 1: Filter by min detections
    filtered = [p for p in parts_3d if p["n_detections"] >= min_part_detections]
    print(f"[Parts] After min_detections filter: {len(filtered)} (from {len(parts_3d)})")

    # Step 2: Voxel downsample (5mm for parts — finer than objects)
    for part in filtered:
        pts = part["pcd_np"]
        cols = part["pcd_color_np"]
        voxel_size = 0.005
        quantized = np.floor(pts / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(quantized, axis=0, return_index=True)
        part["pcd_np"] = pts[unique_idx]
        part["pcd_color_np"] = cols[unique_idx]

    # Step 3: DBSCAN outlier removal (tighter eps for small parts)
    print("[Parts] Running DBSCAN outlier removal...")
    for part in filtered:
        part["pcd_np"], part["pcd_color_np"] = _dbscan_filter(
            part["pcd_np"], part["pcd_color_np"], eps=0.05, min_points=5
        )

    # Step 4: Remove parts with too few points
    filtered = [p for p in filtered if len(p["pcd_np"]) >= 10]

    del sam, sam_predictor, gdino_model, clip_model
    torch.cuda.empty_cache()

    print(f"[Parts] Final: {len(filtered)} parts (from {len(parts_3d)} pre-filter)")
    return filtered


def _run_graph_construction(
    objects_3d: list,
    parts_3d: list,
    openai_model: str = "gpt-4o",
) -> list:
    """Stage 5: Build functional scene graph edges using GPT."""
    import numpy as np
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var
    edges = []

    # Build node descriptions
    obj_descs = []
    for i, obj in enumerate(objects_3d):
        centroid = obj["pcd_np"].mean(axis=0)
        obj_descs.append(f"Object {i}: {obj.get('class_name', 'unknown')} at ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")

    part_descs = []
    for j, part in enumerate(parts_3d):
        centroid = part["pcd_np"].mean(axis=0)
        part_descs.append(f"Part {j}: {part.get('class_name', 'unknown')} at ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")

    # --- Local relationships (object-part spatial overlap) ---
    print("[Graph] Inferring local relationships...")
    for j, part in enumerate(parts_3d):
        part_center = part["pcd_np"].mean(axis=0)

        # Find closest object
        best_dist = float("inf")
        best_obj_idx = -1
        for i, obj in enumerate(objects_3d):
            obj_center = obj["pcd_np"].mean(axis=0)
            dist = np.linalg.norm(part_center - obj_center)
            if dist < best_dist:
                best_dist = dist
                best_obj_idx = i

        if best_obj_idx < 0 or best_dist > 1.0:
            continue

        obj = objects_3d[best_obj_idx]
        prompt = (
            f"An indoor scene contains:\n"
            f"- {obj.get('class_name', 'object')} (Object {best_obj_idx})\n"
            f"- {part.get('class_name', 'part')} (Part {j}) located {best_dist:.2f}m away\n\n"
            f"Is there a functional relationship between the part and the object? "
            f"If yes, describe it in 3-5 words (e.g., 'pulling handle opens door'). "
            f"If no, respond with 'none'. Only output the relationship or 'none'."
        )

        try:
            resp = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
            )
            answer = resp.choices[0].message.content.strip().lower()
            if answer != "none" and len(answer) > 2:
                edges.append((best_obj_idx, -1, j, answer, 0.8))
                print(f"  Local edge: Object {best_obj_idx} ({obj.get('class_name')}) <-> Part {j} ({part.get('class_name')}): {answer}")
        except Exception as e:
            print(f"  GPT error (local): {e}")

    # --- Remote relationships (object-object functional links) ---
    print("[Graph] Inferring remote relationships...")
    scene_desc = "\n".join(obj_descs + part_descs)
    prompt = (
        f"Given this indoor scene:\n{scene_desc}\n\n"
        f"List all plausible remote functional relationships between objects "
        f"(e.g., 'switch controls light', 'remote controls TV'). "
        f"Output as a JSON array of objects with fields: "
        f"source_idx (int), target_idx (int), description (string), confidence (0.0-1.0). "
        f"Only include relationships with confidence >= 0.5. "
        f"If no remote relationships exist, return an empty array []."
    )

    try:
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
        # Parse JSON
        import json
        # Extract JSON from possible markdown code blocks
        if "```" in answer:
            answer = answer.split("```")[1]
            if answer.startswith("json"):
                answer = answer[4:]
            answer = answer.strip()
        remote_edges = json.loads(answer)
        for re_edge in remote_edges:
            src = re_edge.get("source_idx", -1)
            tgt = re_edge.get("target_idx", -1)
            desc = re_edge.get("description", "")
            conf = re_edge.get("confidence", 0.5)
            if 0 <= src < len(objects_3d) and 0 <= tgt < len(objects_3d) and src != tgt:
                edges.append((src, -1, tgt, desc, conf))
                print(f"  Remote edge: Object {src} -> Object {tgt}: {desc} ({conf:.2f})")
    except Exception as e:
        print(f"  GPT error (remote): {e}")

    print(f"[Graph] Total edges: {len(edges)}")
    return edges


# ---------------------------------------------------------------------------
# Modal function: full pipeline
# ---------------------------------------------------------------------------

with ofg_image.imports():
    import os
    import sys
    import json


@app.function(
    image=ofg_image,
    gpu="A100-80GB",
    timeout=3600,
    memory=65536,
    secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])],
)
def run_pipeline(
    video_bytes: bytes,
    target_fps: int = 5,
    conf_thres: float = 50.0,
    openai_model: str = "gpt-4o",
) -> bytes:
    """
    Full pipeline: Video → MapAnything → OpenFunGraph → Scene Graph.

    Returns JSON bytes with objects, parts, edges.
    """
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input video
        video_path = os.path.join(tmpdir, "input.mov")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Stage 1: MapAnything
        print("=" * 60)
        print("STAGE 1: MapAnything Inference")
        print("=" * 60)
        predictions, frames_dir = _run_mapanything(video_path, target_fps)

        # Bridge MapAnything → OpenFunGraph file format
        ofg_dir = os.path.join(tmpdir, "ofg_input")
        os.makedirs(ofg_dir, exist_ok=True)
        n_frames, h, w = _bridge_to_ofg(predictions, ofg_dir, frames_dir)

        # Stage 2: Object Detection
        print("=" * 60)
        print("STAGE 2: Object Detection (RAM + GroundingDINO + SAM)")
        print("=" * 60)
        _run_detection(ofg_dir, n_frames, h, w)

        # Stage 3: 3D Fusion
        print("=" * 60)
        print("STAGE 3: 3D Object Fusion")
        print("=" * 60)
        objects_3d = _run_3d_fusion(ofg_dir, n_frames, h, w)

        # Stage 4: Part Detection
        print("=" * 60)
        print("STAGE 4: Interactive Part Detection")
        print("=" * 60)
        parts_3d = _run_part_detection(ofg_dir, n_frames, h, w, objects_3d)

        # Stage 5: Graph Construction
        print("=" * 60)
        print("STAGE 5: Functional Graph Construction (GPT)")
        print("=" * 60)
        edges = _run_graph_construction(objects_3d, parts_3d, openai_model)

        # Package results as JSON (avoids numpy pickle deserialization issues locally)
        def _serialize_obj(obj):
            return {
                "pcd_np": obj["pcd_np"].tolist(),
                "pcd_color_np": obj["pcd_color_np"].tolist(),
                "clip_ft": obj["clip_ft"].tolist(),
                "text_ft": obj.get("text_ft", np.zeros(1024)).tolist(),
                "class_name": obj.get("class_name", "unknown"),
                "n_detections": obj["n_detections"],
            }

        # Include camera poses for visualization
        cam_poses = predictions["camera_poses"]  # (N, 4, 4) cam-to-world
        cam_intrinsics = predictions["intrinsics"]  # (N, 3, 3)

        result = {
            "objects": [_serialize_obj(o) for o in objects_3d],
            "parts": [_serialize_obj(p) for p in parts_3d],
            "edges": edges,
            "n_frames": n_frames,
            "camera_poses": cam_poses.tolist(),
            "camera_intrinsics": cam_intrinsics.tolist(),
        }

        print("=" * 60)
        print(f"DONE: {len(objects_3d)} objects, {len(parts_3d)} parts, {len(edges)} edges")
        print("=" * 60)

        return json.dumps(result).encode("utf-8")


@app.local_entrypoint()
def main(
    video_path: str,
    fps: int = 5,
    conf: float = 50.0,
    openai_model: str = "gpt-4o",
):
    """Run OpenFunGraph pipeline on a local video. Saves scene graph as .json."""
    import pathlib
    import json

    p = pathlib.Path(video_path).expanduser().resolve()
    print(f"Reading {p.name} ({p.stat().st_size / 1024:.0f} KB)")
    print(f"Settings: fps={fps}, conf={conf}, model={openai_model}")

    result_bytes = run_pipeline.remote(
        p.read_bytes(),
        target_fps=fps,
        conf_thres=conf,
        openai_model=openai_model,
    )

    out = p.with_suffix(".json")
    out.write_bytes(result_bytes)

    result = json.loads(result_bytes)
    print(f"Wrote {out} ({len(result_bytes) / 1024:.1f} KB)")
    print(f"Scene graph: {len(result['objects'])} objects, {len(result['parts'])} parts, {len(result['edges'])} edges")
