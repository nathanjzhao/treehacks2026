"""
OpenFunGraph on Modal — Functional 3D Scene Graphs from Video.

Chains VGGT (video → depth/poses) → OpenFunGraph (RGB-D → functional scene graph).

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
# Image: VGGT + OpenFunGraph + all dependencies
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
    # ---- VGGT ----
    .run_commands(
        "git clone https://github.com/facebookresearch/vggt.git /opt/vggt",
        "cd /opt/vggt && pip install -e .",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
    })
    # Pre-download VGGT weights
    .run_commands(
        "python -c \""
        "from vggt.models.vggt import VGGT; "
        "VGGT.from_pretrained('facebook/VGGT-1B'); "
        "print('VGGT weights downloaded')\"",
        gpu="any",
    )
    # ---- PyTorch3D (build from source) ----
    .run_commands(
        "pip install 'fvcore>=0.1.5' iopath",
        "git clone https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d",
        "cd /opt/pytorch3d && pip install -e .",
        gpu="any",
    )
    # ---- ChamferDist ----
    .run_commands(
        "git clone https://github.com/krrish94/chamferdist.git /opt/chamferdist",
        "cd /opt/chamferdist && pip install -e .",
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
        # GroundingDINO
        "cd /opt/Grounded-SAM/GroundingDINO && pip install -e .",
        # SAM
        "pip install git+https://github.com/facebookresearch/segment-anything.git",
        # RAM
        "pip install git+https://github.com/xinyu1205/recognize-anything.git",
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
        "cd /opt/LLaVA && pip install -e .",
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
# Helper: convert VGGT output → OpenFunGraph-compatible files on disk
# ---------------------------------------------------------------------------

def _bridge_vggt_to_ofg(predictions: dict, workdir: str):
    """
    Convert VGGT predictions to the file layout OpenFunGraph expects.

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

    images = predictions["images"]        # (N, H, W, 3) float [0,1] or uint8
    depth_maps = predictions["depth"]     # (N, H, W) float meters
    extrinsics = predictions["extrinsic"] # (N, 4, 4) camera-to-world
    intrinsics = predictions["intrinsic"] # (N, 3, 3)

    n_frames = images.shape[0]

    for i in range(n_frames):
        # Save RGB
        img = images[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        # Convert RGB to BGR for cv2
        cv2.imwrite(os.path.join(color_dir, f"{i:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Save depth as uint16 PNG (millimeters)
        d = depth_maps[i]
        d_mm = (d * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{i:06d}.png"), d_mm)

    # Save poses (camera-to-world, one 4x4 per frame, flattened 16 values per line)
    with open(os.path.join(workdir, "poses.txt"), "w") as f:
        for i in range(n_frames):
            vals = extrinsics[i].flatten().tolist()
            f.write(" ".join(f"{v:.8f}" for v in vals) + "\n")

    # Save intrinsics from first frame (assume constant)
    K = intrinsics[0]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    with open(os.path.join(workdir, "intrinsics.txt"), "w") as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")

    h, w = images.shape[1], images.shape[2]
    print(f"Bridged {n_frames} frames ({w}x{h}) to {workdir}")
    return n_frames, h, w


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_vggt(video_path: str, target_fps: int) -> dict:
    """Stage 1: VGGT inference — extract frames, run model, return predictions."""
    import os
    import sys
    import cv2
    import torch
    import numpy as np

    sys.path.insert(0, "/opt/vggt")
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("[VGGT] Loading model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Extract frames
    vs = cv2.VideoCapture(video_path)
    source_fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / target_fps))

    frames_dir = os.path.join(os.path.dirname(video_path), "vggt_frames")
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
    print(f"[VGGT] Extracted {len(image_paths)} frames (interval={frame_interval})")

    # Inference
    images = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    if "images" not in predictions:
        predictions["images"] = images

    # Convert to numpy
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            arr = predictions[key].cpu().numpy()
            if arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            predictions[key] = arr

    # Free GPU memory for next stages
    del model
    torch.cuda.empty_cache()

    print("[VGGT] Inference complete")
    return predictions


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

    for i in range(n_frames):
        img_path = os.path.join(color_dir, f"{i:06d}.png")
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # RAM tagging
        from torchvision import transforms
        ram_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ram_input = ram_transform(image_rgb).unsqueeze(0).to("cuda")
        with torch.no_grad():
            tags, _ = inference_ram.inference(ram_input, ram_model)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        text_prompt = ". ".join(tag_list) + "." if tag_list else "object."

        # GroundingDINO detection
        from groundingdino.util.inference import predict as gdino_predict
        import groundingdino.datasets.transforms as T

        gd_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        from PIL import Image as PILImage
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
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h
        boxes_np = boxes_xyxy.cpu().numpy()

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
            if mask.sum() < 10:
                continue

            # Backproject masked pixels to 3D
            ys, xs = np.where(mask)
            zs = depth_m[ys, xs]
            valid = zs > 0.01
            if valid.sum() < 10:
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

            # Try to match to existing object
            best_score = -1
            best_idx = -1
            for k, obj in enumerate(objects_3d):
                # Spatial similarity: overlap of bounding boxes
                obj_min = obj["pcd_np"].min(axis=0)
                obj_max = obj["pcd_np"].max(axis=0)
                new_min = pts_world.min(axis=0)
                new_max = pts_world.max(axis=0)

                inter_min = np.maximum(obj_min, new_min)
                inter_max = np.minimum(obj_max, new_max)
                inter_vol = np.prod(np.maximum(inter_max - inter_min, 0))
                obj_vol = np.prod(np.maximum(obj_max - obj_min, 1e-8))
                new_vol = np.prod(np.maximum(new_max - new_min, 1e-8))
                union_vol = obj_vol + new_vol - inter_vol
                iou = inter_vol / max(union_vol, 1e-8)

                # Visual similarity: CLIP cosine distance
                cos_sim = np.dot(clip_feat, obj["clip_ft"]) / (
                    np.linalg.norm(clip_feat) * np.linalg.norm(obj["clip_ft"]) + 1e-8
                )

                score = (1 + phys_bias) * iou + (1 - phys_bias) * cos_sim

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

        if (i + 1) % 5 == 0:
            print(f"[3D Fusion] Frame {i+1}/{n_frames}: {len(objects_3d)} objects so far")

    # Post-process: filter small objects, downsample
    filtered = []
    for obj in objects_3d:
        if obj["n_detections"] < 3:
            continue
        pts = obj["pcd_np"]
        cols = obj["pcd_color_np"]
        # Simple voxel downsampling at 1cm
        if len(pts) > 5000:
            voxel_size = 0.01
            quantized = np.floor(pts / voxel_size).astype(np.int64)
            _, unique_idx = np.unique(quantized, axis=0, return_index=True)
            pts = pts[unique_idx]
            cols = cols[unique_idx]
        obj["pcd_np"] = pts
        obj["pcd_color_np"] = cols
        filtered.append(obj)

    print(f"[3D Fusion] Final: {len(filtered)} objects (from {len(objects_3d)} pre-filter)")
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

    # Process every other frame for speed
    for i in range(0, n_frames, 2):
        img_path = os.path.join(color_dir, f"{i:06d}.png")
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        from groundingdino.util.inference import predict as gdino_predict
        import groundingdino.datasets.transforms as T
        from PIL import Image as PILImage

        gd_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
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
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h
        boxes_np = boxes_xyxy.cpu().numpy()

        # Filter: parts should be small (< 10% of image area)
        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        img_area = img_h * img_w
        keep = areas < 0.1 * img_area
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

            # Try to match to existing part
            best_score = -1
            best_idx = -1
            for k, part in enumerate(parts_3d):
                obj_min = part["pcd_np"].min(axis=0)
                obj_max = part["pcd_np"].max(axis=0)
                new_min = pts_world.min(axis=0)
                new_max = pts_world.max(axis=0)
                inter_min = np.maximum(obj_min, new_min)
                inter_max = np.minimum(obj_max, new_max)
                inter_vol = np.prod(np.maximum(inter_max - inter_min, 0))
                obj_vol = np.prod(np.maximum(obj_max - obj_min, 1e-8))
                new_vol = np.prod(np.maximum(new_max - new_min, 1e-8))
                union_vol = obj_vol + new_vol - inter_vol
                iou = inter_vol / max(union_vol, 1e-8)
                cos_sim = np.dot(clip_feat, part["clip_ft"]) / (
                    np.linalg.norm(clip_feat) * np.linalg.norm(part["clip_ft"]) + 1e-8
                )
                score = (1 + phys_bias) * iou + (1 - phys_bias) * cos_sim
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

    # Filter and downsample
    filtered = []
    for part in parts_3d:
        if part["n_detections"] < 2:
            continue
        pts = part["pcd_np"]
        cols = part["pcd_color_np"]
        if len(pts) > 2000:
            voxel_size = 0.005
            quantized = np.floor(pts / voxel_size).astype(np.int64)
            _, unique_idx = np.unique(quantized, axis=0, return_index=True)
            pts = pts[unique_idx]
            cols = cols[unique_idx]
        part["pcd_np"] = pts
        part["pcd_color_np"] = cols
        filtered.append(part)

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
    import pickle


@app.function(
    image=ofg_image,
    gpu="A100-80GB",
    timeout=2400,
    memory=65536,
    secrets=[modal.Secret.from_name("openai-secret")],
)
def run_pipeline(
    video_bytes: bytes,
    target_fps: int = 1,
    conf_thres: float = 50.0,
    openai_model: str = "gpt-4o",
) -> bytes:
    """
    Full pipeline: Video → VGGT → OpenFunGraph → Scene Graph.

    Returns pickled dict with objects, parts, edges.
    """
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input video
        video_path = os.path.join(tmpdir, "input.mov")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Stage 1: VGGT
        print("=" * 60)
        print("STAGE 1: VGGT Inference")
        print("=" * 60)
        predictions = _run_vggt(video_path, target_fps)

        # Bridge VGGT → OpenFunGraph file format
        ofg_dir = os.path.join(tmpdir, "ofg_input")
        os.makedirs(ofg_dir, exist_ok=True)
        n_frames, h, w = _bridge_vggt_to_ofg(predictions, ofg_dir)

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

        # Package results
        result = {
            "objects": objects_3d,
            "parts": parts_3d,
            "edges": edges,
            "n_frames": n_frames,
        }

        print("=" * 60)
        print(f"DONE: {len(objects_3d)} objects, {len(parts_3d)} parts, {len(edges)} edges")
        print("=" * 60)

        return pickle.dumps(result)


@app.local_entrypoint()
def main(
    video_path: str,
    fps: int = 1,
    conf: float = 50.0,
    openai_model: str = "gpt-4o",
):
    """Run OpenFunGraph pipeline on a local video. Saves scene graph as .pkl."""
    import pathlib

    p = pathlib.Path(video_path).expanduser().resolve()
    print(f"Reading {p.name} ({p.stat().st_size / 1024:.0f} KB)")
    print(f"Settings: fps={fps}, conf={conf}, model={openai_model}")

    result_bytes = run_pipeline.remote(
        p.read_bytes(),
        target_fps=fps,
        conf_thres=conf,
        openai_model=openai_model,
    )

    out = p.with_suffix(".pkl")
    out.write_bytes(result_bytes)

    import pickle
    result = pickle.loads(result_bytes)
    print(f"Wrote {out} ({len(result_bytes) / 1024:.1f} KB)")
    print(f"Scene graph: {len(result['objects'])} objects, {len(result['parts'])} parts, {len(result['edges'])} edges")
