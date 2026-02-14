"""
VGGT on Modal — Visual Geometry Grounded Transformer (Meta, CVPR 2025).

Runs the Gradio demo on an A100 with model weights baked into the image.

Deploy:  modal deploy vggt/app.py
Dev:     modal serve vggt/app.py
Video:   modal run vggt/app.py --video-path ~/Desktop/IMG_4706.MOV
"""

import pathlib
import modal

app = modal.App("vggt")

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

_local_dir = pathlib.Path(__file__).parent

vggt_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "Pillow",
        "huggingface_hub",
        "einops",
        "safetensors",
        "opencv-python",
        "gradio==5.17.1",
        "viser==0.2.23",
        "tqdm",
        "hydra-core",
        "omegaconf",
        "scipy",
        "onnxruntime",
        "requests",
        "trimesh",
        "matplotlib",
        "pydantic==2.10.6",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/vggt.git /opt/vggt",
        "cd /opt/vggt && pip install -e .",
    )
    .env({
        "HF_HOME": "/opt/hf_cache",
        "TORCH_HOME": "/opt/torch_cache",
        "GRADIO_SERVER_NAME": "0.0.0.0",
        "GRADIO_SERVER_PORT": "7860",
    })
    # Pre-download model via torch.hub (what demo_gradio.py uses)
    .run_commands(
        "python -c \""
        "import torch; "
        "torch.hub.load_state_dict_from_url("
        "'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt', "
        "map_location='cpu'); "
        "print('torch.hub download done')\"",
    )
    # Also via huggingface_hub (what VGGT.from_pretrained uses)
    .run_commands(
        "python -c \""
        "from vggt.models.vggt import VGGT; "
        "VGGT.from_pretrained('facebook/VGGT-1B'); "
        "print('HF download done')\"",
        gpu="any",
    )
    # Mount wrapper script that patches Gradio for HTTPS proxy
    .add_local_file(str(_local_dir / "run_demo.py"), "/opt/vggt/run_demo.py")
)

with vggt_image.imports():
    import os
    import subprocess
    import sys


@app.function(
    image=vggt_image,
    gpu="A100",
    timeout=1800,
    memory=32768,
)
@modal.web_server(port=7860, startup_timeout=180)
def gradio_demo():
    """Launch VGGT Gradio demo with HTTPS-aware wrapper."""
    subprocess.Popen(
        [sys.executable, "run_demo.py"],
        cwd="/opt/vggt",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


@app.function(
    image=vggt_image,
    gpu="A100",
    timeout=600,
    memory=32768,
)
def predict_video(
    video_bytes: bytes,
    target_fps: int = 1,
    conf_thres: float = 50.0,
    use_depth: bool = True,
) -> dict:
    """
    Run VGGT on a video. Extracts frames, runs inference, returns GLB.

    Args:
        video_bytes: Raw video file content.
        target_fps: Frames to extract per second (default: 1).
        conf_thres: Confidence percentile threshold — lower = more points (default: 50).
        use_depth: Use depth branch (denser) vs pointmap (default: True).
    """
    import tempfile

    import cv2
    import torch

    sys.path.insert(0, "/opt/vggt")
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from visual_util import predictions_to_glb

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("Loading model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save video to temp file
        video_path = os.path.join(tmpdir, "input.mov")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Extract frames at target_fps
        images_dir = os.path.join(tmpdir, "images")
        os.makedirs(images_dir)

        vs = cv2.VideoCapture(video_path)
        source_fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(source_fps / target_fps))

        image_paths = []
        count = 0
        frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                path = os.path.join(images_dir, f"{frame_num:06d}.png")
                cv2.imwrite(path, frame)
                image_paths.append(path)
                frame_num += 1
        vs.release()
        print(f"Extracted {len(image_paths)} frames from {count} total (interval={frame_interval})")

        # Run inference
        images = load_and_preprocess_images(image_paths).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        # Post-process: pose → extrinsic/intrinsic (on tensors, before numpy)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Ensure images are in predictions for GLB color extraction
        if "images" not in predictions:
            predictions["images"] = images

        # Convert all tensors to numpy, squeeze batch dim (only if dim 0 == 1)
        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                arr = predictions[key].cpu().numpy()
                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                predictions[key] = arr

        # Generate world points from depth
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )
        predictions["world_points_from_depth"] = world_points

        # Generate GLB scene
        mode = "Depthmap and Camera Branch" if use_depth else "Predicted Pointmap"
        print(f"Generating 3D scene (mode={mode}, conf_thres={conf_thres})...")
        scene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            show_cam=True,
            target_dir=tmpdir,
            prediction_mode=mode,
        )

        glb_path = os.path.join(tmpdir, "output.glb")
        scene.export(file_obj=glb_path)

        with open(glb_path, "rb") as f:
            glb_bytes = f.read()

    print(f"GLB size: {len(glb_bytes) / 1024 / 1024:.1f} MB")
    return {
        "glb": glb_bytes,
        "num_frames": len(image_paths),
        "source_fps": source_fps,
    }


@app.local_entrypoint()
def main(
    video_path: str,
    fps: int = 2,
    conf: float = 25.0,
    depth: bool = True,
):
    """Run VGGT on a local video file. Saves .glb next to the input."""
    p = pathlib.Path(video_path).expanduser().resolve()
    print(f"Reading {p.name} ({p.stat().st_size / 1024:.0f} KB)")
    print(f"Settings: fps={fps}, conf_thres={conf}, depth={depth}")

    result = predict_video.remote(
        p.read_bytes(),
        target_fps=fps,
        conf_thres=conf,
        use_depth=depth,
    )

    out = p.with_suffix(".glb")
    out.write_bytes(result["glb"])
    print(f"Wrote {out} ({len(result['glb']) / 1024 / 1024:.1f} MB)")
    print(f"Processed {result['num_frames']} frames from {result['source_fps']:.1f} fps video")
