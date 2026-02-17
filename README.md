# Mira — TreeHacks 2026

**An AI companion on smart glasses that understands your home in 3D.** When a dementia patient says "where are my pills?", Mira actually finds them.

Voice-first assisted living platform combining 3D scene understanding, medical knowledge, and real-time caregiver alerts. Runs on Ray-Ban Meta glasses.

**Tech Stack:** Next.js · React · Tailwind CSS · Supabase (Postgres + Realtime) · OpenRouter · Whisper STT · OpenAI TTS · Modal (A100 GPU) · Viser · ONNX Runtime Web · Python · FastAPI

## System Architecture

<video src="https://github.com/user-attachments/assets/a5fb430a-5176-4695-9428-8e3962875725" autoplay loop muted playsinline width="100%"></video>

[Interactive version](backendviewer/flowchart.html) — click nodes to see video demos of each pipeline stage.

**Modes:** EXPLORE (hover/click to inspect connections) · DATA FLOW (animated walkthrough of each pipeline)

| Key | Action |
|-----|--------|
| `1` / `2` | Switch EXPLORE / DATA FLOW mode |
| `←` / `→` | Skip between data flow paths |


## Inspiration

Three of us have grandparents with Alzheimer's or dementia. If you've been around it, you know the loop: "Have you seen my pills?" five times in an afternoon. It's not just forgetting — it's the loss of autonomy, the slow erosion of someone's confidence that they can manage their own life.

Many dementia patients refuse traditional care, insisting family handle everything, which puts enormous strain on people who can't be present 24/7. Mira is designed for this gap: asynchronous monitoring without pulling out a phone, assistance at 3 AM, and a caregiver dashboard that keeps family informed.

For someone with Alzheimer's, a searchable spatial memory isn't a convenience. It's dignity.

## Modules

### 3D Reconstruction

| Module | What it does |
|--------|-------------|
| **[scenegraph/](scenegraph/)** | Video → 3D scene graph. End-to-end pipeline on Modal (A100): structure-from-motion, Grounding DINO object detection per frame, backprojection into 3D via depth maps + camera poses, geometric overlap + CLIP similarity merging into a Scene Object Graph. See [scenegraph/README.md](scenegraph/README.md). |
| **[reconstruction/](reconstruction/)** | Dense 3D reconstruction — video → dense depth, camera poses, point clouds. |

### Camera Localization

| Module | What it does |
|--------|-------------|
| **[hloc_localization/](hloc_localization/)** | Visual localization via SuperPoint + LightGlue + PnP. Builds an SfM reference map from video (Modal GPU), then localizes new frames against it in 6DoF. Live pose tracking uses DPVO for continuous 6DoF with periodic HLoc re-localization to correct drift. |

### Object Localization

| Module | What it does |
|--------|-------------|
| **[segmentation/](segmentation/)** | Image + language query → object segmentation. Open-vocabulary detection and mask generation. |
| **[depthanything/](depthanything/)** | Monocular depth estimation per frame. Used to get depth of segmented objects — median/trimmed mean since mask edges bleed into background. |

### Apps & Services

| Module | What it does |
|--------|-------------|
| **[explorer/](explorer/)** | Interactive 3D point cloud viewer (Viser + FastAPI). Natural language query → Gemini multi-view detection → raycast into point cloud → animated camera fly-to. Auto-detects scene orientation via Gemini rotation voting. |
| **[mira-chat/](mira-chat/)** | Full-stack Next.js app — resident voice/text chat (LLM + function calling for spatial nav, medical Q&A, medication lookup), supervisor dashboard with real-time event timeline, caregiver escalation via email. Supabase Realtime for zero-polling updates. See [mira-chat/README.md](mira-chat/README.md). |
| **[android/](android/)** | Android client app — receives Bluetooth audio + JPEG frames from Ray-Ban Meta glasses, bridges to backend via Tailscale. |
| **[securitycam/](securitycam/)** | Security camera streaming — mediamtx config for RTMP/RTSP/HLS/WebRTC ingestion. |
| **[vic-backend/](vic-backend/)** | Backend services. |
| **[backendviewer/](backendviewer/)** | Interactive architecture flowchart with animated data flow visualization and video demos. |

## Team

| Member | Contributions |
|--------|--------------|
| **Nathan Zhao** | ML pipeline & GPU compute on Modal — object localization, 3D scene reconstruction, camera pose tracking with visual odometry. Backend flowchart viewer, security cam integration, point cloud viewer. |
| **Victor** | Streaming video and audio from the Ray-Ban Meta SDK, linking to frontend, delivering audio feedback back to the user. |
| **Madhuhaas** | Agentic system and tool-calling pipeline — pulling patient info, web search for medication lookup. Frontend development. |
| **Antonio** | Online object localization, animated camera fly-to in the 3D viewer, high-level pitch direction, demo video editing. |

## What's Next

- **Continuous scene updates.** The 3D model is currently from a one-time walkthrough. Objects move. The system should update incrementally as the resident goes about their day.
- **Predictive object tracking.** If the event log shows someone leaves their glasses in the fridge every Tuesday, the system should learn that and check there first.
- **Dedicated sensor hardware.** Ray-Bans don't expose IMU data, so fall detection needs external hardware — a wearable with accelerometer and gyroscope combined with 3D scene context.
- **Health device integration.** Apple Watch vitals, cameras in common areas, other health peripherals feeding into the same event stream.
- **Scaling point clouds.** One room works. A full facility requires procedural rendering and level-of-detail management for very large point clouds.
