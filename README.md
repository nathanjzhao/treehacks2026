# Mira — TreeHacks 2026

**AI-powered assisted living platform** that combines 3D scene understanding with a conversational AI assistant to help elderly residents find lost objects, answer medical questions, and escalate emergencies to caregivers.

A resident says "Where are my glasses?" → the system reconstructs their room in 3D, locates the object via multi-view Gemini 3 Flash vision + raycasting, and guides them to it — while a supervisor dashboard tracks everything in real-time.

### Core Pipeline

1. **3D Reconstruction** — Video → dense point cloud + camera poses (MapAnything / VGGT on Modal A100s)
2. **Object Finding** — Natural language query → Gemini 3 Flash multi-view detection → raycast into point cloud → 3D coordinates
3. **Camera Localization** — Live camera frame → 6DoF pose in the reconstructed scene (SuperPoint + LightGlue + PnP)
4. **Conversational AI** — Voice/text chat (GPT-5-mini via OpenRouter) with function calling for object search, medical Q&A, medication lookup, and caregiver escalation
5. **Real-time Dashboard** — Supabase Realtime event spine powering a supervisor view of all patients, alerts, and activity

**Tech Stack:** Next.js 15 · React 19 · Tailwind CSS · Supabase (Postgres + Realtime) · OpenRouter (GPT-5-mini, Gemini 3 Flash) · Whisper STT · OpenAI TTS · Twilio SMS · Modal (A100 GPU) · Viser · ONNX Runtime Web · Python · FastAPI

## System Architecture

<video src="https://github.com/user-attachments/assets/5a214eb2-ba6b-4e09-b9b9-fe8bcc5b3d0d" autoplay loop muted playsinline width="100%"></video>

[Interactive version](backendviewer/flowchart.html) — click nodes to see video demos of each pipeline stage.

**Modes:** EXPLORE (hover/click to inspect connections) · DATA FLOW (animated walkthrough of each pipeline)

| Key | Action |
|-----|--------|
| `1` / `2` | Switch EXPLORE / DATA FLOW mode |
| `←` / `→` | Skip between data flow paths |

## Camera Localization

Determine where a camera is in a previously reconstructed 3D scene.

| Module | What it does |
|--------|-------------|
| **[hloc_localization/](hloc_localization/)** | Visual localization via SuperPoint + LightGlue + PnP. Builds an SfM reference map from video (Modal GPU), then localizes new frames against it in 6DoF. ~10-20fps. |

## Object Localization

Find and segment objects in a frame, then estimate their 3D position.

| Module | What it does |
|--------|-------------|
| **[segmentation/](segmentation/)** | Image + language query → object segmentation. Open-vocabulary detection and mask generation. |
| **[depthanything/](depthanything/)** | Monocular depth estimation per frame. Used to get depth of objects segmented by SAM — use median/trimmed mean since mask edges bleed into background depth. |

## 3D Reconstruction

Build 3D scene representations from video.

| Module | What it does |
|--------|-------------|
| **[scenegraph/](scenegraph/)** | Video → 3D scene graph. End-to-end pipeline on Modal (A100). See [scenegraph/README.md](scenegraph/README.md). |
| **[reconstruction/](reconstruction/)** | Dense 3D reconstruction — video → dense depth, camera poses, point clouds. |

## Scratch / Experimental

Models explored but not in the main pipeline.

| Module | What it does |
|--------|-------------|
| **[vggt/](vggt/)** | Facebook's VGGT — video → point cloud + camera poses. Similar to MapAnything but different approach. |
| **[sam3/](sam3/)** | Facebook's SAM 3 — image + language → segmentation. Similar to Grounded SAM 2. |

## Other

| Module | What it does |
|--------|-------------|
| **[explorer/](explorer/)** | Interactive 3D point cloud viewer (Viser + FastAPI). Natural language query → Gemini 3 Flash multi-view detection → raycast into point cloud → animated camera fly-to. Auto-detects scene orientation on load. |
| **[mira-chat/](mira-chat/)** | Full-stack Next.js app — resident voice/text chat (GPT-5-mini + function calling), supervisor dashboard with real-time event timeline, object finding integration, caregiver escalation via Twilio SMS. See [mira-chat/README.md](mira-chat/README.md). |
| **[securitycam/](securitycam/)** | Security camera streaming — mediamtx config for RTMP/RTSP/HLS/WebRTC ingestion from IP cameras or phones. |
| **[android/](android/)** | Android client app with AR overlay. |
| **[vic-backend/](vic-backend/)** | Backend services. |
| **[backendviewer/](backendviewer/)** | Interactive architecture flowchart with animated data flow visualization and video demos. |
| **[data/](data/)** | Reference videos (`.MOV`), scene graph outputs (`.json`), and built reference maps. |