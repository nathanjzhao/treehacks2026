# TreeHacks 2026

AR scene understanding pipeline: reconstruct 3D scenes from video, localize a live camera within them, and identify/segment objects with language queries.

## System Architecture

<video src="https://github.com/nathanjzhao/treehacks2026/raw/main/assets/flowchart_dataflow.mp4" autoplay loop muted playsinline width="100%"></video>

[Interactive version](backendviewer/flowchart.html) — click nodes to see video demos of each pipeline stage.

**Modes:** EXPLORE (hover/click to inspect connections) · DATA FLOW (animated walkthrough of each pipeline)

| Key | Action |
|-----|--------|
| `1` / `2` | Switch EXPLORE / DATA FLOW mode |
| `←` / `→` | Skip between data flow paths |
| `A` / `S` / `D` | Switch LLM to Claude / GPT-4o / Gemini |

## Camera Localization

Determine where a camera is in a previously reconstructed 3D scene.

| Module | What it does |
|--------|-------------|
| **[hloc_localization/](hloc_localization/)** | Visual localization via SuperPoint + LightGlue + PnP. Builds an SfM reference map from video (Modal GPU), then localizes new frames against it in 6DoF. ~10-20fps. |

## Object Localization

Find and segment objects in a frame, then estimate their 3D position.

| Module | What it does |
|--------|-------------|
| **[groundedsam2/](groundedsam2/)** | Image + language query → object segmentation. Grounded SAM 2 for open-vocabulary detection and mask generation. |
| **[depthanything/](depthanything/)** | Monocular depth estimation per frame. Used to get depth of objects segmented by SAM — use median/trimmed mean since mask edges bleed into background depth. |

## 3D Reconstruction

Build 3D scene representations from video.

| Module | What it does |
|--------|-------------|
| **[openfungraph/](openfungraph/)** | Video → 3D scene graph. End-to-end pipeline on Modal (A100). See [openfungraph/README.md](openfungraph/README.md). |
| **[mapanything/](mapanything/)** | Standalone MapAnything viewer/runner. Facebook's Map Anything — video → dense depth, camera poses, point clouds. |

## Scratch / Experimental

Models explored but not in the main pipeline.

| Module | What it does |
|--------|-------------|
| **[vggt/](vggt/)** | Facebook's VGGT — video → point cloud + camera poses. Similar to MapAnything but different approach. |
| **[sam3/](sam3/)** | Facebook's SAM 3 — image + language → segmentation. Similar to Grounded SAM 2. |

## Other

| Module | What it does |
|--------|-------------|
| **[explorer/](explorer/)** | 3D point cloud viewer (viser-based). |
| **[mira-chat/](mira-chat/)** | Chat interface. |
| **[android/](android/)** | Android client app. |
| **[vic-backend/](vic-backend/)** | Backend services. |
| **[data/](data/)** | Reference videos (`.MOV`), scene graph outputs (`.json`), and built reference maps. |