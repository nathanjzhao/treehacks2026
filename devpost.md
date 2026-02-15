## Inspiration

Millions of elderly residents in assisted living facilities lose track of everyday objects — glasses, pill organizers, walkers. For someone with mild cognitive impairment, this means missed medications, anxiety, and preventable falls. Caregivers spend hours helping residents locate items instead of providing direct care.

We asked: *what if the building itself could remember where everything is?* With advances in 3D scene reconstruction, a simple video walkthrough can become a dense, searchable point cloud. Combine that with a voice-first AI assistant, and a resident can just say "Where are my glasses?" through their Meta Ray-Ban glasses and get an answer.

## What it does

**Mira** combines 3D scene understanding with voice interaction and real-time caregiver monitoring.

**For residents:** Ask questions by voice through Ray-Ban Gen 3 glasses or a web UI. "Where are my pills?" triggers a search through a 3D reconstruction of their room. "What meds do I take?" pulls from their health record. "I need help" escalates instantly via SMS.

**For caregivers:** A real-time dashboard shows health summaries, a live event timeline, emergency alerts, and evidence-graded medical search results with citations from PubMed and Cochrane.

**The core innovation is the 3D object-finding pipeline.** A video walkthrough is reconstructed into a point cloud. When a resident asks for an object, Gemini 3 Flash analyzes rendered views from 6 camera angles, returns 2D bounding boxes, and raycasts them into 3D space to triangulate the real-world location.

## How we built it

- **Frontend/API:** Next.js 16 + React 19. Chat streams GPT-4o-mini's reasoning via SSE with five function-calling tools (find_object, escalate, medication lookup, medical search, clinical guidelines).
- **3D Pipeline (Modal A100 GPUs):** MapAnything (video to point cloud) then Grounded SAM 2 (segmentation) then Depth Anything V2 (monocular depth) then hloc/SuperPoint+LightGlue (camera localization).
- **Object Finding:** Viser 3D explorer renders 6 diverse views via farthest-point sampling. Gemini 3 Flash returns box_2d detections. Raycasting triangulates 3D position. Camera flies to target with SLERP interpolation.
- **Voice:** Whisper (STT) + ElevenLabs (TTS). Android app captures Bluetooth audio from Ray-Bans and streams JPEG frames for reconstruction.
- **Real-time:** Supabase Postgres with Realtime subscriptions. Every interaction is an immutable event — dashboard updates instantly via postgres_changes.
- **Privacy:** Patient data is de-identified before hitting the LLM — age ranges and condition names only, never raw identifiers.

## Challenges we ran into

**3D coordinate mapping.** Gemini's box_3d returns coordinates in its own frame, not the point cloud's world frame. After days of trying to map between them, we pivoted to box_2d with raycasting — having the LLM identify objects in 2D and projecting back into 3D ourselves. Far more reliable.

**Scene orientation.** Point clouds don't have a consistent "up." We built a Gemini-based detector that renders three candidate orientations and asks which looks right-side up.

**Multi-hop video streaming.** Bluetooth audio from Ray-Bans to Android phone to JPEG frames over hotspot to laptop to RTMP to mediamtx. Getting this stable across networks led us to Tailscale for cross-network tunneling.

**Browser-side ML.** Running YOLOv8n via ONNX Runtime Web required careful WASM webpack config and render loop optimization for a smooth AR HUD overlay.

## Accomplishments that we're proud of

- End-to-end object finding in under 10 seconds — voice command to 3D fly-to animation
- Multi-view consensus localization using 6 angles for robust triangulation
- Fully real-time event propagation to the caregiver dashboard — zero polling
- An AR HUD with live YOLO detection, chat overlay, and escalation toasts running entirely in-browser
- Evidence-graded medical citations so caregivers know how much to trust search results
- Privacy by design — the LLM never sees raw patient identifiers

## What we learned

- **Raycasting beats direct 3D prediction.** 2D bounding boxes projected into a known point cloud are far more reliable than asking a model to predict 3D coordinates.
- **Event sourcing is perfect for healthcare** — audit trail, real-time updates, and replay from one pattern.
- **Voice-first design changes everything.** When your user is 80 with cognitive impairment, every UI decision is different.
- **Serverless GPUs (Modal) make heavy CV accessible at hackathon scale** — MapAnything, SAM 2, Depth Anything, and hloc with zero infrastructure.

## What's next for Mira

- Continuous scene updates as objects move, rather than one-time video capture
- Facility-wide multi-room 3D mapping
- Predictive object tracking — learn where residents typically leave items
- Fall detection using Ray-Ban IMU data combined with 3D scene context
- EHR integration for automated medication schedules
- On-device inference for offline capability
