# Mira

## Inspiration

Three of us have grandparents with Alzheimer's or dementia. If you've been around it, you know the loop: "Have you seen my pills?" five times in an afternoon. It's not just forgetting. It's the loss of autonomy, the slow erosion of someone's confidence that they can manage their own life.

- Several of us have backgrounds in robotics, simulation infrastructure, manipulation policies, and 3D perception. Across the team we'd been exploring XR, lab automation, and healthcare.
- We considered other directions: smart glasses for blind navigation, lab automation assistants. But we kept coming back to this: we all have moments where we'd kill for a recording of our lives. Where did I put my keys? Did I already take that pill?
- Looking at the tech available right now (real-time 3D reconstruction, visual foundation models, tool-using LLMs, consumer AR glasses), we realized this is the most futuristic thing we could actually build in a weekend.
- Many dementia patients refuse traditional care, insisting family handle everything, which puts enormous strain on people who can't be present 24/7. Mira is designed for this gap: asynchronous monitoring without pulling out a phone, assistance at 3 AM, and a caregiver dashboard that keeps family informed.

For someone with Alzheimer's, a searchable spatial memory isn't a convenience. It's dignity. So we built Mira: an AI companion on smart glasses that understands your home in 3D, and when you say "where are my pills?", actually finds them.

## What It Does

Mira is a voice-first AI assistant for dementia patients that combines 3D scene understanding with medical knowledge and caregiver alerts. It runs on Ray-Ban Meta glasses.

**For the patient:**
- *"Where's my pill bottle?"* Mira searches a 3D reconstruction of the room (built from a short video walkthrough), locates the object, and flies the 3D viewer to it.
- *"What medications do I take and when?"* Pulls from the patient's health record, cross-references clinical guidelines, and returns evidence-graded answers with every claim tagged to its source (PubMed, NIH, clinical databases).
- *"Help, I've fallen."* The caregiver gets an email within seconds with full context: the event, actions leading up to it, and the patient's recent activity timeline.

**For the caregiver:**
- Real-time dashboard showing health summaries, a live event timeline, and emergency alerts. Updates are instant with zero polling, so the caregiver always sees the latest state without refreshing.
- Email escalation for emergencies so they don't need to be actively watching the dashboard. When something urgent happens, they get full context in their inbox: what the patient said, what Mira did, and the recent activity leading up to it.
- Every interaction logged as an immutable event for audit. Questions asked, tools called, alerts triggered, all timestamped and searchable.

No menus, no apps, no screens to read. The patient just talks.

## How We Built It

- **3D Scene.** A video walkthrough gets reconstructed into a dense 3D point cloud via structure-from-motion on Modal A100 GPUs. Grounding DINO detects objects per frame, and each detection gets backprojected into 3D using depth maps and camera poses. Detections of the same object across different views are merged by geometric overlap and CLIP similarity, producing a 3D Scene Object Graph mapping object labels to real-world coordinates.
- **Object Finding.** We render the scene from multiple camera angles (farthest-point sampling for coverage), Gemini 2.0 Flash returns 2D bounding boxes, and we project those back into the point cloud to triangulate real-world coordinates. The camera flies to the object. Voice command to 3D localization in under ten seconds.
- **Live Pose Tracking.** The glasses' video stream feeds DPVO for continuous 6DoF pose tracking. DPVO is fast but drifts over time, so after a few minutes your estimated position can be meters off. HLoc re-localization periodically anchors the pose against the pre-built 3D map, correcting drift.
- **Voice Pipeline.** Ray-Ban Meta SDK only exposes camera and mic to Android, so glasses connect to an Android phone first. Bluetooth audio to phone, JPEG frames over hotspot to laptop, RTMP to media server, Tailscale for tunneling. An absurd number of hops for "listen to a sentence." It works.
- **Agent.** LLM agent (OpenRouter) with function-calling tools: spatial navigation (3D scene graph + live pose), patient info (OpenEvidence/FHIR), medication lookup (Perplexity Sonar with evidence grading). Whisper for STT, OpenAI TTS for response.
- **Escalation.** Emergencies trigger an email to the caregiver with full context. Works alongside the real-time dashboard.
- **Real-time.** Supabase Postgres with Realtime subscriptions. Every interaction is an immutable event. Dashboard updates via postgres_changes, zero polling.
- **Privacy.** Records are based off of FHIR R4 and are de-identified before reaching any LLM. Age ranges and condition names only, never raw identifiers.

## Challenges We Ran Into

- **3D Orientation.** Point clouds don't know which way is up. A room can render upside-down and the geometry is equally valid. We built a Gemini-based detector that renders three candidate rotations and asks which looks like an actual room. Inelegant. Works.
- **Stanford WiFi.** Stanford's network blocks a lot of traffic. Our streaming pipeline kept dying because the university WiFi was dropping connections. Tailscale mesh VPN was the only thing that reliably punched through.
- **Browser-side ML.** Running YOLOv8n via ONNX Runtime Web for the patient experience required WASM webpack config and render loop optimization. Smooth real-time detection overlays in a browser are harder than they sound.
- **Modal at Hackathon Speed.** Multiple heavy models (Grounding DINO, DPVO, depth estimation, 3D scene reconstruction) on Modal A100 GPUs. Every new model or dependency change meant rebuilding containers with complex dependency trees. A lot of pure engineering to get latency down and orchestrate everything under time pressure.

## Accomplishments We're Proud Of

- End-to-end voice to 3D object localization in under ten seconds.
- Complete 3D scene reconstruction from casual phone video on Modal GPUs. The point clouds came out detailed enough to identify individual objects on shelves. Seeing the first reconstruction was one of those moments where the whole team gathered around a screen.
- Real-time event propagation to the caregiver dashboard plus email escalation. Zero polling, instant updates, every interaction logged.
- Evidence-graded medical citations where every answer is tagged with source and confidence level. Most AI assistants just give you an answer. Mira tells you why you should or shouldn't trust it.
- Patient records de-identified before reaching any LLM. The model never sees a name, date of birth, or identifying information.

## What We Learned

- **Don't ask a foundation model to do geometry.** We tried getting Gemini to predict 3D coordinates directly. Models hallucinate spatial information. Let the model do perception in 2D, handle the 3D math yourself with projection matrices and known geometry.
- **Composing specialist models beats monoliths.** Our pipeline chains Whisper, Gemini, Grounding DINO, depth estimation, visual odometry, and more. Same argument behind Grounded SAM: composition is more flexible than one unified model. Add a capability, plug in a specialist. Nothing retrains.
- **Voice-first design changes everything.** We prototyped with a touchscreen. Then we thought about using it with low vision, tremor, and cognitive impairment. Scrolling, tapping a 44px button, parsing a settings menu, all become walls. When your user is 80, the only interface that works is speech.
- **Consumer AR hardware isn't there yet.** Ray-Ban Metas have no IMU data exposed to developers, no depth sensor, SDK is Android-only. Meta seems to be deliberately restricting developer access because they want to ship their own products on these features first. No display control, no raw sensor data. We hit the ceiling quickly.

## What's Next for Mira

- **Continuous scene updates.** The 3D model is currently from a one-time walkthrough. Objects move. The system should update incrementally as the resident goes about their day.
- **Predictive object tracking.** If the event log shows Mrs. Chen leaves her glasses in the fridge every Tuesday, the system should learn that and check there first.
- **Dedicated sensor hardware for fall detection.** Ray-Bans don't expose IMU data, so fall detection needs external hardware. A wearable with accelerometer and gyroscope, combined with 3D scene context, could distinguish a real fall from sitting down.
- **Remote monitoring and health device integration.** Cameras in common areas, Apple Watch vitals, other health peripherals feeding into the same event stream.
- **Scaling point clouds.** One room works. A full facility with hallways, common areas, and dozens of rooms requires procedural rendering and level-of-detail management for very large point clouds, similar to what Foxglove does for robotics data.