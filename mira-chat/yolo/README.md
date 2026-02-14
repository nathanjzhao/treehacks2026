# YOLO Real-Time Object Detection — `/yolo`

Browser-side YOLOv8n object detection that runs on the `/stream` AR HUD page. Draws HUD-style bounding boxes over a live webcam feed. Built for the Mira eldercare assistant (TreeHacks 2026).

## How It Works

```
Webcam video (getUserMedia)
       │
       ▼
  useDetection hook (yolo/useDetection.ts)
       │
       ├─ Captures video frame → offscreen canvas (640×640)
       ├─ Converts RGBA → NCHW float32 [1, 3, 640, 640] normalized [0,1]
       ├─ Runs ONNX inference (WebGL preferred, WASM fallback)
       ├─ Parses YOLOv8 output tensor [1, 84, 8400]
       │    └─ 8400 candidate boxes × (4 bbox coords + 80 class scores)
       ├─ Filters by confidence threshold (default 0.35)
       ├─ Applies Non-Maximum Suppression (IoU threshold 0.45)
       └─ Returns Detection[] with label, confidence, x, y, width, height
              │
              ▼
  DetectionOverlay (yolo/DetectionOverlay.tsx)
       │
       ├─ Maps Detection[] bbox coords from video-space to screen-space
       ├─ Draws on a <canvas> overlay positioned over the video
       ├─ HUD-style rendering: glow outlines, thick corner accents, label pills
       ├─ Color-coded by category (people=teal, furniture=purple, etc.)
       └─ Shows FPS/inference badge in top-right corner
```

## File Structure

```
yolo/
├── types.ts              # Detection interface, 80 COCO class labels, color map
├── useDetection.ts       # React hook: loads ONNX model, runs inference loop
├── DetectionOverlay.tsx   # Canvas component: draws HUD bounding boxes
├── index.ts              # Barrel exports
└── README.md             # This file
```

## Key Implementation Details

### Model Loading (`useDetection.ts`)
- Dynamically imports `onnxruntime-web` (avoids SSR issues)
- Tries `webgl` execution provider first (GPU-accelerated, ~2-4x faster)
- Falls back to `wasm` if WebGL is unavailable
- Model loaded from `/models/yolov8n.onnx` (~12MB, gitignored)

### Inference Loop (`useDetection.ts`)
- Uses `requestAnimationFrame` throttled to ~8 FPS (configurable via `targetFps`)
- Frame capture: `ctx.drawImage(video, ...)` → `getImageData()` → Float32Array
- Pixel format conversion: RGBA (canvas) → NCHW (model expects channels-first)
- Output parsing: YOLOv8 outputs a transposed [1, 84, 8400] tensor. For each of 8400 anchor boxes, the first 4 values are (cx, cy, w, h) in 640×640 space and the remaining 80 are per-class confidence scores
- Bbox coords are scaled from 640×640 model space back to actual video dimensions
- NMS removes overlapping boxes (greedy, sorted by confidence, IoU > 0.45 = suppressed)

### Overlay Rendering (`DetectionOverlay.tsx`)
- Canvas sized to container with `devicePixelRatio` scaling for crisp rendering
- Coordinates mapped from video-space (e.g. 1920×1080) to display-space (canvas CSS size)
- Each detection drawn with: outline rect (2px + shadowBlur glow), corner accents (3px thick, 16px long), label pill (dark bg, colored border, colored text)
- Category colors defined in `types.ts` → `LABEL_COLORS` map

### Integration in `/stream` page
- `useDetection(videoRef, { enabled: yoloEnabled && videoReady })` — hook consumes the same `<video>` element used for the webcam background
- `<DetectionOverlay>` is placed inside the background layer, above the video but below HUD UI elements
- Toggle button in bottom-left HUD corner enables/disables detection
- Status line shows: model loaded state, detection count, inference time

## Setup

### 1. Get the ONNX model

Export via Python (recommended — guaranteed correct format):
```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
mv yolov8n.onnx public/models/
```

Or download directly:
```bash
mkdir -p public/models
# Check https://github.com/ultralytics/assets/releases for latest
curl -L -o public/models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx
```

### 2. Install dependency
```bash
npm install onnxruntime-web
```

### 3. Webpack config
`next.config.ts` needs WASM asset handling:
```ts
webpack: (config) => {
  config.module.rules.push({ test: /\.wasm$/, type: "asset/resource" });
  return config;
},
serverExternalPackages: ["onnxruntime-web"],
```

## Detected Objects (80 COCO classes)

The model detects these categories:

| Category | Classes |
|----------|---------|
| People | person |
| Vehicles | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| Animals | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| Kitchen | bottle, wine glass, cup, fork, knife, spoon, bowl |
| Food | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| Furniture | chair, couch, potted plant, bed, dining table, toilet |
| Electronics | tv, laptop, mouse, remote, keyboard, cell phone |
| Appliances | microwave, oven, toaster, sink, refrigerator |
| Other | book, clock, vase, scissors, teddy bear, hair drier, toothbrush, traffic light, fire hydrant, stop sign, parking meter, bench |
| Accessories | backpack, umbrella, handbag, tie, suitcase |
| Sports | frisbee, skis, snowboard, sports ball, kite, baseball bat/glove, skateboard, surfboard, tennis racket |

## Swapping to a Server-Side Backend

The architecture is designed so only `useDetection.ts` needs to change. The overlay, types, and colors all remain the same.

### Option A: REST API backend
1. Create `app/api/yolo/detect/route.ts`
   - Accept a JPEG frame via POST (base64 or multipart)
   - Run inference server-side (Ultralytics Python, NVIDIA Triton, etc.)
   - Return `Detection[]` JSON
2. In `useDetection.ts`, replace the ONNX inference with a `fetch()` to your endpoint
3. Everything else (overlay, colors, types, integration) stays identical

### Option B: WebSocket stream
1. Server runs YOLOv8 on frames from a camera feed (e.g. Ray-Ban JPEG stream)
2. Pushes `Detection[]` to client via WebSocket
3. `useDetection.ts` subscribes to the socket instead of running local inference

### Backend comparison

| Backend | Latency | Setup | Notes |
|---------|---------|-------|-------|
| **ONNX Runtime Web (current)** | 50-300ms | npm install only | Runs in browser, no server needed |
| Ultralytics Python | 20-50ms | `pip install ultralytics` | Local GPU, fastest for dev |
| NVIDIA NIM / Triton | 10-30ms | API key / Docker | Cloud GPU, production-grade |
| Modal | 30-100ms | `pip install modal` | Serverless GPU, scales to zero |
| Roboflow | 50-150ms | API key | Hosted, easiest cloud setup |

## Tuning Parameters

In `useDetection.ts`:
- `CONFIDENCE_THRESHOLD` (default 0.35) — lower = more detections, more false positives
- `IOU_THRESHOLD` (default 0.45) — lower = more aggressive NMS, fewer overlapping boxes
- `TARGET_FPS` (default 8) — higher = smoother but more CPU/GPU load
- `INPUT_SIZE` (640) — YOLOv8n expects 640×640, don't change unless using a different model

These can also be passed via the hook options:
```tsx
const { detections } = useDetection(videoRef, {
  enabled: true,
  confidenceThreshold: 0.25,  // more sensitive
  targetFps: 12,              // smoother updates
});
```
