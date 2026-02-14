# YOLO Object Detection — `/yolo`

Real-time object detection overlay for the AR HUD stream page.

## Setup

1. Download YOLOv8n ONNX model (~6MB):
   ```bash
   mkdir -p public/models
   curl -L -o public/models/yolov8n.onnx \
     https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx
   ```

2. The detection runs automatically on `/stream` when the model file is present.

## Architecture

```
yolo/
├── types.ts              # Detection types, COCO labels, color map
├── useDetection.ts       # React hook — video → ONNX inference → detections
├── DetectionOverlay.tsx   # Canvas overlay — draws HUD-style bounding boxes
├── index.ts              # Barrel exports
└── README.md
```

**Current backend:** ONNX Runtime Web (browser-side, YOLOv8n, ~6MB model).

## Swapping Backends

To switch from browser ONNX to a server-side backend:

1. Create `app/api/yolo/detect/route.ts`:
   - Accept a JPEG frame via POST (multipart or base64)
   - Run inference (Ultralytics, NVIDIA NIM, Modal, GCP Vertex, Roboflow, etc.)
   - Return `Detection[]` JSON

2. Replace `useDetection.ts` internals:
   - Instead of ONNX inference, POST frame to `/api/yolo/detect`
   - Parse response into `Detection[]`
   - Everything else (overlay, colors, types) stays the same

### Backend options

| Backend | Latency | Setup | Notes |
|---------|---------|-------|-------|
| ONNX Runtime Web (current) | 50-200ms | npm install only | Runs in browser, no server |
| Ultralytics Python | 20-50ms | `pip install ultralytics` | Local GPU, fastest |
| NVIDIA NIM / Triton | 10-30ms | API key | Cloud GPU, production-grade |
| Modal | 30-100ms | `pip install modal` | Serverless GPU, scales to zero |
| Roboflow | 50-150ms | API key | Hosted, easiest setup |
| Google Cloud Vision | 100-300ms | GCP project | Enterprise, expensive |

## Detected Objects (COCO 80 classes)

Relevant for eldercare: **person**, cup, bottle, chair, couch, bed, dining table,
cell phone, remote, book, clock, potted plant, scissors, etc.
