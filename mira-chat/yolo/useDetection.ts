"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Detection, COCO_LABELS } from "./types";

const MODEL_URL = "/models/yolov8n.onnx";
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.35;
const IOU_THRESHOLD = 0.45;
const TARGET_FPS = 8; // inference FPS (not render FPS)

interface UseDetectionOptions {
  enabled?: boolean;
  confidenceThreshold?: number;
  targetFps?: number;
}

export function useDetection(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  options: UseDetectionOptions = {}
) {
  const {
    enabled = true,
    confidenceThreshold = CONFIDENCE_THRESHOLD,
    targetFps = TARGET_FPS,
  } = options;

  const [detections, setDetections] = useState<Detection[]>([]);
  const [inferenceMs, setInferenceMs] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);

  const sessionRef = useRef<any>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number>(0);
  const lastInferenceRef = useRef<number>(0);
  const runningRef = useRef(false);

  // Load ONNX model
  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    async function loadModel() {
      try {
        const ort = await import("onnxruntime-web");

        // Try WebGL first (GPU-accelerated), fall back to WASM
        ort.env.wasm.numThreads = 1;

        let session;
        try {
          session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ["webgl"],
          });
        } catch {
          // WebGL not available — fall back to WASM
          session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ["wasm"],
          });
        }

        if (!cancelled) {
          sessionRef.current = session;
          setModelLoaded(true);
        }
      } catch (err) {
        if (!cancelled) {
          setModelError(
            err instanceof Error ? err.message : "Failed to load YOLO model"
          );
        }
      }
    }

    loadModel();
    return () => { cancelled = true; };
  }, [enabled]);

  // Create offscreen canvas for frame capture
  useEffect(() => {
    const c = document.createElement("canvas");
    c.width = INPUT_SIZE;
    c.height = INPUT_SIZE;
    canvasRef.current = c;
  }, []);

  const runInference = useCallback(async () => {
    const video = videoRef.current;
    const session = sessionRef.current;
    const canvas = canvasRef.current;
    if (!video || !session || !canvas || video.readyState < 2) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    const start = performance.now();

    // Capture frame and resize to 640x640
    ctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const pixels = imageData.data; // RGBA

    // Convert to NCHW float32 normalized [0,1]
    const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const pixelCount = INPUT_SIZE * INPUT_SIZE;
    for (let i = 0; i < pixelCount; i++) {
      float32[i] = pixels[i * 4] / 255;                    // R
      float32[pixelCount + i] = pixels[i * 4 + 1] / 255;   // G
      float32[2 * pixelCount + i] = pixels[i * 4 + 2] / 255; // B
    }

    const ort = await import("onnxruntime-web");
    const tensor = new ort.Tensor("float32", float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    const results = await session.run({ images: tensor });
    const output = results[Object.keys(results)[0]].data as Float32Array;

    // YOLOv8 output: [1, 84, 8400] — 4 bbox coords + 80 class scores
    const numDetections = 8400;
    const rawBoxes: Detection[] = [];

    const videoW = video.videoWidth || INPUT_SIZE;
    const videoH = video.videoHeight || INPUT_SIZE;
    const scaleX = videoW / INPUT_SIZE;
    const scaleY = videoH / INPUT_SIZE;

    for (let i = 0; i < numDetections; i++) {
      let maxScore = 0;
      let maxClass = 0;
      for (let c = 0; c < 80; c++) {
        const score = output[(4 + c) * numDetections + i];
        if (score > maxScore) {
          maxScore = score;
          maxClass = c;
        }
      }

      if (maxScore < confidenceThreshold) continue;

      const cx = output[0 * numDetections + i];
      const cy = output[1 * numDetections + i];
      const w = output[2 * numDetections + i];
      const h = output[3 * numDetections + i];

      rawBoxes.push({
        label: COCO_LABELS[maxClass] || `class_${maxClass}`,
        confidence: Math.round(maxScore * 100) / 100,
        x: (cx - w / 2) * scaleX,
        y: (cy - h / 2) * scaleY,
        width: w * scaleX,
        height: h * scaleY,
      });
    }

    // NMS
    const final = nms(rawBoxes, IOU_THRESHOLD);

    setDetections(final);
    setInferenceMs(Math.round(performance.now() - start));
  }, [videoRef, confidenceThreshold]);

  // Detection loop
  useEffect(() => {
    if (!enabled || !modelLoaded) return;
    runningRef.current = true;
    const interval = 1000 / targetFps;

    function loop() {
      if (!runningRef.current) return;
      const now = performance.now();
      if (now - lastInferenceRef.current >= interval) {
        lastInferenceRef.current = now;
        runInference();
      }
      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      runningRef.current = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [enabled, modelLoaded, targetFps, runInference]);

  return { detections, inferenceMs, modelLoaded, modelError };
}

function nms(boxes: Detection[], iouThreshold: number): Detection[] {
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  for (const box of sorted) {
    let dominated = false;
    for (const k of kept) {
      if (iou(box, k) > iouThreshold) { dominated = true; break; }
    }
    if (!dominated) kept.push(box);
  }
  return kept;
}

function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter);
}
