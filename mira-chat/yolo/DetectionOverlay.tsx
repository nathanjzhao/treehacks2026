"use client";

import { useEffect, useRef } from "react";
import { Detection, getColor } from "./types";

interface DetectionOverlayProps {
  detections: Detection[];
  /** Video element dimensions (used to map box coords to canvas) */
  videoWidth: number;
  videoHeight: number;
  /** Canvas CSS size (usually the container size) */
  canvasWidth: number;
  canvasHeight: number;
  inferenceMs?: number;
  showFps?: boolean;
}

export default function DetectionOverlay({
  detections,
  videoWidth,
  videoHeight,
  canvasWidth,
  canvasHeight,
  inferenceMs = 0,
  showFps = true,
}: DetectionOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas resolution to match container
    canvas.width = canvasWidth * (window.devicePixelRatio || 1);
    canvas.height = canvasHeight * (window.devicePixelRatio || 1);
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Scale factor: video coords → canvas display coords
    const sx = canvasWidth / videoWidth;
    const sy = canvasHeight / videoHeight;

    for (const det of detections) {
      const x = det.x * sx;
      const y = det.y * sy;
      const w = det.width * sx;
      const h = det.height * sy;
      const color = getColor(det.label);
      const pct = Math.round(det.confidence * 100);

      // Box outline (double stroke for glow effect)
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.strokeRect(x, y, w, h);
      ctx.shadowBlur = 0;

      // Corner accents (thicker corners for HUD feel)
      const cornerLen = Math.min(w, h, 16);
      ctx.lineWidth = 3;
      // Top-left
      ctx.beginPath();
      ctx.moveTo(x, y + cornerLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLen, y);
      ctx.stroke();
      // Top-right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerLen);
      ctx.stroke();
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x, y + h - cornerLen);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + cornerLen, y + h);
      ctx.stroke();
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - cornerLen);
      ctx.stroke();

      ctx.lineWidth = 1;

      // Label background
      const label = `${det.label} ${pct}%`;
      ctx.font = "bold 11px 'DM Mono', monospace";
      const textWidth = ctx.measureText(label).width;
      const labelH = 18;
      const labelY = y > labelH + 4 ? y - labelH - 2 : y + h + 2;

      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.beginPath();
      ctx.roundRect(x, labelY, textWidth + 12, labelH, 4);
      ctx.fill();

      // Label border
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x, labelY, textWidth + 12, labelH, 4);
      ctx.stroke();

      // Label text
      ctx.fillStyle = color;
      ctx.fillText(label, x + 6, labelY + 13);
    }

    // FPS / inference info (bottom-right of overlay)
    if (showFps && inferenceMs > 0) {
      const fps = Math.round(1000 / inferenceMs);
      const info = `YOLO ${inferenceMs}ms · ${fps} FPS · ${detections.length} obj`;
      ctx.font = "500 10px 'DM Mono', monospace";
      const tw = ctx.measureText(info).width;
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.beginPath();
      ctx.roundRect(canvasWidth - tw - 16, 8, tw + 12, 18, 4);
      ctx.fill();
      ctx.fillStyle = "rgba(120,255,200,0.6)";
      ctx.fillText(info, canvasWidth - tw - 10, 21);
    }
  }, [detections, videoWidth, videoHeight, canvasWidth, canvasHeight, inferenceMs, showFps]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 1,
      }}
    />
  );
}
