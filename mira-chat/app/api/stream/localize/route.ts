import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DPVO_SERVER = process.env.DPVO_SERVER_URL || "http://localhost:8091";

// Localization state
let localizationActive = false;
let lastPose: Record<string, unknown> | null = null;
let poseCount = 0;
let lastLocalizeTime = 0;
const LOCALIZE_INTERVAL_MS = 2500; // Localize every 2.5 seconds max

// SSE clients waiting for pose updates
const poseClients = new Set<ReadableStreamDefaultController>();

/**
 * GET - Subscribe to pose updates via SSE
 */
export async function GET(req: NextRequest) {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      poseClients.add(controller);

      // Send current state
      const initMsg = JSON.stringify({
        type: "pose_init",
        active: localizationActive,
        lastPose,
        poseCount,
      });
      controller.enqueue(encoder.encode(`data: ${initMsg}\n\n`));

      req.signal.addEventListener("abort", () => {
        poseClients.delete(controller);
        try {
          controller.close();
        } catch {
          // Already closed
        }
      });
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

/**
 * Broadcast pose to all subscribed clients
 */
function broadcastPose(pose: Record<string, unknown>) {
  const encoder = new TextEncoder();
  const message = JSON.stringify({ type: "pose", ...pose });
  const data = encoder.encode(`data: ${message}\n\n`);

  poseClients.forEach((controller) => {
    try {
      controller.enqueue(data);
    } catch {
      // Client disconnected
    }
  });
}

/**
 * POST - Submit a frame for localization
 *
 * Accepts JPEG image data and forwards to DPVO server's /anchor endpoint.
 * Rate-limited to avoid overwhelming the GPU.
 */
export async function POST(req: NextRequest) {
  const now = Date.now();

  // Rate limit
  if (now - lastLocalizeTime < LOCALIZE_INTERVAL_MS) {
    return new Response(
      JSON.stringify({
        skipped: true,
        reason: "rate_limited",
        next_allowed_ms: LOCALIZE_INTERVAL_MS - (now - lastLocalizeTime),
      }),
      { status: 429, headers: { "Content-Type": "application/json" } }
    );
  }

  lastLocalizeTime = now;
  localizationActive = true;

  try {
    // Get the JPEG frame data
    const jpegData = Buffer.from(await req.arrayBuffer());

    if (jpegData.length < 100) {
      return new Response(JSON.stringify({ error: "Frame too small" }), {
        status: 400,
      });
    }

    // Forward to DPVO server's /anchor endpoint as multipart form
    const formData = new FormData();
    const blob = new Blob([jpegData], { type: "image/jpeg" });
    formData.append("image", blob, "frame.jpg");

    const t0 = Date.now();
    const response = await fetch(`${DPVO_SERVER}/anchor`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    const latencyMs = Date.now() - t0;

    const pose = {
      ...result,
      latency_ms: latencyMs,
      timestamp: new Date().toISOString(),
      frame_size: jpegData.length,
    };

    if (result.success) {
      lastPose = pose;
      poseCount++;
      console.log(
        `[Localize] Pose #${poseCount}: t=(${result.tx.toFixed(3)}, ${result.ty.toFixed(3)}, ${result.tz.toFixed(3)}) ` +
          `inliers=${result.num_inliers} latency=${latencyMs}ms`
      );
    } else {
      console.log(
        `[Localize] Failed: ${result.error || "unknown"} latency=${latencyMs}ms`
      );
    }

    // Broadcast to all SSE clients
    broadcastPose(pose);

    return new Response(JSON.stringify(pose), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("[Localize] Error:", error);
    const errorMsg =
      error instanceof Error ? error.message : "Localization failed";
    return new Response(JSON.stringify({ error: errorMsg, success: false }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
