import { NextRequest } from "next/server";
import frameStore, { FrameData } from "@/lib/frameStore";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// Store active streaming clients for frame broadcasts
const streamingClients = new Set<ReadableStreamDefaultController>();
let frameCount = 0;
let bytesReceived = 0;
const connectedAt = Date.now();

/**
 * GET - Stream frames to web clients using Server-Sent Events
 */
export async function GET(req: NextRequest) {
  const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  console.log(`[Stream] ${clientId} connected`);

  const encoder = new TextEncoder();

  // Create streaming response
  const stream = new ReadableStream({
    start(controller) {
      streamingClients.add(controller);

      // Send initial connection message
      const initMsg = JSON.stringify({
        type: "connected",
        timestamp: new Date().toISOString(),
        clientId,
      });
      controller.enqueue(encoder.encode(`data: ${initMsg}\n\n`));

      // Handle client disconnect
      req.signal.addEventListener("abort", () => {
        streamingClients.delete(controller);
        console.log(`[Stream] ${clientId} disconnected (${streamingClients.size} remaining)`);
        try {
          controller.close();
        } catch (e) {
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
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

/**
 * Broadcast frame to all connected streaming clients
 */
function broadcastFrame(frame: FrameData) {
  const encoder = new TextEncoder();
  const message = JSON.stringify({
    type: "frame",
    timestamp: frame.timestamp,
    frame_number: frame.frameNumber,
    width: frame.width,
    height: frame.height,
    jpeg_base64: frame.data.toString("base64"),
  });

  const data = encoder.encode(`data: ${message}\n\n`);

  streamingClients.forEach((controller) => {
    try {
      controller.enqueue(data);
    } catch (e) {
      // Client disconnected, will be removed by abort handler
    }
  });
}

export async function POST(req: NextRequest) {
  try {
    // Handle binary frame data from Android
    const contentType = req.headers.get("content-type");

    if (contentType?.includes("application/octet-stream")) {
      // Binary frame data
      const buffer = Buffer.from(await req.arrayBuffer());

      if (buffer.length < 4) {
        return new Response(JSON.stringify({ error: "Frame too short" }), { status: 400 });
      }

      // Parse wire format: [4-byte length] + [JSON] + [JPEG]
      const headerLength = buffer.readUInt32BE(0);

      if (buffer.length < 4 + headerLength) {
        return new Response(JSON.stringify({ error: "Header length mismatch" }), { status: 400 });
      }

      const metadataBytes = buffer.subarray(4, 4 + headerLength);
      const metadata = JSON.parse(metadataBytes.toString("utf-8"));

      const jpegData = buffer.subarray(4 + headerLength);

      if (jpegData.length !== metadata.jpeg_size) {
        return new Response(
          JSON.stringify({
            error: `JPEG size mismatch: expected ${metadata.jpeg_size}, got ${jpegData.length}`,
          }),
          { status: 400 }
        );
      }

      // Store frame
      const frame: FrameData = {
        data: jpegData,
        timestamp: metadata.timestamp,
        width: metadata.width,
        height: metadata.height,
        frameNumber: metadata.frame_number,
      };

      frameStore.setFrame(frame);
      frameCount++;
      bytesReceived += buffer.length;

      // Broadcast to connected web clients
      broadcastFrame(frame);

      // Log every 30th frame
      if (frameCount % 30 === 0) {
        const uptime = (Date.now() - connectedAt) / 1000;
        const fps = (frameCount / uptime).toFixed(1);
        const mbps = (bytesReceived / uptime / 1024 / 1024).toFixed(2);
        console.log(`[Stream] ${frameCount} frames, ${mbps} MB/s, ${fps} FPS, ${streamingClients.size} clients`);
      }

      return new Response(JSON.stringify({ success: true, frameNumber: metadata.frame_number }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Handle control messages (text JSON)
    const message = await req.json();

    switch (message.type) {
      case "ping":
        return new Response(JSON.stringify({ type: "pong", timestamp: new Date().toISOString() }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });

      case "stats_request":
        return new Response(
          JSON.stringify({
            type: "stats_response",
            frames_received: frameStore.getTotalFrames(),
            uptime_seconds: frameStore.getUptimeSeconds(),
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }
        );

      default:
        console.warn(`[WS] Unknown control message: ${message.type}`);
        return new Response(JSON.stringify({ error: "Unknown message type" }), { status: 400 });
    }
  } catch (error) {
    console.error("[WS] Error processing request:", error);
    return new Response(JSON.stringify({ error: "Processing failed" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
