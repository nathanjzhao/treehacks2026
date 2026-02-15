import { NextRequest } from "next/server";
import frameStore from "@/lib/frameStore";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  console.log("[MJPEG] Client connected to stream");

  const encoder = new TextEncoder();

  // Check if we have frames before starting
  if (!frameStore.hasFrame()) {
    console.log("[MJPEG] No frames available yet");
    return new Response("No frames available. Waiting for Android stream...", {
      status: 503,
      headers: { "Content-Type": "text/plain" },
    });
  }

  // Create a readable stream that serves Motion JPEG
  const stream = new ReadableStream({
    start(controller) {
      const boundary = "FRAME_BOUNDARY";
      let frameCount = 0;

      const sendFrame = () => {
        try {
          const frame = frameStore.getFrame();
          if (frame) {
            // Send multipart headers
            const header =
              `--${boundary}\r\n` +
              `Content-Type: image/jpeg\r\n` +
              `Content-Length: ${frame.data.length}\r\n\r\n`;

            controller.enqueue(encoder.encode(header));
            controller.enqueue(frame.data);
            controller.enqueue(encoder.encode("\r\n"));

            frameCount++;
            if (frameCount % 30 === 0) {
              console.log(`[MJPEG] Sent ${frameCount} frames, latest: #${frame.frameNumber}`);
            }
          }
        } catch (error) {
          console.error("[MJPEG] Error sending frame:", error);
          clearInterval(interval);
          controller.close();
        }
      };

      // Send frames at 30 FPS (every ~33ms)
      const interval = setInterval(sendFrame, 33);

      // Cleanup on close
      request.signal.addEventListener("abort", () => {
        console.log(`[MJPEG] Client disconnected after ${frameCount} frames`);
        clearInterval(interval);
        try {
          controller.close();
        } catch (e) {
          // Already closed
        }
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "multipart/x-mixed-replace; boundary=FRAME_BOUNDARY",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "Connection": "keep-alive",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
