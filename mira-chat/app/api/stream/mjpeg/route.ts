import { NextRequest } from "next/server";
import frameStore from "@/lib/frameStore";

export async function GET(request: NextRequest) {
  const encoder = new TextEncoder();

  // Create a readable stream that serves Motion JPEG
  const stream = new ReadableStream({
    async start(controller) {
      const boundary = "FRAME_BOUNDARY";

      const sendFrame = () => {
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
        }
      };

      // Send frames at 30 FPS (every ~33ms)
      const interval = setInterval(sendFrame, 33);

      // Cleanup on close
      request.signal.addEventListener("abort", () => {
        clearInterval(interval);
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "multipart/x-mixed-replace; boundary=FRAME_BOUNDARY",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "Connection": "keep-alive",
    },
  });
}
