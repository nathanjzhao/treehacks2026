import { NextRequest } from "next/server";
import { subscribeSteps } from "@/lib/step-bus";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const patientId = request.nextUrl.searchParams.get("patient_id");
  if (!patientId) {
    return new Response(
      JSON.stringify({ error: "patient_id required" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      // Initial keepalive
      controller.enqueue(encoder.encode(": connected\n\n"));

      const unsub = subscribeSteps(patientId, (data) => {
        try {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
          );
        } catch {
          unsub();
        }
      });

      // Clean up when client disconnects
      request.signal.addEventListener("abort", () => {
        unsub();
        try {
          controller.close();
        } catch {
          /* already closed */
        }
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
