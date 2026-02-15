import frameStore from "@/lib/frameStore";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** GET /api/stream/frame/jpeg — Returns the latest JPEG frame as raw binary. */
export async function GET() {
  const frame = frameStore.getFrame();

  if (!frame) {
    return new Response("No frame", { status: 404 });
  }

  return new Response(frame.data, {
    status: 200,
    headers: {
      "Content-Type": "image/jpeg",
      "Cache-Control": "no-cache, no-store",
      "X-Frame-Number": String(frame.frameNumber),
      "X-Timestamp": frame.timestamp,
    },
  });
}
