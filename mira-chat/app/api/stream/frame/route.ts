import { NextRequest, NextResponse } from "next/server";
import frameStore from "@/lib/frameStore";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const imageFile = formData.get("image") as File;
    const timestamp = formData.get("timestamp") as string;
    const frameNumber = parseInt(formData.get("frame_number") as string);
    const width = parseInt(formData.get("width") as string);
    const height = parseInt(formData.get("height") as string);

    if (!imageFile) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 });
    }

    // Store latest frame in shared store (optimized - no extra copy)
    const arrayBuffer = await imageFile.arrayBuffer();
    frameStore.setFrame({
      data: Buffer.from(arrayBuffer),
      timestamp,
      width,
      height,
      frameNumber,
    });

    // Log every 30th frame
    if (frameNumber % 30 === 0) {
      console.log(`[Frame] Stored frame #${frameNumber}, ${arrayBuffer.byteLength} bytes, ${width}x${height}`);
    }

    // Fast response (don't wait for logging)
    return new Response(JSON.stringify({ success: true, frameNumber }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("[Frame] Upload error:", error);
    return new Response(JSON.stringify({ error: "Upload failed" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

export async function GET() {
  const frame = frameStore.getFrame();

  if (!frame) {
    return NextResponse.json({ error: "No frames available" }, { status: 404 });
  }

  return NextResponse.json({
    timestamp: frame.timestamp,
    width: frame.width,
    height: frame.height,
    frameNumber: frame.frameNumber,
    size: frame.data.length,
  });
}
