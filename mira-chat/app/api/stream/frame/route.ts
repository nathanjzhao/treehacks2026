import { NextRequest, NextResponse } from "next/server";
import frameStore from "@/lib/frameStore";

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

    // Store latest frame in shared store
    const arrayBuffer = await imageFile.arrayBuffer();
    const frameData = {
      data: Buffer.from(arrayBuffer),
      timestamp,
      width,
      height,
      frameNumber,
    };

    frameStore.setFrame(frameData);

    // Log every 30th frame
    if (frameNumber % 30 === 0) {
      console.log(`[Frame] Stored frame #${frameNumber}, ${frameData.data.length} bytes, ${width}x${height}`);
    }

    return NextResponse.json({
      success: true,
      frameNumber,
      size: frameData.data.length,
    });
  } catch (error) {
    console.error("[Frame] Upload error:", error);
    return NextResponse.json({ error: "Upload failed" }, { status: 500 });
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
