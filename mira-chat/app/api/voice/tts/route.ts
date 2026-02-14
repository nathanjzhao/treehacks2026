import { NextRequest, NextResponse } from "next/server";

// Default to Alice; override with ELEVENLABS_VOICE_ID env var
// To use Katie: find her voice_id in your ElevenLabs Voice Library and set it
const DEFAULT_VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"; // Alice

export async function POST(request: NextRequest) {
  try {
    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ ok: false, error: "ELEVENLABS_API_KEY not configured" }, { status: 500 });
    }

    const body = await request.json();
    const { text } = body;

    if (!text || typeof text !== "string") {
      return NextResponse.json({ ok: false, error: "text is required" }, { status: 400 });
    }

    const voiceId = process.env.ELEVENLABS_VOICE_ID || DEFAULT_VOICE_ID;

    const res = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        method: "POST",
        headers: {
          "xi-api-key": apiKey,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        body: JSON.stringify({
          text,
          model_id: "eleven_turbo_v2_5",
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.3,
            use_speaker_boost: true,
          },
        }),
      }
    );

    if (!res.ok) {
      const errText = await res.text();
      console.error("[ElevenLabs] Error:", res.status, errText);
      return NextResponse.json(
        { ok: false, error: `ElevenLabs API error: ${res.status}` },
        { status: 502 }
      );
    }

    // Stream audio back
    const audioBuffer = await res.arrayBuffer();
    return new NextResponse(audioBuffer, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Cache-Control": "no-cache",
      },
    });
  } catch (error) {
    console.error("[ElevenLabs] Error:", error);
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "unknown_error" },
      { status: 500 }
    );
  }
}
