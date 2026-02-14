import { NextRequest, NextResponse } from "next/server";
import { appendEvent } from "@/lib/event-spine";
import { getSupabaseServerClient } from "@/lib/supabase-server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { patient_id, object_name, context } = body;

    if (!patient_id || typeof patient_id !== "string") {
      return NextResponse.json({ ok: false, error: "patient_id is required" }, { status: 400 });
    }
    if (!object_name || typeof object_name !== "string") {
      return NextResponse.json({ ok: false, error: "object_name is required" }, { status: 400 });
    }

    const supabase = getSupabaseServerClient();

    // Create object_requests row with PENDING status
    const { data: reqRow, error: reqErr } = await supabase
      .from("object_requests")
      .insert({
        patient_id,
        object_name: object_name.trim(),
        context: context || "chat_request",
        status: "PENDING",
      })
      .select("*")
      .single();

    if (reqErr) throw reqErr;

    // Emit FIND_OBJECT_REQUESTED event with request_id
    const event = await appendEvent({
      patient_id,
      type: "FIND_OBJECT_REQUESTED",
      severity: "GREEN",
      receipt_text: `Looking for: ${object_name}`,
      payload: {
        object_name: object_name.trim(),
        request_id: reqRow.id,
        context: context || "chat_request",
      },
      source: "device",
    });

    return NextResponse.json(
      { ok: true, request_id: reqRow.id, event },
      { status: 201 }
    );
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "unknown_error" },
      { status: 400 }
    );
  }
}
