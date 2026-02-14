import { NextRequest, NextResponse } from "next/server";
import { appendEvent } from "@/lib/event-spine";
import { getSupabaseServerClient } from "@/lib/supabase-server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { request_id, patient_id, object_name, found, location, confidence, source } = body;

    if (!patient_id || typeof patient_id !== "string") {
      return NextResponse.json({ ok: false, error: "patient_id is required" }, { status: 400 });
    }
    if (!object_name || typeof object_name !== "string") {
      return NextResponse.json({ ok: false, error: "object_name is required" }, { status: 400 });
    }
    if (typeof found !== "boolean") {
      return NextResponse.json({ ok: false, error: "found (boolean) is required" }, { status: 400 });
    }

    const supabase = getSupabaseServerClient();

    // Update object_requests row if request_id provided
    if (request_id) {
      const { error: updateErr } = await supabase
        .from("object_requests")
        .update({
          status: found ? "FOUND" : "NOT_FOUND",
          resolved_at: new Date().toISOString(),
          resolved_location: found && location ? (typeof location === "string" ? { description: location } : location) : null,
          confidence: confidence ?? null,
        })
        .eq("id", request_id);

      if (updateErr) console.error("[objects/update] Failed to update object_requests:", updateErr);
    }

    // Upsert object_state if found
    if (found) {
      const locationJson = typeof location === "string" ? { description: location } : (location || {});
      const { error: upsertErr } = await supabase
        .from("object_state")
        .upsert(
          {
            patient_id,
            object_name: object_name.trim(),
            location: locationJson,
            confidence: confidence ?? null,
            updated_at: new Date().toISOString(),
            source: source || "cv_team",
          },
          { onConflict: "patient_id,object_name" }
        );

      if (upsertErr) console.error("[objects/update] Failed to upsert object_state:", upsertErr);
    }

    // Emit event
    const event = await appendEvent({
      patient_id,
      type: found ? "OBJECT_LOCATED" : "OBJECT_NOT_FOUND",
      severity: found ? "GREEN" : "YELLOW",
      receipt_text: found
        ? `Found ${object_name}: ${typeof location === "string" ? location : JSON.stringify(location) || "location reported"}`
        : `Could not find ${object_name}`,
      payload: {
        object_name: object_name.trim(),
        request_id: request_id || null,
        location: typeof location === "string" ? { description: location } : (location || null),
        confidence: confidence ?? null,
      },
      source: "cv_team",
    });

    return NextResponse.json({ ok: true, event }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "unknown_error" },
      { status: 400 }
    );
  }
}
