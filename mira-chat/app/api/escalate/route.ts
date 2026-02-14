import { NextRequest, NextResponse } from "next/server";
import { appendEvent, logEscalation } from "@/lib/event-spine";
import { sendSms, isE164 } from "@/lib/twilio";
import { getSupabaseServerClient } from "@/lib/supabase-server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { patient_id, reason, channel, target_phone } = body;

    if (!patient_id || typeof patient_id !== "string") {
      return NextResponse.json({ ok: false, error: "patient_id is required" }, { status: 400 });
    }
    if (!reason || typeof reason !== "string") {
      return NextResponse.json({ ok: false, error: "reason is required" }, { status: 400 });
    }
    if (target_phone && !isE164(target_phone)) {
      return NextResponse.json({ ok: false, error: "target_phone must be E.164 format (e.g. +15551234567)" }, { status: 400 });
    }

    const event = await appendEvent({
      patient_id,
      type: "ESCALATED",
      severity: "RED",
      receipt_text: reason,
      payload: { channel: channel || "sms", target_phone: target_phone || null },
      source: "dashboard",
    });

    await logEscalation({
      patient_id,
      event_id: event.id,
      channel: channel || "sms",
      target: target_phone,
      status: "sent",
      response_payload: { reason },
    });

    // Get patient name for SMS
    const supabase = getSupabaseServerClient();
    const { data: patient } = await supabase
      .from("patients")
      .select("display_name")
      .eq("id", patient_id)
      .single();

    sendSms(
      `Mira Alert: ${patient?.display_name || "Resident"} - ${reason}`,
      target_phone
    ).catch(console.error);

    return NextResponse.json({ ok: true, event }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "unknown_error" },
      { status: 400 }
    );
  }
}
