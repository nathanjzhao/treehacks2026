import { NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabase-server";

export async function GET() {
  try {
    const supabase = getSupabaseServerClient();

    // Get all patients
    const { data: patients, error: pErr } = await supabase
      .from("patients")
      .select("*")
      .order("display_name");

    if (pErr) throw pErr;

    // Get latest patient card for each patient
    const patientIds = (patients ?? []).map((p) => p.id);
    const { data: cards, error: cErr } = await supabase
      .from("patient_cards")
      .select("*")
      .in("patient_id", patientIds)
      .order("created_at", { ascending: false });

    if (cErr) throw cErr;

    // Deduplicate: keep only the latest card per patient
    const latestCards = new Map<string, Record<string, unknown>>();
    for (const card of cards ?? []) {
      if (!latestCards.has(card.patient_id)) {
        latestCards.set(card.patient_id, card);
      }
    }

    const result = (patients ?? []).map((p) => ({
      ...p,
      card: latestCards.get(p.id)?.card_json || null,
    }));

    return NextResponse.json({ ok: true, patients: result });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "unknown_error" },
      { status: 500 }
    );
  }
}
