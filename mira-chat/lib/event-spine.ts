import type { EventInput } from "./contracts";
import { getSupabaseServerClient } from "./supabase-server";

export async function appendEvent(event: EventInput) {
  const supabase = getSupabaseServerClient();
  const row = {
    patient_id: event.patient_id,
    type: event.type,
    severity: event.severity,
    receipt_text: event.receipt_text ?? null,
    decision: event.decision ?? {},
    payload: event.payload ?? {},
    evidence: event.evidence ?? {},
    source: event.source ?? "backend",
    schema_version: 1,
    idempotency_key: event.idempotency_key ?? null,
  };

  const { data, error } = await supabase.from("events").insert(row).select("*").single();
  if (error) throw error;
  return data;
}

export async function getEvents(patient_id: string, since?: string, limit: number = 50) {
  const supabase = getSupabaseServerClient();
  let query = supabase
    .from("events")
    .select("*")
    .eq("patient_id", patient_id)
    .order("created_at", { ascending: false })
    .limit(limit);

  if (since) query = query.gte("created_at", since);

  const { data, error } = await query;
  if (error) throw error;
  return data ?? [];
}

export async function getRecentEvents(patient_id: string, count: number = 10) {
  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase
    .from("events")
    .select("*")
    .eq("patient_id", patient_id)
    .order("created_at", { ascending: false })
    .limit(count);
  if (error) throw error;
  return data ?? [];
}

export async function logEscalation(params: {
  patient_id: string;
  event_id?: string;
  channel: string;
  target?: string;
  status: string;
  response_payload?: Record<string, unknown>;
}) {
  const supabase = getSupabaseServerClient();
  const { error } = await supabase.from("escalations").insert({
    patient_id: params.patient_id,
    event_id: params.event_id ?? null,
    channel: params.channel,
    target: params.target ?? null,
    status: params.status,
    response_payload: params.response_payload ?? {},
  });
  if (error) throw error;
}
