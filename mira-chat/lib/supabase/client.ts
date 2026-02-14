import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder";

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// --- Type definitions ---

export type EventType =
  | "CHAT_USER_UTTERANCE"
  | "CHAT_ASSISTANT_RESPONSE"
  | "FIND_OBJECT_REQUESTED"
  | "OBJECT_LOCATED"
  | "OBJECT_NOT_FOUND"
  | "ESCALATED"
  | "WEB_SEARCH_COMPLETED";

export type Severity = "GREEN" | "YELLOW" | "ORANGE" | "RED";

export interface MiraEvent {
  id: string;
  patient_id: string;
  type: EventType;
  severity: Severity;
  receipt_text?: string;
  decision: Record<string, unknown>;
  payload: Record<string, unknown>;
  evidence: Record<string, unknown>;
  source: string;
  schema_version: number;
  idempotency_key?: string;
  created_at: string;
  updated_at: string;
}

export interface Patient {
  id: string;
  external_id?: string;
  display_name: string;
  room_number?: string;
  photo_url?: string;
  created_at: string;
  updated_at: string;
}

export interface PatientCard {
  display_name: string;
  room_number?: string;
  demographics: { age_range: string; sex: string };
  conditions: Array<{ name: string; onset_year?: number }>;
  allergies: Array<{ substance: string; reaction?: string }>;
  meds: Array<{
    name: string;
    strength?: string;
    schedule_times?: string[];
    purpose?: string;
    with_food?: boolean;
    is_critical?: boolean;
  }>;
  contacts: {
    caregiver_name: string;
    caregiver_phone: string;
    emergency_phone: string;
  };
}

export interface ObjectRequest {
  id: string;
  patient_id: string;
  object_name: string;
  context?: string;
  status: "PENDING" | "FOUND" | "NOT_FOUND";
  created_at: string;
  resolved_at?: string;
  resolved_location?: Record<string, unknown>;
  confidence?: number;
}

export interface ObjectState {
  patient_id: string;
  object_name: string;
  location?: Record<string, unknown>;
  confidence?: number;
  updated_at: string;
  source?: string;
}
