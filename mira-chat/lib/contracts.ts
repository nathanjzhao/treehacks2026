export const EVENT_TYPES = [
  "CHAT_USER_UTTERANCE",
  "CHAT_ASSISTANT_RESPONSE",
  "CHAT_STEP",
  "FIND_OBJECT_REQUESTED",
  "OBJECT_LOCATED",
  "OBJECT_NOT_FOUND",
  "ESCALATED",
  "WEB_SEARCH_COMPLETED",
  "CLINICAL_GUIDELINE_CHECKED",
] as const;

export const SEVERITIES = ["GREEN", "YELLOW", "ORANGE", "RED"] as const;
export const SOURCES = ["device", "dashboard", "backend", "cv_team"] as const;

export type EventType = (typeof EVENT_TYPES)[number];
export type Severity = (typeof SEVERITIES)[number];
export type Source = (typeof SOURCES)[number];

export type EventInput = {
  patient_id: string;
  type: EventType;
  severity: Severity;
  receipt_text?: string;
  decision?: Record<string, unknown>;
  payload?: Record<string, unknown>;
  evidence?: Record<string, unknown>;
  source?: Source;
  idempotency_key?: string;
};

const eventTypeSet = new Set<string>(EVENT_TYPES);
const severitySet = new Set<string>(SEVERITIES);
const sourceSet = new Set<string>(SOURCES);

export function parseEventInput(body: Record<string, unknown>): EventInput {
  const { patient_id, type, severity, receipt_text, decision, payload, evidence, source, idempotency_key } = body;

  if (!patient_id || typeof patient_id !== "string") {
    throw new Error("patient_id (string) is required");
  }
  if (!type || typeof type !== "string" || !eventTypeSet.has(type)) {
    throw new Error(`type must be one of: ${EVENT_TYPES.join(", ")}`);
  }
  if (!severity || typeof severity !== "string" || !severitySet.has(severity)) {
    throw new Error(`severity must be one of: ${SEVERITIES.join(", ")}`);
  }
  if (source && (typeof source !== "string" || !sourceSet.has(source))) {
    throw new Error(`source must be one of: ${SOURCES.join(", ")}`);
  }

  return {
    patient_id: patient_id as string,
    type: type as EventType,
    severity: severity as Severity,
    receipt_text: typeof receipt_text === "string" ? receipt_text : undefined,
    decision: (decision as Record<string, unknown>) ?? undefined,
    payload: (payload as Record<string, unknown>) ?? undefined,
    evidence: (evidence as Record<string, unknown>) ?? undefined,
    source: (source as Source) ?? undefined,
    idempotency_key: typeof idempotency_key === "string" ? idempotency_key : undefined,
  };
}

export function parseQueryFilters(params: URLSearchParams): { patient_id: string; since?: string } {
  const patient_id = params.get("patient_id");
  if (!patient_id) throw new Error("patient_id query param is required");
  const since = params.get("since") ?? undefined;
  return { patient_id, since };
}
