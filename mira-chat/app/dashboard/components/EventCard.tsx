"use client";

import { MiraEvent } from "@/lib/supabase/client";

interface EventCardProps {
  event: MiraEvent;
}

const TYPE_META: Record<string, { label: string; icon: string; color: string }> = {
  CHAT_USER_UTTERANCE: { label: "Resident Message", icon: "\u{1F4AC}", color: "bg-slate-100 text-slate-700" },
  CHAT_ASSISTANT_RESPONSE: { label: "Mira Response", icon: "\u{1F916}", color: "bg-teal-100 text-teal-700" },
  FIND_OBJECT_REQUESTED: { label: "Object Search", icon: "\u{1F50D}", color: "bg-blue-100 text-blue-700" },
  OBJECT_LOCATED: { label: "Object Found", icon: "\u{1F4CD}", color: "bg-emerald-100 text-emerald-700" },
  OBJECT_NOT_FOUND: { label: "Not Found", icon: "\u{274C}", color: "bg-amber-100 text-amber-700" },
  ESCALATED: { label: "Escalation", icon: "\u{1F6A8}", color: "bg-red-100 text-red-700" },
  WEB_SEARCH_COMPLETED: { label: "Web Search", icon: "\u{1F310}", color: "bg-indigo-100 text-indigo-700" },
  CLINICAL_GUIDELINE_CHECKED: { label: "Clinical Guideline", icon: "\u{1F3E5}", color: "bg-emerald-100 text-emerald-700" },
};

const SEVERITY_COLORS: Record<string, string> = {
  GREEN: "bg-emerald-500",
  YELLOW: "bg-amber-500",
  ORANGE: "bg-orange-500",
  RED: "bg-red-500",
};

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(dateStr).toLocaleDateString();
}

export default function EventCard({ event }: EventCardProps) {
  const meta = TYPE_META[event.type] || {
    label: event.type,
    icon: "\u{1F535}",
    color: "bg-slate-100 text-slate-700",
  };
  const isObjectLocated = event.type === "OBJECT_LOCATED";
  const isEscalation = event.type === "ESCALATED";

  return (
    <div
      className={`rounded-2xl border p-4 transition-all fade-in ${
        isEscalation
          ? "border-red-200 bg-red-50/80"
          : isObjectLocated
          ? "border-emerald-200 bg-emerald-50/80"
          : "border-border bg-surface"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-base">{meta.icon}</span>
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-lg text-[11px] font-bold ${meta.color}`}
          >
            {meta.label}
          </span>
          <span
            className={`w-2 h-2 rounded-full ${
              SEVERITY_COLORS[event.severity] || "bg-slate-400"
            }`}
            title={event.severity}
          />
        </div>
        <span className="text-[11px] text-muted shrink-0 font-medium">
          {timeAgo(event.created_at)}
        </span>
      </div>

      {event.receipt_text && (
        <p className="mt-2.5 text-sm text-foreground leading-relaxed">
          {event.receipt_text}
        </p>
      )}

      {/* Object located details */}
      {isObjectLocated && event.payload && (
        <div className="mt-3 p-3 bg-emerald-100/80 rounded-xl border border-emerald-200/50">
          <div className="text-[10px] font-bold text-emerald-800 uppercase tracking-wider">
            Location
          </div>
          <div className="text-sm text-emerald-900 mt-1 font-medium">
            {(
              event.payload as {
                location?: { description?: string };
              }
            ).location?.description || JSON.stringify(event.payload.location)}
          </div>
          {event.payload.confidence != null && (
            <div className="text-xs text-emerald-700 mt-1">
              Confidence:{" "}
              {Math.round((event.payload.confidence as number) * 100)}%
            </div>
          )}
        </div>
      )}

      {/* Clinical guideline evidence level */}
      {event.type === "CLINICAL_GUIDELINE_CHECKED" && event.payload && (
        <div className="mt-3 p-3 bg-emerald-50 rounded-xl border border-emerald-200/50">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold text-emerald-800 uppercase tracking-wider">
              Evidence Level
            </span>
            <span
              className={`text-[9px] font-bold px-2 py-0.5 rounded ${
                (event.payload as any).evidence_level === "high"
                  ? "bg-emerald-200 text-emerald-800"
                  : (event.payload as any).evidence_level === "moderate"
                  ? "bg-amber-200 text-amber-800"
                  : "bg-slate-200 text-slate-600"
              }`}
            >
              {((event.payload as any).evidence_level || "unknown").toUpperCase()}
            </span>
          </div>
          {(event.payload as any).condition && (
            <div className="text-sm text-emerald-900 mt-1 font-medium">
              {(event.payload as any).condition}
            </div>
          )}
        </div>
      )}

      {/* Source badge */}
      <div className="mt-2.5 flex items-center gap-2">
        <span className="text-[10px] text-muted uppercase tracking-wider font-bold">
          {event.source}
        </span>
      </div>
    </div>
  );
}
