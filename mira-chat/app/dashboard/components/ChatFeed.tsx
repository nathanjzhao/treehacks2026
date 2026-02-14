"use client";

import { useEffect, useRef } from "react";
import { Streamdown } from "streamdown";
import { MiraEvent } from "@/lib/supabase/client";
import CitationCard from "@/app/components/CitationCard";

interface ChatFeedProps {
  events: MiraEvent[];
}

export default function ChatFeed({ events }: ChatFeedProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  // Events come newest-first; reverse for chronological chat view
  const chronological = [...events].reverse();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  if (chronological.length === 0) {
    return (
      <div className="text-center py-12 text-muted text-sm">
        No chat messages yet
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5 py-2">
      {chronological.map((event) => {
        const isUser = event.type === "CHAT_USER_UTTERANCE";
        const payload = event.payload as Record<string, unknown> | undefined;
        const action = payload?.action as string | undefined;
        const citations = payload?.citations as
          | Array<{ title?: string; url: string }>
          | undefined;

        return (
          <div
            key={event.id}
            className={`flex ${isUser ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] ${
                isUser
                  ? "bg-slate-100 rounded-2xl rounded-tr-sm"
                  : "bg-teal-50 border border-teal-100 rounded-2xl rounded-tl-sm"
              } px-4 py-2.5`}
            >
              <div
                className={`text-[10px] font-bold uppercase tracking-wider mb-1 ${
                  isUser ? "text-slate-400" : "text-teal-600"
                }`}
              >
                {isUser ? "Resident" : "Mira"}
              </div>

              <div className="text-sm text-slate-700 leading-relaxed prose prose-sm prose-slate max-w-none">
                <Streamdown mode="static">
                  {event.receipt_text || "..."}
                </Streamdown>
              </div>

              {/* Action badge */}
              {!isUser && action && action !== "ANSWER" && (
                <div
                  className={`mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-bold ${
                    action === "ESCALATE"
                      ? "bg-red-100 text-red-700"
                      : "bg-blue-100 text-blue-700"
                  }`}
                >
                  {action === "ESCALATE" ? "ESCALATED" : "OBJECT SEARCH"}
                </div>
              )}

              {/* Citations */}
              {!isUser && citations && citations.length > 0 && (
                <div className="mt-2 pt-2 border-t border-teal-100 space-y-1.5">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    Sources
                  </div>
                  {citations.map((c, i) => (
                    <CitationCard key={i} citation={c} variant="light" />
                  ))}
                </div>
              )}

              <div className="text-[10px] text-slate-400 mt-1.5">
                {new Date(event.created_at).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </div>
            </div>
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}
