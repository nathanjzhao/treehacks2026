"use client";

import { useState, useRef, useEffect } from "react";
import { MiraEvent } from "@/lib/supabase/client";
import EventCard from "./EventCard";
import ChatFeed from "./ChatFeed";

interface EventTimelineProps {
  events: MiraEvent[];
}

type Filter = "all" | "chat" | "objects" | "escalations";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "chat", label: "Chat" },
  { key: "objects", label: "Objects" },
  { key: "escalations", label: "Alerts" },
];

function filterEvents(events: MiraEvent[], filter: Filter): MiraEvent[] {
  switch (filter) {
    case "chat":
      return events.filter(
        (e) =>
          e.type === "CHAT_USER_UTTERANCE" ||
          e.type === "CHAT_ASSISTANT_RESPONSE"
      );
    case "objects":
      return events.filter(
        (e) =>
          e.type === "FIND_OBJECT_REQUESTED" ||
          e.type === "OBJECT_LOCATED" ||
          e.type === "OBJECT_NOT_FOUND"
      );
    case "escalations":
      return events.filter((e) => e.type === "ESCALATED");
    default:
      return events;
  }
}

export default function EventTimeline({ events }: EventTimelineProps) {
  const [filter, setFilter] = useState<Filter>("all");
  const scrollRef = useRef<HTMLDivElement>(null);
  const filtered = filterEvents(events, filter);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events.length]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-5 pt-5 pb-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
          Event Timeline
        </h3>
        <div className="flex gap-1.5 mt-3">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                filter === f.key
                  ? "bg-teal-600 text-white shadow-sm"
                  : "bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Events */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 pb-5 space-y-3">
        {filter === "chat" ? (
          <ChatFeed events={filtered} />
        ) : filtered.length === 0 ? (
          <div className="text-center py-12 text-muted text-sm">
            No events to display
          </div>
        ) : (
          filtered.map((event) => (
            <EventCard key={event.id} event={event} />
          ))
        )}
      </div>
    </div>
  );
}
