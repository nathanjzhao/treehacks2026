"use client";

import { MiraEvent } from "@/lib/supabase/client";

interface AlertBannerProps {
  alerts: MiraEvent[];
  onDismiss: (id: string) => void;
}

export default function AlertBanner({ alerts, onDismiss }: AlertBannerProps) {
  if (alerts.length === 0) return null;

  return (
    <div className="space-y-2 px-6 pt-4">
      {alerts.map((alert) => (
        <div
          key={alert.id}
          className="slide-down flex items-center justify-between bg-red-50 border border-red-200 rounded-xl px-4 py-3"
        >
          <div className="flex items-center gap-3">
            <span className="text-lg">{"\uD83D\uDEA8"}</span>
            <div>
              <span className="text-sm font-semibold text-red-800">
                {alert.type === "ESCALATED" ? "Escalation Alert" : "Critical Event"}
              </span>
              <p className="text-sm text-red-700 mt-0.5">{alert.receipt_text}</p>
            </div>
          </div>
          <button
            onClick={() => onDismiss(alert.id)}
            className="text-red-400 hover:text-red-600 transition-colors p-1"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
}
