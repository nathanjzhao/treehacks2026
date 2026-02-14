"use client";

import { Patient } from "@/lib/supabase/client";

interface SidebarProps {
  patients: (Patient & { card: Record<string, unknown> | null })[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

function getInitials(name: string): string {
  return name
    .split(" ")
    .map((w) => w[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);
}

const AVATAR_COLORS = [
  "from-teal-500 to-emerald-500",
  "from-amber-500 to-orange-500",
  "from-rose-500 to-pink-500",
  "from-cyan-500 to-blue-500",
  "from-violet-500 to-purple-500",
  "from-lime-500 to-green-500",
];

export default function Sidebar({ patients, selectedId, onSelect }: SidebarProps) {
  return (
    <aside className="w-[280px] h-screen bg-slate-900 text-white flex flex-col shrink-0">
      {/* Header */}
      <div className="p-5 border-b border-slate-700/60">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-teal-500/20">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 2a3 3 0 0 0-3 3v1a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 10H5a2 2 0 0 0-2 2v1a8 8 0 0 0 8 8h2a8 8 0 0 0 8-8v-1a2 2 0 0 0-2-2Z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight font-display">
              Mira
            </h1>
            <p className="text-[11px] text-slate-400 font-medium tracking-wide uppercase">
              Care Dashboard
            </p>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="px-5 py-3.5 border-b border-slate-700/60 flex gap-6">
        <div className="text-center">
          <div className="text-lg font-bold text-teal-400">
            {patients.length}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
            Residents
          </div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-emerald-400">
            {patients.filter((p) => p.card).length}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
            Active Cards
          </div>
        </div>
      </div>

      {/* Patient list */}
      <div className="flex-1 overflow-y-auto py-2">
        <div className="px-5 py-2.5">
          <span className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">
            Residents
          </span>
        </div>
        {patients.map((patient, idx) => {
          const isSelected = patient.id === selectedId;
          const gradientClass = AVATAR_COLORS[idx % AVATAR_COLORS.length];

          return (
            <button
              key={patient.id}
              onClick={() => onSelect(patient.id)}
              className={`w-full flex items-center gap-3 px-4 py-3.5 text-left transition-all duration-200 ${
                isSelected
                  ? "bg-teal-600/20 text-white border-r-2 border-teal-400"
                  : "hover:bg-slate-800/60 text-slate-300"
              }`}
            >
              <div
                className={`w-10 h-10 rounded-xl bg-gradient-to-br ${gradientClass} flex items-center justify-center text-sm font-bold text-white shrink-0 shadow-md`}
              >
                {getInitials(patient.display_name)}
              </div>
              <div className="min-w-0">
                <div className="font-semibold text-sm truncate">
                  {patient.display_name}
                </div>
                <div className="text-xs text-slate-400 mt-0.5">
                  {patient.room_number
                    ? `Room ${patient.room_number}`
                    : "No room assigned"}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700/60">
        <div className="text-[10px] text-slate-500 text-center font-medium">
          Mira v1.0 &middot; TreeHacks 2026
        </div>
      </div>
    </aside>
  );
}
