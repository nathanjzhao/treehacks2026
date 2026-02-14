"use client";

interface StatusIndicatorProps {
  connected: boolean;
  searchingFor?: string | null;
}

export default function StatusIndicator({ connected, searchingFor }: StatusIndicatorProps) {
  return (
    <div className="flex items-center gap-3 text-xs">
      {searchingFor ? (
        <div className="flex items-center gap-2 bg-white/15 backdrop-blur-sm text-white px-3.5 py-2 rounded-full border border-white/10">
          <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
          <span className="font-medium text-[11px]">Searching for {searchingFor}...</span>
        </div>
      ) : (
        <div className="flex items-center gap-2 bg-white/10 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/10">
          <div
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-emerald-400" : "bg-red-400"
            }`}
          />
          <span className="text-white/80 text-[11px] font-medium">
            {connected ? "Connected" : "Offline"}
          </span>
        </div>
      )}
    </div>
  );
}
