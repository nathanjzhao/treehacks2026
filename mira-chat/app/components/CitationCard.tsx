"use client";

interface CitationCardProps {
  citation: { title?: string; url: string };
  variant?: "light" | "dark";
}

export default function CitationCard({
  citation,
  variant = "light",
}: CitationCardProps) {
  let hostname = "";
  try {
    hostname = new URL(citation.url).hostname.replace("www.", "");
  } catch {
    hostname = citation.url;
  }
  const faviconUrl = `https://www.google.com/s2/favicons?domain=${hostname}&sz=32`;
  const displayTitle = citation.title || hostname;

  if (variant === "dark") {
    return (
      <a
        href={citation.url}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "6px 10px",
          background: "rgba(120,255,200,0.06)",
          border: "1px solid rgba(120,255,200,0.15)",
          borderRadius: 8,
          textDecoration: "none",
          transition: "background 0.15s",
        }}
        onMouseEnter={(e) =>
          (e.currentTarget.style.background = "rgba(120,255,200,0.12)")
        }
        onMouseLeave={(e) =>
          (e.currentTarget.style.background = "rgba(120,255,200,0.06)")
        }
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={faviconUrl}
          alt=""
          width={16}
          height={16}
          style={{ borderRadius: 3, flexShrink: 0 }}
        />
        <div style={{ minWidth: 0, flex: 1 }}>
          <div
            style={{
              fontSize: 12,
              color: "rgba(120,255,200,0.9)",
              fontWeight: 600,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {displayTitle}
          </div>
          <div
            style={{
              fontSize: 10,
              color: "rgba(255,255,255,0.35)",
              fontFamily: "'DM Mono', monospace",
            }}
          >
            {hostname}
          </div>
        </div>
        <svg
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="none"
          style={{ flexShrink: 0 }}
        >
          <path
            d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"
            stroke="rgba(120,255,200,0.5)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </a>
    );
  }

  // Light variant
  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-2.5 px-3 py-2 bg-slate-50 hover:bg-teal-50 border border-slate-200/60 hover:border-teal-200/60 rounded-xl transition-all duration-200 no-underline group"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={faviconUrl}
        alt=""
        width={16}
        height={16}
        className="rounded-sm shrink-0"
      />
      <div className="min-w-0 flex-1">
        <div className="text-[12px] font-semibold text-slate-700 group-hover:text-teal-700 truncate">
          {displayTitle}
        </div>
        <div className="text-[10px] text-slate-400 font-mono truncate">
          {hostname}
        </div>
      </div>
      <svg
        className="w-3 h-3 text-slate-300 group-hover:text-teal-500 shrink-0 transition-colors"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3" />
      </svg>
    </a>
  );
}
