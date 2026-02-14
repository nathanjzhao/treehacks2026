import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 gap-14 bg-[#f4f6f5] relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-gradient-to-bl from-teal-100/40 to-transparent rounded-full blur-3xl" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-gradient-to-tr from-emerald-100/30 to-transparent rounded-full blur-3xl" />

      {/* Logo + tagline */}
      <div className="text-center relative z-10">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-teal-600 to-emerald-500 mb-8 shadow-xl shadow-teal-500/20">
          <svg
            width="40"
            height="40"
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 2a3 3 0 0 0-3 3v1a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10H5a2 2 0 0 0-2 2v1a8 8 0 0 0 8 8h2a8 8 0 0 0 8-8v-1a2 2 0 0 0-2-2Z" />
            <path d="M12 18v4" />
            <path d="M8 22h8" />
          </svg>
        </div>
        <h1 className="text-5xl font-bold tracking-tight text-foreground font-display">
          Mira
        </h1>
        <p className="text-lg mt-4 text-muted max-w-md mx-auto leading-relaxed">
          Intelligent care platform for assisted living facilities
        </p>
      </div>

      {/* Navigation cards */}
      <div className="flex flex-col sm:flex-row gap-5 w-full max-w-lg relative z-10">
        <Link
          href="/device"
          className="flex-1 group relative overflow-hidden rounded-2xl p-8 text-center text-white transition-all duration-300 hover:scale-[1.02] hover:shadow-xl bg-gradient-to-br from-teal-600 to-teal-700 shadow-lg shadow-teal-600/20"
        >
          <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent" />
          <div className="absolute -top-6 -right-6 w-24 h-24 rounded-full bg-white/5" />
          <div className="relative z-10">
            <svg
              className="mx-auto mb-4 w-10 h-10"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <div className="text-xl font-bold">Resident Chat</div>
            <div className="text-sm mt-1 text-white/70 font-medium">
              Voice & text assistant
            </div>
          </div>
        </Link>

        <Link
          href="/dashboard"
          className="flex-1 group relative overflow-hidden rounded-2xl p-8 text-center text-white transition-all duration-300 hover:scale-[1.02] hover:shadow-xl bg-gradient-to-br from-slate-700 to-slate-800 shadow-lg shadow-slate-700/20"
        >
          <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent" />
          <div className="absolute -top-6 -right-6 w-24 h-24 rounded-full bg-white/5" />
          <div className="relative z-10">
            <svg
              className="mx-auto mb-4 w-10 h-10"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="7" height="9" rx="1" />
              <rect x="14" y="3" width="7" height="5" rx="1" />
              <rect x="14" y="12" width="7" height="9" rx="1" />
              <rect x="3" y="16" width="7" height="5" rx="1" />
            </svg>
            <div className="text-xl font-bold">Supervisor Dashboard</div>
            <div className="text-sm mt-1 text-white/70 font-medium">
              Patient management
            </div>
          </div>
        </Link>
      </div>

      <p className="text-xs text-muted-foreground font-medium relative z-10">
        Built for TreeHacks 2026
      </p>
    </div>
  );
}
