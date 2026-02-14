"use client";

export interface StepInfo {
  label: string;
  detail?: string;
  searches?: string[];
  status: "pending" | "active" | "done";
}

interface ThinkingStepperProps {
  steps: StepInfo[];
  expanded: boolean;
  onToggle: () => void;
  finished: boolean;
}

export default function ThinkingStepper({
  steps,
  expanded,
  onToggle,
  finished,
}: ThinkingStepperProps) {
  const doneCount = steps.filter((s) => s.status === "done").length;
  const label = finished
    ? `${steps.length} steps completed`
    : `${doneCount} of ${steps.length} steps\u2026`;

  return (
    <div className="bg-teal-50/60 border border-teal-200/40 rounded-2xl overflow-hidden transition-all duration-300">
      {/* Collapsed header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2.5 px-4 py-3 bg-transparent border-none cursor-pointer text-left group"
      >
        {/* Spinner or checkmark */}
        {finished ? (
          <div className="w-5 h-5 rounded-full bg-teal-600 flex items-center justify-center shrink-0">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none">
              <path
                d="M5 13l4 4L19 7"
                stroke="white"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        ) : (
          <div className="w-5 h-5 rounded-full border-[2.5px] border-transparent border-t-teal-600 border-r-teal-600 animate-spin shrink-0" />
        )}

        {/* Label */}
        <span
          className={`text-[13px] font-semibold flex-1 ${
            finished ? "text-teal-700" : "text-teal-600"
          }`}
        >
          {label}
        </span>

        {/* Chevron */}
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          className={`transition-transform duration-300 ${
            expanded ? "rotate-180" : "rotate-0"
          }`}
        >
          <path
            d="M6 9l6 6 6-6"
            stroke="#94a3b8"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {/* Expandable step list */}
      <div
        className="transition-all duration-350 ease-[cubic-bezier(0.4,0,0.2,1)] overflow-hidden"
        style={{
          maxHeight: expanded ? 600 : 0,
          opacity: expanded ? 1 : 0,
        }}
      >
        <div className="px-4 pb-3 flex flex-col">
          {steps.map((step, i) => {
            const done = step.status === "done";
            const active = step.status === "active";
            const pending = step.status === "pending";

            return (
              <div key={i}>
                <div
                  className={`flex items-start gap-2.5 py-2 transition-opacity duration-300 ${
                    pending ? "opacity-30" : "opacity-100"
                  }`}
                >
                  {/* Status indicator */}
                  <div className="w-5 flex justify-center shrink-0 pt-0.5">
                    {done ? (
                      <div className="w-[18px] h-[18px] rounded-full bg-teal-100 flex items-center justify-center">
                        <svg
                          width="10"
                          height="10"
                          viewBox="0 0 24 24"
                          fill="none"
                        >
                          <path
                            d="M5 13l4 4L19 7"
                            stroke="#0d9488"
                            strokeWidth="3"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </div>
                    ) : active ? (
                      <div className="w-[18px] h-[18px] rounded-full border-2 border-teal-500 flex items-center justify-center">
                        <div className="w-1.5 h-1.5 rounded-full bg-teal-500 stepper-pulse" />
                      </div>
                    ) : (
                      <div className="w-[18px] h-[18px] rounded-full border-2 border-slate-200" />
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <span
                      className={`text-[13px] ${
                        active
                          ? "font-semibold text-slate-800"
                          : done
                          ? "font-medium text-teal-700"
                          : "font-medium text-slate-400"
                      }`}
                    >
                      {step.label}
                    </span>

                    {/* Detail text */}
                    {step.detail && (done || active) && (
                      <div className="text-[12px] text-slate-500 mt-0.5 font-mono truncate">
                        {step.detail}
                      </div>
                    )}

                    {/* Search query pills */}
                    {step.searches && (done || active) && (
                      <div className="flex flex-wrap gap-1.5 mt-1.5">
                        {step.searches.map((s, j) => (
                          <span
                            key={j}
                            className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium border ${
                              done
                                ? "bg-teal-50 border-teal-200/60 text-teal-700"
                                : "bg-slate-50 border-slate-200/60 text-slate-600"
                            }`}
                          >
                            <svg
                              width="10"
                              height="10"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2.5"
                            >
                              <circle cx="11" cy="11" r="8" />
                              <line x1="21" y1="21" x2="16.65" y2="16.65" />
                            </svg>
                            {s}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Connector line */}
                {i < steps.length - 1 && (
                  <div
                    className={`w-0.5 h-2 ml-[9px] rounded-full transition-colors duration-300 ${
                      done ? "bg-teal-200" : "bg-slate-200/60"
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
