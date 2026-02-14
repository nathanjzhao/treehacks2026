"use client";

import { useState } from "react";
import ThinkingStepper, { StepInfo } from "./ThinkingStepper";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  action?: "FIND_OBJECT" | "ESCALATE" | "ANSWER";
  isLoading?: boolean;
  // Stepper fields
  steps?: StepInfo[];
  stepsFinished?: boolean;
}

interface ChatBubbleProps {
  message: ChatMessage;
}

export default function ChatBubble({ message }: ChatBubbleProps) {
  const { role, content, isLoading, action, steps, stepsFinished } = message;
  const [stepperExpanded, setStepperExpanded] = useState(true);

  if (isLoading) {
    return (
      <div className="flex justify-start chat-bubble-enter py-1.5">
        <div className="flex gap-3 max-w-[85%]">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-100 to-emerald-50 flex items-center justify-center shrink-0 mt-0.5 shadow-sm border border-teal-200/50">
            <svg
              className="w-4.5 h-4.5 text-teal-600"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M12 2a3 3 0 0 0-3 3v1a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 10H5a2 2 0 0 0-2 2v1a8 8 0 0 0 8 8h2a8 8 0 0 0 8-8v-1a2 2 0 0 0-2-2Z" />
            </svg>
          </div>
          <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-md px-5 py-3.5 shadow-sm">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-teal-400 rounded-full typing-dot" />
              <span className="w-2 h-2 bg-teal-400 rounded-full typing-dot" />
              <span className="w-2 h-2 bg-teal-400 rounded-full typing-dot" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (role === "system") {
    return (
      <div className="flex justify-center chat-bubble-enter py-3">
        <div className="bg-gradient-to-r from-slate-100/80 to-slate-50/80 text-slate-500 text-xs font-medium rounded-full px-5 py-2 max-w-[85%] text-center border border-slate-200/50 shadow-sm">
          {action === "FIND_OBJECT" && (
            <span className="mr-1.5 text-emerald-500">{"\u{1F4CD}"}</span>
          )}
          {action === "ESCALATE" && (
            <span className="mr-1.5 text-red-500">{"\u{1F6A8}"}</span>
          )}
          {content}
        </div>
      </div>
    );
  }

  if (role === "user") {
    return (
      <div className="flex justify-end chat-bubble-enter py-1.5">
        <div className="max-w-[75%]">
          <div className="bg-gradient-to-br from-teal-600 to-teal-700 text-white rounded-2xl rounded-tr-md px-5 py-3 shadow-md">
            <p className="text-[15px] leading-relaxed">{content}</p>
          </div>
          <div className="text-[10px] text-slate-400 mt-1.5 text-right pr-1 font-medium">
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        </div>
      </div>
    );
  }

  // Assistant - with optional stepper
  return (
    <div className="flex justify-start chat-bubble-enter py-1.5">
      <div className="flex gap-3 max-w-[85%]">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-100 to-emerald-50 flex items-center justify-center shrink-0 mt-0.5 shadow-sm border border-teal-200/50">
          <svg
            className="w-4.5 h-4.5 text-teal-600"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M12 2a3 3 0 0 0-3 3v1a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10H5a2 2 0 0 0-2 2v1a8 8 0 0 0 8 8h2a8 8 0 0 0 8-8v-1a2 2 0 0 0-2-2Z" />
          </svg>
        </div>
        <div className="flex flex-col gap-2 min-w-0">
          {/* Thinking stepper */}
          {steps && steps.length > 0 && (
            <ThinkingStepper
              steps={steps}
              expanded={stepperExpanded}
              onToggle={() => setStepperExpanded((e) => !e)}
              finished={stepsFinished ?? false}
            />
          )}

          {/* Text reply (only if we have content) */}
          {content && (
            <div className="bg-white rounded-2xl rounded-tl-md px-5 py-3 shadow-sm border border-slate-100">
              <p className="text-[15px] leading-relaxed text-slate-700">
                {content}
              </p>
            </div>
          )}

          {/* Timestamp */}
          {content && (
            <div className="text-[10px] text-slate-400 ml-1 font-medium">
              {message.timestamp.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
