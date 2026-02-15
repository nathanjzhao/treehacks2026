"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { supabase, MiraEvent } from "@/lib/supabase/client";
import { useDetection, DetectionOverlay } from "@/yolo";
import { Streamdown } from "streamdown";
import CitationCard from "@/app/components/CitationCard";

/* ================================================================
   ACTION CHAINS — scripted demos for hackathon presentation
   In production these would come from the real /api/chat SSE stream.
   ================================================================ */

interface Step {
  label: string;
  detail?: string;
  searches?: string[];
  icon: string;
  duration: number;
}

interface Chain {
  steps: Step[];
  reply: string;
  type: string;
  location?: string;
  confidence?: number;
  coords?: { x: number; y: number; z: number };
}

const ACTION_CHAINS: Record<string, Chain> = {
  pills: {
    steps: [
      { label: "Interpreting speech", detail: "ASR confidence: 0.96 \u2014 intent: find_object", icon: "brain", duration: 600 },
      { label: "Checking patient context", searches: ["Margaret Chen medication schedule", "pill organizer last seen"], icon: "search", duration: 900 },
      { label: "Dispatching to CV pipeline", detail: "POST /api/objects/request \u2192 pill organizer", icon: "api", duration: 400 },
      { label: "FoundationPose scanning", detail: "3 camera feeds \u00b7 patio, garden, porch", icon: "camera", duration: 1800 },
      { label: "Object located", detail: "Garden table, near water glass \u2014 94% confidence", icon: "pin", duration: 300 },
      { label: "Generating audio response", detail: "Including medication reminder", icon: "sparkle", duration: 500 },
    ],
    reply: "I found your pill organizer, Margaret. It\u2019s on the garden table, near your water glass. Looks like you haven\u2019t taken your afternoon Metformin yet.",
    type: "object_found",
    location: "Garden table \u00b7 near water glass",
    confidence: 94,
    coords: { x: -1.3, y: 0.9, z: 0.5 },
  },
  glasses: {
    steps: [
      { label: "Interpreting speech", detail: "ASR confidence: 0.93 \u2014 intent: find_object", icon: "brain", duration: 500 },
      { label: "Querying location history", searches: ["reading glasses location log", "patio scan cache"], icon: "search", duration: 800 },
      { label: "Dispatching to CV pipeline", detail: "POST /api/objects/request \u2192 reading glasses", icon: "api", duration: 350 },
      { label: "FoundationPose scanning", detail: "Patio, garden bench cameras", icon: "camera", duration: 1500 },
      { label: "Object located", detail: "Porch bench, left cushion \u2014 88%", icon: "pin", duration: 300 },
      { label: "Generating audio response", icon: "sparkle", duration: 400 },
    ],
    reply: "Found them \u2014 your reading glasses are on the porch bench, left side on the cushion.",
    type: "object_found",
    location: "Porch bench \u00b7 left cushion",
    confidence: 88,
    coords: { x: 2.1, y: 0.8, z: 1.2 },
  },
  walker: {
    steps: [
      { label: "Interpreting speech", detail: "ASR confidence: 0.97 \u2014 intent: find_object", icon: "brain", duration: 500 },
      { label: "Checking recent events", searches: ["walker last seen", "entryway objects"], icon: "search", duration: 700 },
      { label: "Dispatching to CV pipeline", detail: "POST /api/objects/request \u2192 walker", icon: "api", duration: 300 },
      { label: "FoundationPose scanning", detail: "Front porch, pathway cameras", icon: "camera", duration: 1200 },
      { label: "Object located", detail: "Front porch \u2014 97% confidence", icon: "pin", duration: 300 },
      { label: "Generating audio response", icon: "sparkle", duration: 400 },
    ],
    reply: "Your walker is on the front porch, Margaret. Right where you left it after your morning walk.",
    type: "object_found",
    location: "Front porch \u00b7 by railing",
    confidence: 97,
    coords: { x: 0.2, y: 0.0, z: 3.1 },
  },
  emergency: {
    steps: [
      { label: "Analyzing urgency", detail: "CRITICAL \u2014 fall risk / safety event", icon: "brain", duration: 300 },
      { label: "Loading emergency protocol", searches: ["Margaret Chen emergency contacts", "escalation SOP"], icon: "search", duration: 500 },
      { label: "Twilio escalation", detail: "SMS \u2192 David Chen (+1 555-123-4567)", icon: "phone", duration: 800 },
      { label: "Escalation confirmed", detail: "Delivered \u2014 SID: SM8a2f\u2026c91e", icon: "check", duration: 300 },
      { label: "Generating audio response", detail: "Including 911 advisory", icon: "sparkle", duration: 400 },
    ],
    reply: "I\u2019ve texted your son David, Margaret. Stay calm \u2014 help is on the way. If you\u2019re hurt, I can call 911.",
    type: "escalation",
  },
  meds: {
    steps: [
      { label: "Interpreting speech", detail: "intent: medication_query", icon: "brain", duration: 500 },
      { label: "Loading patient record", searches: ["Margaret Chen medications", "dosage schedule"], icon: "search", duration: 1000 },
      { label: "Safety guardrails", detail: "No diagnosis \u2014 informational only", icon: "shield", duration: 400 },
      { label: "Generating audio response", icon: "sparkle", duration: 500 },
    ],
    reply: "You take three medications: Metformin 500mg twice daily for diabetes, Lisinopril 10mg once daily for blood pressure, and Vitamin D3 for bone health.",
    type: "info",
  },
  greeting: {
    steps: [
      { label: "Processing speech", icon: "brain", duration: 400 },
      { label: "Loading patient context", searches: ["Margaret Chen profile"], icon: "search", duration: 500 },
      { label: "Generating response", icon: "sparkle", duration: 300 },
    ],
    reply: "Good morning, Margaret! Beautiful day out here. How can I help you?",
    type: "greeting",
  },
  default: {
    steps: [
      { label: "Interpreting speech", detail: "Analyzing intent\u2026", icon: "brain", duration: 600 },
      { label: "Searching knowledge base", searches: ["patient history", "general health KB"], icon: "search", duration: 900 },
      { label: "Generating response", icon: "sparkle", duration: 500 },
    ],
    reply: "Let me look into that for you. I\u2019ll check with your care team if needed.",
    type: "thinking",
  },
};

function getChain(text: string): Chain {
  const l = text.toLowerCase();
  if (l.includes("pill") || l.includes("metformin")) return ACTION_CHAINS.pills;
  if (l.includes("glass")) return ACTION_CHAINS.glasses;
  if (l.includes("walker")) return ACTION_CHAINS.walker;
  if (l.includes("emergency") || l.includes("help") || l.includes("fall")) return ACTION_CHAINS.emergency;
  if (l.includes("medication") || l.includes("what do i take") || l.includes("meds")) return ACTION_CHAINS.meds;
  if (l.includes("hello") || l.includes("hi") || l.includes("morning")) return ACTION_CHAINS.greeting;
  return ACTION_CHAINS.default;
}

/* ================================================================
   Step icon SVGs
   ================================================================ */

function StepIcon({ type, done, active }: { type: string; done: boolean; active: boolean }) {
  const c = done ? "rgba(120,255,200,0.9)" : active ? "rgba(120,255,200,0.9)" : "rgba(255,255,255,0.25)";
  const icons: Record<string, React.ReactNode> = {
    brain: <path d="M12 2a7 7 0 017 7c0 2.5-1.3 4.7-3.2 6H8.2C6.3 13.7 5 11.5 5 9a7 7 0 017-7zM9 22v-2h6v2M9 18h6" stroke={c} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none" />,
    search: <><circle cx="11" cy="11" r="7" stroke={c} strokeWidth="1.8" fill="none" /><path d="M21 21l-4.35-4.35" stroke={c} strokeWidth="1.8" strokeLinecap="round" /></>,
    api: <><rect x="3" y="3" width="18" height="18" rx="3" stroke={c} strokeWidth="1.8" fill="none" /><path d="M8 12h8M12 8v8" stroke={c} strokeWidth="1.8" strokeLinecap="round" /></>,
    camera: <><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" stroke={c} strokeWidth="1.8" fill="none" /><circle cx="12" cy="13" r="4" stroke={c} strokeWidth="1.8" fill="none" /></>,
    pin: <><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" stroke={c} strokeWidth="1.8" fill="none" /><circle cx="12" cy="9" r="2.5" fill={c} /></>,
    sparkle: <path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z" stroke={c} strokeWidth="1.8" fill="none" strokeLinejoin="round" />,
    phone: <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6A19.79 19.79 0 012.12 4.18 2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z" stroke={c} strokeWidth="1.8" fill="none" />,
    shield: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" stroke={c} strokeWidth="1.8" fill="none" /><path d="M9 12l2 2 4-4" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" /></>,
    check: <><circle cx="12" cy="12" r="10" stroke={c} strokeWidth="1.8" fill="none" /><path d="M9 12l2 2 4-4" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></>,
    globe: <><circle cx="12" cy="12" r="10" stroke={c} strokeWidth="1.8" fill="none" /><path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z" stroke={c} strokeWidth="1.8" fill="none" /></>,
  };
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      {icons[type] || icons.brain}
    </svg>
  );
}

/* ================================================================
   HUD Thinking Stepper (dark glass variant)
   ================================================================ */

interface Processing {
  id: number;
  steps: Step[];
  currentStep: number;
  chain: Chain;
  finished: boolean;
}

function HudStepper({
  steps,
  currentStep,
  expanded,
  onToggle,
  finished,
}: {
  steps: Step[];
  currentStep: number;
  expanded: boolean;
  onToggle: () => void;
  finished: boolean;
}) {
  const n = finished ? steps.length : currentStep;
  const label = finished ? `${steps.length} steps completed` : `${n} of ${steps.length} steps\u2026`;

  return (
    <div
      className="hud-fadein"
      style={{
        background: "rgba(0,0,0,0.6)",
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        border: "1px solid rgba(120,255,200,0.15)",
        borderRadius: 14,
        overflow: "hidden",
      }}
    >
      <button
        onClick={onToggle}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "10px 14px",
          background: "none",
          border: "none",
          cursor: "pointer",
          fontFamily: "inherit",
        }}
      >
        {finished ? (
          <div
            style={{
              width: 18,
              height: 18,
              borderRadius: "50%",
              background: "rgba(120,255,200,0.2)",
              border: "1.5px solid rgba(120,255,200,0.6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none">
              <path d="M5 13l4 4L19 7" stroke="rgba(120,255,200,0.9)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        ) : (
          <div className="hud-spin" style={{ width: 18, height: 18, borderRadius: "50%", border: "2px solid transparent", borderTopColor: "rgba(120,255,200,0.8)", borderRightColor: "rgba(120,255,200,0.8)", flexShrink: 0 }} />
        )}
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: finished ? "rgba(120,255,200,0.9)" : "rgba(120,255,200,0.7)",
            flex: 1,
            textAlign: "left",
            fontFamily: "'DM Mono', 'Courier New', monospace",
          }}
        >
          {label}
        </span>
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          style={{
            transform: expanded ? "rotate(180deg)" : "rotate(0)",
            transition: "transform 0.25s ease",
            flexShrink: 0,
          }}
        >
          <path d="M6 9l6 6 6-6" stroke="rgba(255,255,255,0.4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      <div
        style={{
          maxHeight: expanded ? 600 : 0,
          opacity: expanded ? 1 : 0,
          overflow: "hidden",
          transition: "max-height 0.35s cubic-bezier(0.4,0,0.2,1), opacity 0.25s ease",
        }}
      >
        <div style={{ padding: "0 14px 10px" }}>
          {steps.map((step, i) => {
            const done = finished || i < currentStep;
            const active = !finished && i === currentStep;
            const pending = !finished && i > currentStep;
            return (
              <div key={i}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 8,
                    padding: "6px 0",
                    opacity: pending ? 0.25 : 1,
                    transition: "opacity 0.3s ease",
                  }}
                >
                  <div style={{ width: 16, flexShrink: 0, display: "flex", justifyContent: "center", paddingTop: 2 }}>
                    {done ? (
                      <div style={{ width: 14, height: 14, borderRadius: "50%", background: "rgba(120,255,200,0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <svg width="8" height="8" viewBox="0 0 24 24" fill="none"><path d="M5 13l4 4L19 7" stroke="rgba(120,255,200,0.8)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                      </div>
                    ) : active ? (
                      <div style={{ width: 14, height: 14, borderRadius: "50%", border: "1.5px solid rgba(120,255,200,0.6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <div className="hud-pulse" style={{ width: 5, height: 5, borderRadius: "50%", background: "rgba(120,255,200,0.8)" }} />
                      </div>
                    ) : (
                      <div style={{ width: 14, height: 14, borderRadius: "50%", border: "1.5px solid rgba(255,255,255,0.15)" }} />
                    )}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <StepIcon type={step.icon} done={done} active={active} />
                      <span
                        style={{
                          fontSize: 13,
                          fontWeight: active ? 600 : 400,
                          color: done ? "rgba(120,255,200,0.9)" : active ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.25)",
                          textShadow: "0 1px 2px rgba(0,0,0,0.4)",
                        }}
                      >
                        {step.label}
                      </span>
                    </div>
                    {step.detail && (done || active) && (
                      <div className={active ? "hud-fadein" : ""} style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginTop: 2, marginLeft: 20, fontFamily: "'DM Mono', monospace" }}>
                        {step.detail}
                      </div>
                    )}
                    {step.searches && (done || active) && (
                      <div className={active ? "hud-fadein" : ""} style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4, marginLeft: 20 }}>
                        {step.searches.map((s, j) => (
                          <span
                            key={j}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 4,
                              padding: "2px 10px",
                              background: done ? "rgba(120,255,200,0.08)" : "rgba(255,255,255,0.05)",
                              border: `1px solid ${done ? "rgba(120,255,200,0.15)" : "rgba(255,255,255,0.08)"}`,
                              borderRadius: 12,
                              fontSize: 11,
                              color: done ? "rgba(120,255,200,0.7)" : "rgba(255,255,255,0.4)",
                              fontFamily: "'DM Mono', monospace",
                            }}
                          >
                            <svg width="9" height="9" viewBox="0 0 24 24" fill="none">
                              <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
                              <path d="M21 21l-4.35-4.35" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                            </svg>
                            {s}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                {i < steps.length - 1 && (
                  <div style={{ width: 1.5, height: 6, background: done ? "rgba(120,255,200,0.15)" : "rgba(255,255,255,0.06)", marginLeft: 7, borderRadius: 1 }} />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   Location tag
   ================================================================ */

function LocationTag({ location, confidence, coords }: { location: string; confidence: number; coords: { x: number; y: number; z: number } }) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 14px", background: "rgba(0,0,0,0.5)", backdropFilter: "blur(12px)", border: "1px solid rgba(120,255,200,0.25)", borderRadius: 10, marginTop: 8 }}>
      <svg width="14" height="14" fill="none" viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5A2.5 2.5 0 1112 7a2.5 2.5 0 010 5z" fill="rgba(120,255,200,0.8)" /></svg>
      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: "rgba(120,255,200,0.9)" }}>{location}</div>
        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)", fontFamily: "'DM Mono', monospace" }}>
          {confidence}% \u00b7 ({coords.x.toFixed(1)}, {coords.y.toFixed(1)}, {coords.z.toFixed(1)})
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   Escalation toast
   ================================================================ */

function EscalationToast({ visible }: { visible: boolean }) {
  return (
    <div
      style={{
        position: "absolute",
        top: visible ? 80 : -60,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 50,
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 20px",
        background: "rgba(200,80,30,0.85)",
        backdropFilter: "blur(12px)",
        borderRadius: 12,
        border: "1px solid rgba(255,150,100,0.3)",
        transition: "top 0.4s cubic-bezier(0.34,1.56,0.64,1)",
        boxShadow: "0 8px 32px rgba(200,80,30,0.4)",
      }}
    >
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24">
        <path d="M9 12l2 2 4-4" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="12" cy="12" r="10" stroke="white" strokeWidth="2" />
      </svg>
      <span style={{ color: "white", fontSize: 13, fontWeight: 600 }}>
        Caregiver notified via SMS
      </span>
    </div>
  );
}

/* ================================================================
   Activity Feed — always-visible tool call log on left side of HUD
   ================================================================ */

function ActivityFeed({
  steps,
  currentStep,
  finished,
}: {
  steps: Step[];
  currentStep: number;
  finished: boolean;
}) {
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: "smooth" });
  }, [currentStep, finished]);

  if (steps.length === 0) return null;

  return (
    <div
      className="hud-fadein"
      style={{
        background: "rgba(0,0,0,0.55)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        border: "1px solid rgba(120,255,200,0.12)",
        borderRadius: 10,
        overflow: "hidden",
        maxHeight: 280,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "6px 12px",
          borderBottom: "1px solid rgba(120,255,200,0.08)",
        }}
      >
        <div
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: finished ? "rgba(120,255,200,0.8)" : "rgba(255,200,60,0.8)",
            boxShadow: finished ? "none" : "0 0 6px rgba(255,200,60,0.4)",
            animation: finished ? "none" : "hud-pulse-kf 1.5s ease-in-out infinite",
          }}
        />
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            color: "rgba(120,255,200,0.6)",
            fontFamily: "'DM Mono', monospace",
            letterSpacing: "0.08em",
          }}
        >
          ACTIVITY
        </span>
        <span
          style={{
            fontSize: 10,
            color: "rgba(255,255,255,0.3)",
            fontFamily: "'DM Mono', monospace",
            marginLeft: "auto",
          }}
        >
          {finished ? `${steps.length}/${steps.length}` : `${currentStep}/${steps.length}`}
        </span>
      </div>

      {/* Feed lines */}
      <div ref={feedRef} style={{ overflowY: "auto", maxHeight: 244, padding: "4px 0" }}>
        {steps.map((step, i) => {
          const done = finished || i < currentStep;
          const active = !finished && i === currentStep;
          const pending = !finished && i > currentStep;

          return (
            <div
              key={i}
              className={active ? "hud-fadein" : ""}
              style={{
                display: "flex",
                gap: 8,
                padding: "3px 12px",
                opacity: pending ? 0.2 : 1,
                transition: "opacity 0.3s ease",
              }}
            >
              {/* Status indicator */}
              <div style={{ width: 16, flexShrink: 0, display: "flex", justifyContent: "center", paddingTop: 3 }}>
                {done ? (
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none">
                    <path d="M5 13l4 4L19 7" stroke="rgba(120,255,200,0.7)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : active ? (
                  <span className="hud-pulse" style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "rgba(255,200,60,0.8)" }} />
                ) : (
                  <span style={{ display: "inline-block", width: 4, height: 4, borderRadius: "50%", background: "rgba(255,255,255,0.15)" }} />
                )}
              </div>

              {/* Icon */}
              <div style={{ paddingTop: 2, flexShrink: 0 }}>
                <StepIcon type={step.icon} done={done} active={active} />
              </div>

              {/* Label + detail + searches */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <span
                  style={{
                    fontSize: 12,
                    fontFamily: "'DM Mono', monospace",
                    fontWeight: active ? 600 : 400,
                    color: done ? "rgba(120,255,200,0.8)" : active ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.2)",
                    display: "block",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {step.label}
                </span>
                {step.detail && (done || active) && (
                  <span
                    style={{
                      fontSize: 10,
                      fontFamily: "'DM Mono', monospace",
                      color: "rgba(255,255,255,0.35)",
                      display: "block",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      marginTop: 1,
                    }}
                  >
                    {step.detail}
                  </span>
                )}
                {step.searches && step.searches.length > 0 && (done || active) && (
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 3 }}>
                    {step.searches.map((s, si) => (
                      <span
                        key={si}
                        style={{
                          fontSize: 9,
                          fontFamily: "'DM Mono', monospace",
                          padding: "1px 6px",
                          borderRadius: 6,
                          background: "rgba(120,255,200,0.08)",
                          border: "1px solid rgba(120,255,200,0.15)",
                          color: "rgba(120,255,200,0.6)",
                        }}
                      >
                        {s}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ================================================================
   Audio Waveform — canvas-based frequency visualization
   ================================================================ */

function AudioWaveform({ active }: { active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animRef = useRef<number>(0);
  const audioCtxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (!active) {
      cancelAnimationFrame(animRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      analyserRef.current = null;
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    let cancelled = false;

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;

        const audioCtx = new AudioContext();
        audioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        analyser.smoothingTimeConstant = 0.8;
        source.connect(analyser);
        analyserRef.current = analyser;

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext("2d")!;

        function draw() {
          if (cancelled) return;
          animRef.current = requestAnimationFrame(draw);
          analyser.getByteFrequencyData(dataArray);

          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const barCount = dataArray.length;
          const barWidth = canvas.width / barCount;
          const centerY = canvas.height / 2;

          for (let i = 0; i < barCount; i++) {
            const v = dataArray[i] / 255;
            const barHeight = v * centerY * 0.9;
            const x = i * barWidth;
            const alpha = 0.3 + v * 0.6;
            ctx.fillStyle = `rgba(120, 255, 200, ${alpha})`;
            ctx.fillRect(x + 1, centerY - barHeight, barWidth - 2, barHeight);
            ctx.fillRect(x + 1, centerY, barWidth - 2, barHeight);
          }
        }
        draw();
      } catch (err) {
        console.error("Audio waveform error:", err);
      }
    }

    start();
    return () => {
      cancelled = true;
      cancelAnimationFrame(animRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
    };
  }, [active]);

  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={48}
      style={{
        opacity: active ? 1 : 0,
        transition: "opacity 0.3s ease",
        pointerEvents: "none",
      }}
    />
  );
}

/* ================================================================
   Message types
   ================================================================ */

interface MsgBase {
  time: Date;
}
interface UserMsg extends MsgBase {
  role: "user";
  text: string;
}
interface ThinkingMsg extends MsgBase {
  role: "thinking";
  steps: Step[];
  id: number;
}
interface AssistantMsg extends MsgBase {
  role: "assistant";
  text: string;
  type: string;
  location?: string;
  confidence?: number;
  coords?: { x: number; y: number; z: number };
  citations?: Array<{ title?: string; url: string; evidence_grade?: string }>;
}
type Msg = UserMsg | ThinkingMsg | AssistantMsg;

/* ================================================================
   Main: AR Glasses Stream Overlay
   ================================================================ */

const DEMO_PATIENT_ID = "a1b2c3d4-0001-4000-8000-000000000001";

function mapStepToIcon(label: string): string {
  const l = label.toLowerCase();
  if (l.includes("interpret")) return "brain";
  if (l.includes("searching medical") || l.includes("querying perplexity") || l.includes("sonar")) return "globe";
  if (l.includes("checking clinical") || l.includes("clinical guidelines")) return "shield";
  if (l.includes("found") && l.includes("source")) return "check";
  if (l.includes("health record") || l.includes("recalling") || l.includes("looking up")) return "search";
  if (l.includes("thinking") || l.includes("analyzing")) return "brain";
  if (l.includes("searching for")) return "camera";
  if (l.includes("alert") || l.includes("caregiver")) return "phone";
  if (l.includes("composing") || l.includes("generating")) return "sparkle";
  if (l.includes("researching")) return "globe";
  return "brain";
}

export default function StreamPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [processing, setProcessing] = useState<Processing | null>(null);
  const [expandedSteppers, setExpandedSteppers] = useState<Record<number, boolean>>({});
  const [listening, setListening] = useState(false);
  const [showEscalation, setShowEscalation] = useState(false);
  const [input, setInput] = useState("");
  const [clock, setClock] = useState<Date | null>(null);
  const [liveMode, setLiveMode] = useState(true);
  const [videoReady, setVideoReady] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLCanvasElement>(null);
  const pidRef = useRef(0);
  const [yoloEnabled, setYoloEnabled] = useState(false); // Disabled for canvas rendering
  const [viewSize, setViewSize] = useState({ w: 1920, h: 1080 });

  // YOLO object detection on canvas (disabled for now - needs adaptation for canvas)
  const { detections, inferenceMs, modelLoaded, modelError } = useDetection(
    videoRef as any, // Cast to work with canvas
    {
      enabled: false, // Disabled until canvas support is added
    }
  );

  useEffect(() => {
    const update = () => setViewSize({ w: window.innerWidth, h: window.innerHeight });
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  useEffect(() => {
    setClock(new Date());
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Ray-Ban Android stream via Server-Sent Events
  useEffect(() => {
    let cancelled = false;
    let eventSource: EventSource | null = null;
    let retryTimeout: NodeJS.Timeout;
    const canvasRef = videoRef as React.RefObject<HTMLCanvasElement>;

    function startStreamConnection() {
      if (cancelled) return;

      console.log("[Stream] Connecting to SSE stream...");

      try {
        eventSource = new EventSource("/api/stream/ws");

        eventSource.onopen = () => {
          if (!cancelled) {
            console.log("[Stream] SSE connection established");
            setVideoError(null);
          }
        };

        eventSource.onmessage = (event) => {
          if (cancelled) return;

          try {
            const data = JSON.parse(event.data);

            if (data.type === "connected") {
              console.log("[Stream] Connected:", data.clientId);
              setVideoReady(true);
            } else if (data.type === "frame") {
              // Render frame to canvas
              renderFrame(data);
            }
          } catch (e) {
            console.error("[Stream] Error parsing message:", e);
          }
        };

        eventSource.onerror = (err) => {
          if (cancelled) return;

          console.error("[Stream] SSE error:", err);
          setVideoError("Stream disconnected - retrying...");
          setVideoReady(false);

          eventSource?.close();

          // Retry after 2 seconds
          retryTimeout = setTimeout(() => {
            if (!cancelled) {
              console.log("[Stream] Retrying connection...");
              startStreamConnection();
            }
          }, 2000);
        };
      } catch (e) {
        console.error("[Stream] Failed to create EventSource:", e);
        setVideoError("Failed to connect to stream");
      }
    }

    function renderFrame(frameData: any) {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Decode base64 JPEG
      const img = new Image();
      img.onload = () => {
        canvas.width = frameData.width;
        canvas.height = frameData.height;
        ctx.drawImage(img, 0, 0);
      };
      img.onerror = (err) => {
        console.error("[Stream] Error loading frame image:", err);
      };
      img.src = `data:image/jpeg;base64,${frameData.jpeg_base64}`;
    }

    startStreamConnection();

    return () => {
      cancelled = true;
      clearTimeout(retryTimeout);
      eventSource?.close();
    };
  }, [videoRef]);

  // Supabase realtime — mirror device chat in live mode
  useEffect(() => {
    if (!liveMode) return;

    // Load recent chat events on entering live mode
    (async () => {
      const { data } = await supabase
        .from("events")
        .select("*")
        .eq("patient_id", DEMO_PATIENT_ID)
        .in("type", ["CHAT_USER_UTTERANCE", "CHAT_ASSISTANT_RESPONSE"])
        .order("created_at", { ascending: false })
        .limit(20);

      if (data && data.length > 0) {
        const msgs: Msg[] = (data as MiraEvent[]).reverse().map((ev) => {
          if (ev.type === "CHAT_USER_UTTERANCE") {
            return { role: "user" as const, text: ev.receipt_text || "...", time: new Date(ev.created_at) };
          }
          const payload = ev.payload as Record<string, unknown> | undefined;
          return {
            role: "assistant" as const,
            text: ev.receipt_text || "...",
            type: payload?.action === "ESCALATE" ? "escalation" : payload?.action === "FIND_OBJECT" ? "object_found" : "info",
            citations: payload?.citations as AssistantMsg["citations"],
            time: new Date(ev.created_at),
          };
        });
        setMessages(msgs);
      }
    })();

    // Subscribe for new events
    const channel = supabase
      .channel("stream-mirror")
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "events",
          filter: `patient_id=eq.${DEMO_PATIENT_ID}`,
        },
        (payload) => {
          const ev = payload.new as MiraEvent;
          if (ev.type === "CHAT_USER_UTTERANCE") {
            setMessages((p) => [...p, { role: "user", text: ev.receipt_text || "...", time: new Date(ev.created_at) }]);
          } else if (ev.type === "CHAT_ASSISTANT_RESPONSE") {
            const pl = ev.payload as Record<string, unknown> | undefined;
            const steps = (pl?.steps as string[]) || [];

            if (steps.length > 0) {
              // Animate through steps in HUD stepper, then show reply
              const mid = ++pidRef.current;
              const hudSteps: Step[] = steps.map((label) => ({
                label,
                icon: mapStepToIcon(label),
                duration: 400,
              }));
              setProcessing({ id: mid, steps: hudSteps, currentStep: 0, chain: { steps: hudSteps, reply: "", type: "info" }, finished: false });
              setExpandedSteppers((p) => ({ ...p, [mid]: true }));

              // Animate through steps with visible pacing
              let si = 0;
              const advance = () => {
                if (si < hudSteps.length) {
                  si++;
                  setTimeout(() => {
                    setProcessing((p) => (p && p.id === mid ? { ...p, currentStep: si } : p));
                    advance();
                  }, 800);
                } else {
                  setProcessing((p) => (p && p.id === mid ? { ...p, finished: true } : p));
                  setTimeout(() => {
                    setMessages((p) => [
                      ...p,
                      { role: "thinking" as const, steps: hudSteps, id: mid, time: new Date() },
                      {
                        role: "assistant" as const,
                        text: ev.receipt_text || "...",
                        type: pl?.action === "ESCALATE" ? "escalation" : pl?.action === "FIND_OBJECT" ? "object_found" : "info",
                        citations: pl?.citations as AssistantMsg["citations"],
                        time: new Date(ev.created_at),
                      },
                    ]);
                    setExpandedSteppers((p) => ({ ...p, [mid]: false }));
                    setProcessing(null);
                  }, 2000);
                }
              };
              advance();
            } else {
              // No steps — just show the reply
              setMessages((p) => [...p, {
                role: "assistant",
                text: ev.receipt_text || "...",
                type: pl?.action === "ESCALATE" ? "escalation" : pl?.action === "FIND_OBJECT" ? "object_found" : "info",
                citations: pl?.citations as AssistantMsg["citations"],
                time: new Date(ev.created_at),
              }]);
            }

            if (pl?.action === "ESCALATE") {
              setShowEscalation(true);
              setTimeout(() => setShowEscalation(false), 4000);
            }
          }
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [liveMode]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, processing]);

  const toggle = useCallback(
    (id: number) => setExpandedSteppers((p) => ({ ...p, [id]: !p[id] })),
    []
  );

  const sendDemo = useCallback((text: string) => {
    const chain = getChain(text);
    const mid = ++pidRef.current;
    setMessages((p) => [...p, { role: "user", text: text.trim(), time: new Date() }]);
    setInput("");
    setProcessing({ id: mid, steps: chain.steps, currentStep: 0, chain, finished: false });
    setExpandedSteppers((p) => ({ ...p, [mid]: true }));

    let si = 0;
    const advance = () => {
      if (si < chain.steps.length) {
        const dur = chain.steps[si].duration;
        si++;
        setTimeout(() => {
          setProcessing((p) => (p && p.id === mid ? { ...p, currentStep: si } : p));
          advance();
        }, dur);
      } else {
        setProcessing((p) => (p && p.id === mid ? { ...p, finished: true } : p));
        setTimeout(() => {
          setMessages((p) => [
            ...p,
            { role: "thinking", steps: chain.steps, id: mid, time: new Date() },
            {
              role: "assistant",
              text: chain.reply,
              type: chain.type,
              location: chain.location,
              confidence: chain.confidence,
              coords: chain.coords,
              time: new Date(),
            },
          ]);
          setExpandedSteppers((p) => ({ ...p, [mid]: false }));
          setProcessing(null);
          if (chain.type === "escalation") {
            setShowEscalation(true);
            setTimeout(() => setShowEscalation(false), 4000);
          }
        }, 600);
      }
    };
    advance();
  }, []);

  const send = useCallback((text: string) => {
    if (!text.trim() || liveMode) return;
    sendDemo(text);
  }, [liveMode, sendDemo]);

  const handleMic = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      alert("Speech not supported. Use Chrome.");
      return;
    }
    const r = new SR();
    r.lang = "en-US";
    r.interimResults = false;
    setListening(true);
    r.start();
    r.onresult = (e: SpeechRecognitionEvent) => {
      setListening(false);
      const t = e.results[0][0].transcript;
      setInput(t);
      send(t);
    };
    r.onerror = () => setListening(false);
    r.onend = () => setListening(false);
  };

  const fmt = (d: Date) => {
    const h = d.getHours();
    const m = d.getMinutes().toString().padStart(2, "0");
    return `${h % 12 || 12}:${m} ${h >= 12 ? "PM" : "AM"}`;
  };

  const fmtClock = (d: Date | null) => {
    if (!d) return "--:--:--";
    return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}:${d.getSeconds().toString().padStart(2, "0")}`;
  };

  // Auto-demo on mount (demo mode only)
  useEffect(() => {
    if (liveMode) return;
    const t1 = setTimeout(() => sendDemo("Good morning, Mira"), 2500);
    const t2 = setTimeout(() => sendDemo("Where are my pills?"), 9000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [liveMode, sendDemo]);

  return (
    <div style={{ width: "100%", height: "100vh", position: "relative", overflow: "hidden", background: "#0a0a0a", fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Mono:wght@400;500&family=Instrument+Serif&display=swap');
        @keyframes hud-fadein{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        .hud-fadein{animation:hud-fadein 0.4s ease-out}
        @keyframes hud-spin-kf{to{transform:rotate(360deg)}}
        .hud-spin{animation:hud-spin-kf 0.8s linear infinite}
        @keyframes hud-pulse-kf{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.4;transform:scale(0.7)}}
        .hud-pulse{animation:hud-pulse-kf 1s ease-in-out infinite}
        @keyframes hud-scanline{0%{top:-100%}100%{top:200%}}
        @keyframes hud-rec{0%,100%{opacity:1}50%{opacity:0.3}}
        @keyframes hud-listening{0%,100%{box-shadow:0 0 0 0 rgba(120,255,200,0.4)}50%{box-shadow:0 0 0 12px rgba(120,255,200,0)}}
        @keyframes hud-grid-pulse{0%,100%{opacity:0.03}50%{opacity:0.06}}
        .hud-text{text-shadow:0 1px 4px rgba(0,0,0,0.7),0 0 12px rgba(0,0,0,0.4)}
        .hud-label{text-shadow:0 1px 3px rgba(0,0,0,0.8)}
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:3px}
      `}</style>

      {/* ──── VIDEO BACKGROUND (Ray-Ban Android stream) ──── */}
      <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
        {/* Live stream from Ray-Ban via Android (SSE + Canvas) */}
        <canvas
          ref={videoRef as React.RefObject<HTMLCanvasElement>}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: videoReady ? "block" : "none",
          }}
        />
        {/* Fallback gradient when camera not available */}
        {!videoReady && (
          <div style={{ position: "absolute", inset: 0, background: "linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)" }}>
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <div style={{ textAlign: "center", color: "rgba(255,255,255,0.3)", fontFamily: "'DM Mono', monospace", fontSize: 12 }}>
                {videoError ? `Camera: ${videoError}` : "Connecting camera..."}
              </div>
            </div>
          </div>
        )}
        {/* Lens vignette */}
        <div style={{ position: "absolute", inset: 0, boxShadow: "inset 0 0 150px 60px rgba(0,0,0,0.4)" }} />
        {/* Scanline */}
        <div style={{ position: "absolute", inset: 0, overflow: "hidden", pointerEvents: "none" }}>
          <div style={{ position: "absolute", left: 0, right: 0, height: "50%", background: "linear-gradient(180deg, transparent 0%, rgba(120,255,200,0.012) 50%, transparent 100%)", animation: "hud-scanline 8s linear infinite" }} />
        </div>
        {/* Grid overlay */}
        <div style={{ position: "absolute", inset: 0, opacity: 0.03, backgroundImage: "linear-gradient(rgba(120,255,200,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(120,255,200,0.3) 1px, transparent 1px)", backgroundSize: "60px 60px", animation: "hud-grid-pulse 4s ease-in-out infinite" }} />
        {/* Film grain */}
        <div style={{ position: "absolute", inset: 0, opacity: 0.08, mixBlendMode: "overlay", background: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")" }} />
        {/* YOLO detection overlay - disabled for canvas mode */}
        {videoReady && videoRef.current && yoloEnabled && (
          <DetectionOverlay
            detections={detections}
            videoWidth={videoRef.current.width || 1920}
            videoHeight={videoRef.current.height || 1080}
            canvasWidth={viewSize.w}
            canvasHeight={viewSize.h}
            inferenceMs={inferenceMs}
          />
        )}
      </div>

      <EscalationToast visible={showEscalation} />

      {/* ──── HUD CORNER BRACKETS ──── */}
      {[
        { top: 20, left: 20 },
        { top: 20, right: 20 },
        { bottom: 20, left: 20 },
        { bottom: 20, right: 20 },
      ].map((pos, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            ...pos,
            width: 24,
            height: 24,
            zIndex: 5,
            pointerEvents: "none",
            borderTop: i < 2 ? "2px solid rgba(120,255,200,0.3)" : "none",
            borderBottom: i >= 2 ? "2px solid rgba(120,255,200,0.3)" : "none",
            borderLeft: i % 2 === 0 ? "2px solid rgba(120,255,200,0.3)" : "none",
            borderRight: i % 2 === 1 ? "2px solid rgba(120,255,200,0.3)" : "none",
          }}
        />
      ))}

      {/* ──── TOP LEFT: Mira branding + LIVE ──── */}
      <div className="hud-text" style={{ position: "absolute", top: 28, left: 56, zIndex: 10, display: "flex", alignItems: "center", gap: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 32, height: 32, borderRadius: 10, background: "rgba(120,255,200,0.1)", border: "1px solid rgba(120,255,200,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4" fill="rgba(120,255,200,0.8)" /><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="rgba(120,255,200,0.8)" strokeWidth="2" strokeLinecap="round" /></svg>
          </div>
          <div>
            <div style={{ fontFamily: "'Instrument Serif', serif", fontSize: 18, color: "rgba(255,255,255,0.9)", lineHeight: 1 }}>Mira</div>
            <div style={{ fontSize: 10, color: "rgba(120,255,200,0.6)", fontFamily: "'DM Mono', monospace", fontWeight: 500 }}>ELDERCARE ASSISTANT</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 12px", background: "rgba(255,60,60,0.15)", border: "1px solid rgba(255,60,60,0.3)", borderRadius: 8 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#FF3C3C", animation: "hud-rec 1.5s ease-in-out infinite" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(255,60,60,0.9)", fontFamily: "'DM Mono', monospace", letterSpacing: "0.08em" }}>REC</span>
        </div>
      </div>

      {/* ──── TOP RIGHT: Clock ──── */}
      <div className="hud-text" style={{ position: "absolute", top: 28, right: 56, zIndex: 10, textAlign: "right" }}>
        <div style={{ fontSize: 20, color: "rgba(255,255,255,0.8)", fontFamily: "'DM Mono', monospace", fontWeight: 500, letterSpacing: "0.05em" }}>{fmtClock(clock)}</div>
        <div style={{ fontSize: 10, color: videoReady ? "rgba(120,255,200,0.4)" : "rgba(255,255,255,0.3)", fontFamily: "'DM Mono', monospace", marginTop: 2 }}>{videoReady ? "CAM LIVE \u00b7 1080p" : "CAM \u00b7 NO SIGNAL"}</div>
      </div>

      {/* ──── TOP CENTER: Patient bar ──── */}
      <div className="hud-text" style={{ position: "absolute", top: 28, left: "50%", transform: "translateX(-50%)", zIndex: 10, display: "flex", alignItems: "center", gap: 16, padding: "8px 20px", background: "rgba(0,0,0,0.55)", backdropFilter: "blur(16px)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.12)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "rgba(120,255,200,0.7)" }} />
          <span style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", fontWeight: 600 }}>Margaret Chen</span>
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.3)" }}>\u00b7 78</span>
        </div>
        <div style={{ width: 1, height: 16, background: "rgba(255,255,255,0.1)" }} />
        <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontFamily: "'DM Mono', monospace" }}>3 meds \u00b7 2 allergies</span>
      </div>

      {/* ──── BOTTOM LEFT: Sensor data + mode toggle ──── */}
      <div className="hud-text" style={{ position: "absolute", bottom: 32, left: 56, zIndex: 10 }}>
        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.25)", fontFamily: "'DM Mono', monospace", lineHeight: 1.8 }}>
          <div>DEPTH: 3.2m · OUTDOOR</div>
          <div>IMU: stable · GPS: 37.4275°N</div>
          <div>BATT: 72% · TEMP: 68°F</div>
        </div>
        <button
          onClick={() => { setLiveMode((p) => !p); setMessages([]); setProcessing(null); }}
          style={{
            marginTop: 10,
            padding: "8px 20px",
            fontSize: 13,
            fontFamily: "'DM Mono', monospace",
            fontWeight: 600,
            letterSpacing: "0.05em",
            color: liveMode ? "rgba(120,255,200,0.9)" : "rgba(255,255,255,0.4)",
            background: liveMode ? "rgba(120,255,200,0.1)" : "rgba(255,255,255,0.05)",
            border: `1px solid ${liveMode ? "rgba(120,255,200,0.3)" : "rgba(255,255,255,0.1)"}`,
            borderRadius: 6,
            cursor: "pointer",
            transition: "all 0.2s",
          }}
        >
          {liveMode ? "● LIVE" : "○ DEMO"}
        </button>
        <button
          onClick={() => setYoloEnabled((p) => !p)}
          style={{
            marginTop: 6,
            padding: "8px 20px",
            fontSize: 13,
            fontFamily: "'DM Mono', monospace",
            fontWeight: 600,
            letterSpacing: "0.05em",
            color: yoloEnabled ? "rgba(120,255,200,0.9)" : "rgba(255,255,255,0.4)",
            background: yoloEnabled ? "rgba(120,255,200,0.1)" : "rgba(255,255,255,0.05)",
            border: `1px solid ${yoloEnabled ? "rgba(120,255,200,0.3)" : "rgba(255,255,255,0.1)"}`,
            borderRadius: 6,
            cursor: "pointer",
            transition: "all 0.2s",
          }}
        >
          {yoloEnabled ? "◉ YOLO" : "○ YOLO"}
        </button>
        {modelLoaded && (
          <div style={{ marginTop: 4, fontSize: 9, fontFamily: "'DM Mono', monospace", color: "rgba(120,255,200,0.4)" }}>
            YOLOv8n · {detections.length} obj · {inferenceMs}ms
          </div>
        )}
        {modelError && (
          <div style={{ marginTop: 4, fontSize: 9, fontFamily: "'DM Mono', monospace", color: "rgba(255,150,100,0.6)" }}>
            YOLO: {modelError.slice(0, 30)}
          </div>
        )}
      </div>

      {/* ──── LEFT: Activity Feed ──── */}
      {processing && (
        <div
          className="hud-fadein"
          style={{
            position: "absolute",
            bottom: 240,
            left: 56,
            width: 320,
            zIndex: 15,
          }}
        >
          <ActivityFeed
            steps={processing.steps}
            currentStep={processing.currentStep}
            finished={processing.finished}
          />
        </div>
      )}

      {/* ──── CHAT OVERLAY ──── */}
      <div style={{ position: "absolute", bottom: 28, right: 28, top: 80, width: 440, zIndex: 20, display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 8, paddingBottom: 8 }}>
          <div style={{ flex: 1 }} />
          {messages.map((msg, i) => {
            if (msg.role === "thinking") {
              const tm = msg as ThinkingMsg;
              return (
                <div key={i} className="hud-fadein">
                  <HudStepper steps={tm.steps} currentStep={tm.steps.length} expanded={expandedSteppers[tm.id] ?? false} onToggle={() => toggle(tm.id)} finished />
                </div>
              );
            }

            const isUser = msg.role === "user";
            const am = msg as AssistantMsg;

            return (
              <div key={i} className="hud-fadein" style={{ display: "flex", flexDirection: "column", alignItems: isUser ? "flex-end" : "flex-start" }}>
                <div
                  className="hud-label"
                  style={{
                    fontSize: 11,
                    color: isUser ? "rgba(255,255,255,0.5)" : "rgba(120,255,200,0.7)",
                    fontFamily: "'DM Mono', monospace",
                    fontWeight: 600,
                    marginBottom: 3,
                    padding: "0 6px",
                    letterSpacing: "0.05em",
                  }}
                >
                  {isUser ? "MARGARET" : "MIRA"} \u00b7 {fmt(msg.time)}
                </div>
                <div
                  style={{
                    padding: "12px 18px",
                    borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                    background: isUser ? "rgba(0,0,0,0.55)" : "rgba(0,0,0,0.5)",
                    backdropFilter: "blur(20px)",
                    WebkitBackdropFilter: "blur(20px)",
                    border: `1px solid ${isUser ? "rgba(255,255,255,0.15)" : "rgba(120,255,200,0.2)"}`,
                    color: "rgba(255,255,255,0.95)",
                    fontSize: 17,
                    fontWeight: 600,
                    lineHeight: 1.6,
                    maxWidth: "95%",
                    textShadow: "0 1px 3px rgba(0,0,0,0.6)",
                  }}
                >
                  {isUser ? (msg as UserMsg).text : <Streamdown mode="static">{am.text}</Streamdown>}
                  {!isUser && am.location && am.confidence != null && am.coords && (
                    <LocationTag location={am.location} confidence={am.confidence} coords={am.coords} />
                  )}
                  {!isUser && am.type === "escalation" && (
                    <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 6, padding: "6px 12px", background: "rgba(200,80,30,0.3)", border: "1px solid rgba(255,150,100,0.4)", borderRadius: 8, fontSize: 13, color: "rgba(255,200,160,1)", fontWeight: 600 }}>
                      <svg width="12" height="12" fill="none" viewBox="0 0 24 24"><path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6A19.79 19.79 0 012.12 4.18 2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z" stroke="currentColor" strokeWidth="2" /></svg>
                      Contacting David Chen\u2026
                    </div>
                  )}
                  {/* Citations from Sonar */}
                  {!isUser && am.citations && am.citations.length > 0 && (
                    <div style={{ marginTop: 8, paddingTop: 6, borderTop: "1px solid rgba(120,255,200,0.15)" }}>
                      <div style={{ fontSize: 10, fontWeight: 700, color: "rgba(120,255,200,0.6)", letterSpacing: "0.08em", marginBottom: 4 }}>SOURCES</div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        {am.citations.map((c, ci) => (
                          <CitationCard key={ci} citation={c} variant="dark" />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          {processing && (
            <div className="hud-fadein">
              <HudStepper steps={processing.steps} currentStep={processing.currentStep} expanded={expandedSteppers[processing.id] ?? true} onToggle={() => toggle(processing.id)} finished={processing.finished} />
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {liveMode ? (
          /* Live mode: mirroring indicator */
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, padding: "12px 0" }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "rgba(120,255,200,0.8)", animation: "hud-pulse-kf 1.5s ease-in-out infinite" }} />
            <span style={{ fontSize: 12, fontFamily: "'DM Mono', monospace", color: "rgba(120,255,200,0.7)", fontWeight: 500, letterSpacing: "0.05em", textShadow: "0 1px 3px rgba(0,0,0,0.6)" }}>
              MIRRORING DEVICE CHAT
            </span>
          </div>
        ) : (
          /* Demo mode: quick actions + input */
          <>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "6px 0" }}>
              {["Where are my glasses?", "What meds do I take?", "I need help", "Where is my walker?"].map((q) => (
                <button
                  key={q}
                  onClick={() => !processing && send(q)}
                  style={{
                    padding: "7px 16px",
                    borderRadius: 10,
                    border: "1px solid rgba(255,255,255,0.12)",
                    background: "rgba(0,0,0,0.4)",
                    backdropFilter: "blur(12px)",
                    fontSize: 13,
                    color: processing ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.7)",
                    textShadow: "0 1px 2px rgba(0,0,0,0.5)",
                    fontWeight: 500,
                    cursor: processing ? "default" : "pointer",
                    fontFamily: "inherit",
                    transition: "all 0.2s",
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  background: "rgba(0,0,0,0.55)",
                  backdropFilter: "blur(20px)",
                  borderRadius: 14,
                  padding: "4px 6px 4px 16px",
                  border: "1px solid rgba(255,255,255,0.12)",
                }}
              >
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !processing && send(input)}
                  placeholder="Type or tap mic\u2026"
                  disabled={!!processing}
                  style={{
                    flex: 1,
                    border: "none",
                    outline: "none",
                    fontSize: 14,
                    color: "rgba(255,255,255,0.85)",
                    fontFamily: "inherit",
                    background: "transparent",
                  }}
                />
                <button
                  onClick={() => !processing && send(input)}
                  disabled={!!processing || !input.trim()}
                  style={{
                    width: 34,
                    height: 34,
                    borderRadius: "50%",
                    border: "none",
                    background: input.trim() && !processing ? "rgba(120,255,200,0.15)" : "rgba(255,255,255,0.04)",
                    cursor: input.trim() && !processing ? "pointer" : "default",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.2s",
                  }}
                >
                  <svg width="14" height="14" fill="none" viewBox="0 0 24 24">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke={input.trim() && !processing ? "rgba(120,255,200,0.8)" : "rgba(255,255,255,0.2)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <AudioWaveform active={listening} />
                <button
                  onClick={handleMic}
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: "50%",
                    border: `1.5px solid ${listening ? "rgba(120,255,200,0.6)" : "rgba(255,255,255,0.12)"}`,
                    background: listening ? "rgba(120,255,200,0.15)" : "rgba(0,0,0,0.55)",
                    backdropFilter: "blur(12px)",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.3s ease",
                    flexShrink: 0,
                    animation: listening ? "hud-listening 1.5s ease-in-out infinite" : "none",
                  }}
                >
                  <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
                    <rect x="9" y="2" width="6" height="12" rx="3" fill={listening ? "rgba(120,255,200,0.9)" : "rgba(255,255,255,0.5)"} />
                    <path d="M5 10a7 7 0 0014 0" stroke={listening ? "rgba(120,255,200,0.9)" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" />
                    <path d="M12 19v3M8 22h8" stroke={listening ? "rgba(120,255,200,0.9)" : "rgba(255,255,255,0.5)"} strokeWidth="2" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
