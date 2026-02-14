"use client";

import { useState, useEffect, useRef, useCallback } from "react";

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
        background: "rgba(0,0,0,0.5)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        border: "1px solid rgba(120,255,200,0.12)",
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
                          fontSize: 12,
                          fontWeight: active ? 600 : 400,
                          color: done ? "rgba(120,255,200,0.85)" : active ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.25)",
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
}
type Msg = UserMsg | ThinkingMsg | AssistantMsg;

/* ================================================================
   Main: AR Glasses Stream Overlay
   ================================================================ */

export default function StreamPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [processing, setProcessing] = useState<Processing | null>(null);
  const [expandedSteppers, setExpandedSteppers] = useState<Record<number, boolean>>({});
  const [listening, setListening] = useState(false);
  const [showEscalation, setShowEscalation] = useState(false);
  const [input, setInput] = useState("");
  const [clock, setClock] = useState<Date | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const pidRef = useRef(0);

  useEffect(() => {
    setClock(new Date());
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, processing]);

  const toggle = useCallback(
    (id: number) => setExpandedSteppers((p) => ({ ...p, [id]: !p[id] })),
    []
  );

  const send = useCallback((text: string) => {
    if (!text.trim()) return;
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

  // Auto-demo on mount
  useEffect(() => {
    const t1 = setTimeout(() => send("Good morning, Mira"), 2500);
    const t2 = setTimeout(() => send("Where are my pills?"), 9000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [send]);

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
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:3px}
      `}</style>

      {/* ──── OUTDOOR SCENE BACKGROUND ──── */}
      <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
        {/* Sky gradient — warm outdoor daylight */}
        <div style={{ position: "absolute", inset: 0, background: "linear-gradient(180deg, #3b5998 0%, #87CEEB 25%, #B0D4E8 45%, #8fbc8f 65%, #6B8E6B 80%, #4a6741 100%)" }} />
        {/* Sun glow */}
        <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse 50% 40% at 70% 15%, rgba(255,240,180,0.35) 0%, transparent 70%)" }} />
        {/* Ground warmth */}
        <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: "40%", background: "linear-gradient(0deg, rgba(80,100,60,0.5) 0%, transparent 100%)" }} />
        {/* Tree silhouette shapes */}
        <div style={{ position: "absolute", left: "5%", top: "20%", width: "15%", height: "55%", background: "rgba(40,60,35,0.3)", borderRadius: "50% 50% 5% 5%", filter: "blur(4px)" }} />
        <div style={{ position: "absolute", right: "8%", top: "15%", width: "12%", height: "60%", background: "rgba(35,55,30,0.25)", borderRadius: "50% 50% 5% 5%", filter: "blur(5px)" }} />
        {/* Path/walkway */}
        <div style={{ position: "absolute", bottom: 0, left: "30%", width: "40%", height: "30%", background: "linear-gradient(0deg, rgba(160,140,110,0.2) 0%, transparent 80%)", borderRadius: "50% 50% 0 0", filter: "blur(3px)" }} />
        {/* Lens vignette */}
        <div style={{ position: "absolute", inset: 0, boxShadow: "inset 0 0 150px 60px rgba(0,0,0,0.4)" }} />
        {/* Scanline */}
        <div style={{ position: "absolute", inset: 0, overflow: "hidden", pointerEvents: "none" }}>
          <div style={{ position: "absolute", left: 0, right: 0, height: "50%", background: "linear-gradient(180deg, transparent 0%, rgba(120,255,200,0.012) 50%, transparent 100%)", animation: "hud-scanline 8s linear infinite" }} />
        </div>
        {/* Grid overlay */}
        <div style={{ position: "absolute", inset: 0, opacity: 0.03, backgroundImage: "linear-gradient(rgba(120,255,200,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(120,255,200,0.3) 1px, transparent 1px)", backgroundSize: "60px 60px", animation: "hud-grid-pulse 4s ease-in-out infinite" }} />
        {/* Film grain */}
        <div style={{ position: "absolute", inset: 0, opacity: 0.1, mixBlendMode: "overlay", background: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")" }} />
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
      <div style={{ position: "absolute", top: 28, left: 56, zIndex: 10, display: "flex", alignItems: "center", gap: 14 }}>
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
      <div style={{ position: "absolute", top: 28, right: 56, zIndex: 10, textAlign: "right" }}>
        <div style={{ fontSize: 20, color: "rgba(255,255,255,0.8)", fontFamily: "'DM Mono', monospace", fontWeight: 500, letterSpacing: "0.05em" }}>{fmtClock(clock)}</div>
        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", fontFamily: "'DM Mono', monospace", marginTop: 2 }}>CAM 01 \u00b7 1080p \u00b7 30fps</div>
      </div>

      {/* ──── TOP CENTER: Patient bar ──── */}
      <div style={{ position: "absolute", top: 28, left: "50%", transform: "translateX(-50%)", zIndex: 10, display: "flex", alignItems: "center", gap: 16, padding: "8px 20px", background: "rgba(0,0,0,0.4)", backdropFilter: "blur(12px)", borderRadius: 12, border: "1px solid rgba(255,255,255,0.08)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "rgba(120,255,200,0.7)" }} />
          <span style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", fontWeight: 600 }}>Margaret Chen</span>
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.3)" }}>\u00b7 78</span>
        </div>
        <div style={{ width: 1, height: 16, background: "rgba(255,255,255,0.1)" }} />
        <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontFamily: "'DM Mono', monospace" }}>3 meds \u00b7 2 allergies</span>
      </div>

      {/* ──── BOTTOM LEFT: Sensor data ──── */}
      <div style={{ position: "absolute", bottom: 32, left: 56, zIndex: 10 }}>
        <div style={{ fontSize: 10, color: "rgba(255,255,255,0.25)", fontFamily: "'DM Mono', monospace", lineHeight: 1.8 }}>
          <div>DEPTH: 3.2m \u00b7 OUTDOOR</div>
          <div>IMU: stable \u00b7 GPS: 37.4275\u00b0N</div>
          <div>BATT: 72% \u00b7 TEMP: 68\u00b0F</div>
        </div>
      </div>

      {/* ──── CHAT OVERLAY ──── */}
      <div style={{ position: "absolute", bottom: 28, right: 28, top: 80, width: 400, zIndex: 20, display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", justifyContent: "flex-end", gap: 6, paddingBottom: 8 }}>
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
                  style={{
                    fontSize: 10,
                    color: isUser ? "rgba(255,255,255,0.3)" : "rgba(120,255,200,0.5)",
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
                    padding: "10px 16px",
                    borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                    background: isUser ? "rgba(255,255,255,0.08)" : "rgba(120,255,200,0.06)",
                    backdropFilter: "blur(16px)",
                    border: `1px solid ${isUser ? "rgba(255,255,255,0.1)" : "rgba(120,255,200,0.12)"}`,
                    color: isUser ? "rgba(255,255,255,0.85)" : "rgba(255,255,255,0.9)",
                    fontSize: 14,
                    lineHeight: 1.5,
                    maxWidth: "95%",
                  }}
                >
                  {isUser ? (msg as UserMsg).text : am.text}
                  {!isUser && am.location && am.confidence != null && am.coords && (
                    <LocationTag location={am.location} confidence={am.confidence} coords={am.coords} />
                  )}
                  {!isUser && am.type === "escalation" && (
                    <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 6, padding: "6px 12px", background: "rgba(200,80,30,0.15)", border: "1px solid rgba(200,80,30,0.3)", borderRadius: 8, fontSize: 12, color: "rgba(255,180,130,0.9)", fontWeight: 500 }}>
                      <svg width="12" height="12" fill="none" viewBox="0 0 24 24"><path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6A19.79 19.79 0 012.12 4.18 2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z" stroke="currentColor" strokeWidth="2" /></svg>
                      Contacting David Chen\u2026
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

        {/* Quick actions */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", padding: "6px 0" }}>
          {["Where are my glasses?", "What meds do I take?", "I need help", "Where is my walker?"].map((q) => (
            <button
              key={q}
              onClick={() => !processing && send(q)}
              style={{
                padding: "6px 14px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.08)",
                background: "rgba(255,255,255,0.04)",
                backdropFilter: "blur(8px)",
                fontSize: 12,
                color: processing ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.5)",
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

        {/* Input */}
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              background: "rgba(0,0,0,0.4)",
              backdropFilter: "blur(16px)",
              borderRadius: 14,
              padding: "4px 6px 4px 16px",
              border: "1px solid rgba(255,255,255,0.08)",
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
          <button
            onClick={handleMic}
            style={{
              width: 48,
              height: 48,
              borderRadius: "50%",
              border: `1.5px solid ${listening ? "rgba(120,255,200,0.6)" : "rgba(255,255,255,0.12)"}`,
              background: listening ? "rgba(120,255,200,0.12)" : "rgba(0,0,0,0.4)",
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
    </div>
  );
}
