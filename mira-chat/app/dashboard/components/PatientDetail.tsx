"use client";

import { PatientCard, Patient } from "@/lib/supabase/client";

interface PatientDetailProps {
  patient: Patient;
  card: PatientCard | null;
}

function getNextMedTime(meds: PatientCard["meds"]): string | null {
  const now = new Date();
  const nowMinutes = now.getHours() * 60 + now.getMinutes();
  let closest: string | null = null;
  let closestDiff = Infinity;
  for (const med of meds || []) {
    for (const t of med.schedule_times || []) {
      const [h, m] = t.split(":").map(Number);
      const tMin = h * 60 + m;
      const diff = tMin > nowMinutes ? tMin - nowMinutes : tMin + 1440 - nowMinutes;
      if (diff < closestDiff) {
        closestDiff = diff;
        closest = t;
      }
    }
  }
  return closest;
}

function computeRisk(card: PatientCard) {
  const criticalCount = card.meds?.filter((m) => m.is_critical).length || 0;
  const score =
    (card.conditions?.length || 0) + criticalCount * 2 + (card.allergies?.length || 0);
  if (score >= 6) return { level: "HIGH", color: "text-red-600", bg: "bg-red-50" };
  if (score >= 3) return { level: "MODERATE", color: "text-amber-600", bg: "bg-amber-50" };
  return { level: "LOW", color: "text-emerald-600", bg: "bg-emerald-50" };
}

function SummaryStat({
  icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
}) {
  return (
    <div className="text-center">
      <div className="flex items-center justify-center mb-1.5">{icon}</div>
      <div className={`text-xl font-bold ${color || "text-foreground"}`}>
        {value}
      </div>
      <div className="text-[10px] font-bold uppercase tracking-wider text-muted">
        {label}
      </div>
      {sub && <div className="text-[10px] text-muted mt-0.5">{sub}</div>}
    </div>
  );
}

export default function PatientDetail({ patient, card }: PatientDetailProps) {
  if (!card) {
    return (
      <div className="flex items-center justify-center h-full text-muted">
        <p>No patient card available for {patient.display_name}.</p>
      </div>
    );
  }

  return (
    <div className="p-6 overflow-y-auto h-full">
      {/* Patient header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-foreground tracking-tight font-display">
          {card.display_name}
        </h2>
        <div className="flex items-center gap-3 mt-1.5 text-sm text-muted">
          {card.room_number && (
            <span className="inline-flex items-center gap-1.5 bg-teal-50 text-teal-700 px-2.5 py-1 rounded-lg text-xs font-semibold">
              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              </svg>
              Room {card.room_number}
            </span>
          )}
          {card.demographics && (
            <>
              <span className="text-slate-300">|</span>
              <span className="text-xs font-medium">{card.demographics.age_range} years</span>
              <span className="text-slate-300">|</span>
              <span className="text-xs font-medium capitalize">{card.demographics.sex}</span>
            </>
          )}
        </div>
      </div>

      {/* Summary strip */}
      <div className="mb-6 bg-gradient-to-r from-teal-50 to-emerald-50 border border-teal-200/50 rounded-2xl p-5 shadow-sm">
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-4">
          <SummaryStat
            icon={
              <svg className="w-5 h-5 text-teal-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z" />
                <path d="m8.5 8.5 7 7" />
              </svg>
            }
            label="Medications"
            value={card.meds?.length || 0}
            sub={`${card.meds?.filter((m) => m.is_critical).length || 0} critical`}
          />
          <SummaryStat
            icon={
              <svg className="w-5 h-5 text-amber-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
            }
            label="Conditions"
            value={card.conditions?.length || 0}
          />
          <SummaryStat
            icon={
              <svg className="w-5 h-5 text-rose-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            }
            label="Allergies"
            value={card.allergies?.length || 0}
            sub={card.allergies?.length === 0 ? "NKDA" : undefined}
          />
          <SummaryStat
            icon={
              <svg className="w-5 h-5 text-blue-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            }
            label="Next Med"
            value={getNextMedTime(card.meds) || "N/A"}
          />
          <SummaryStat
            icon={
              <svg className={`w-5 h-5 ${computeRisk(card).color}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            }
            label="Risk Level"
            value={computeRisk(card).level}
            color={computeRisk(card).color}
          />
          <SummaryStat
            icon={
              <svg className="w-5 h-5 text-cyan-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z" />
              </svg>
            }
            label="Emergency"
            value={card.contacts?.caregiver_name?.split(" ")[0] || "N/A"}
          />
        </div>
      </div>

      {/* Cards grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Medications */}
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm card-hover">
          <div className="flex items-center gap-2.5 mb-4">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-100 to-emerald-50 flex items-center justify-center">
              <svg className="w-4 h-4 text-teal-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z" />
                <path d="m8.5 8.5 7 7" />
              </svg>
            </div>
            <h3 className="font-bold text-xs uppercase tracking-wider text-muted">
              Medications ({card.meds?.length || 0})
            </h3>
          </div>
          <div className="space-y-3">
            {card.meds?.map((med, i) => (
              <div key={i} className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm text-foreground">{med.name}</span>
                    {med.is_critical && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded-md text-[10px] font-bold bg-red-100 text-red-700 border border-red-200/50">
                        CRITICAL
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted mt-0.5">
                    {med.strength} &middot; {med.purpose || "No purpose listed"}
                  </div>
                </div>
                <div className="text-xs text-muted text-right shrink-0 ml-2">
                  {med.schedule_times?.length
                    ? med.schedule_times.join(", ")
                    : "As needed"}
                  {med.with_food && (
                    <div className="text-amber-600 font-medium">With food</div>
                  )}
                </div>
              </div>
            ))}
            {(!card.meds || card.meds.length === 0) && (
              <p className="text-sm text-muted">No medications on file</p>
            )}
          </div>
        </div>

        {/* Conditions */}
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm card-hover">
          <div className="flex items-center gap-2.5 mb-4">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-amber-100 to-orange-50 flex items-center justify-center">
              <svg className="w-4 h-4 text-amber-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
            </div>
            <h3 className="font-bold text-xs uppercase tracking-wider text-muted">
              Conditions ({card.conditions?.length || 0})
            </h3>
          </div>
          <div className="space-y-2.5">
            {card.conditions?.map((c, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-sm font-semibold text-foreground">{c.name}</span>
                {c.onset_year && (
                  <span className="text-xs text-muted bg-slate-100 px-2 py-0.5 rounded-md font-medium">
                    Since {c.onset_year}
                  </span>
                )}
              </div>
            ))}
            {(!card.conditions || card.conditions.length === 0) && (
              <p className="text-sm text-muted">No conditions on file</p>
            )}
          </div>
        </div>

        {/* Allergies */}
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm card-hover">
          <div className="flex items-center gap-2.5 mb-4">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-rose-100 to-pink-50 flex items-center justify-center">
              <svg className="w-4 h-4 text-rose-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            </div>
            <h3 className="font-bold text-xs uppercase tracking-wider text-muted">
              Allergies ({card.allergies?.length || 0})
            </h3>
          </div>
          <div className="space-y-2">
            {card.allergies?.map((a, i) => (
              <div key={i}>
                <span className="text-sm font-semibold text-foreground">{a.substance}</span>
                {a.reaction && (
                  <span className="text-xs text-muted ml-2">&mdash; {a.reaction}</span>
                )}
              </div>
            ))}
            {(!card.allergies || card.allergies.length === 0) && (
              <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-600 border border-emerald-200/50">
                No known allergies
              </span>
            )}
          </div>
        </div>

        {/* Emergency Contacts */}
        <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm card-hover">
          <div className="flex items-center gap-2.5 mb-4">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-100 to-blue-50 flex items-center justify-center">
              <svg className="w-4 h-4 text-cyan-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z" />
              </svg>
            </div>
            <h3 className="font-bold text-xs uppercase tracking-wider text-muted">
              Emergency Contacts
            </h3>
          </div>
          {card.contacts ? (
            <div className="space-y-3">
              <div>
                <div className="text-sm font-semibold text-foreground">
                  {card.contacts.caregiver_name}
                </div>
                <a
                  href={`tel:${card.contacts.caregiver_phone}`}
                  className="text-sm text-teal-600 hover:text-teal-700 hover:underline font-medium"
                >
                  {card.contacts.caregiver_phone}
                </a>
                <div className="text-[10px] uppercase tracking-wider text-muted mt-0.5 font-bold">
                  Primary Caregiver
                </div>
              </div>
              <div>
                <div className="text-sm font-semibold text-foreground">Emergency</div>
                <a
                  href={`tel:${card.contacts.emergency_phone}`}
                  className="text-sm text-danger hover:underline font-medium"
                >
                  {card.contacts.emergency_phone}
                </a>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted">No contacts on file</p>
          )}
        </div>
      </div>
    </div>
  );
}
