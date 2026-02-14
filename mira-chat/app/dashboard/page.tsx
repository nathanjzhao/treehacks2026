"use client";

import { useEffect, useState, useCallback } from "react";
import { supabase, Patient, PatientCard, MiraEvent } from "@/lib/supabase/client";
import Sidebar from "./components/Sidebar";
import PatientDetail from "./components/PatientDetail";
import EventTimeline from "./components/EventTimeline";
import AlertBanner from "./components/AlertBanner";

type PatientWithCard = Patient & { card: Record<string, unknown> | null };

export default function DashboardPage() {
  const [patients, setPatients] = useState<PatientWithCard[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [events, setEvents] = useState<MiraEvent[]>([]);
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  // Load patients
  useEffect(() => {
    async function loadPatients() {
      try {
        const res = await fetch("/api/patients");
        const data = await res.json();
        if (data.ok && data.patients) {
          setPatients(data.patients);
          if (data.patients.length > 0 && !selectedId) {
            setSelectedId(data.patients[0].id);
          }
        }
      } catch (err) {
        console.error("Failed to load patients:", err);
      } finally {
        setLoading(false);
      }
    }
    loadPatients();
  }, []);

  // Load events for selected patient
  useEffect(() => {
    if (!selectedId) return;

    async function loadEvents() {
      try {
        const res = await fetch(`/api/events?patient_id=${selectedId}`);
        const data = await res.json();
        if (data.ok) {
          setEvents(data.events || []);
        }
      } catch (err) {
        console.error("Failed to load events:", err);
      }
    }
    loadEvents();

    // Subscribe to realtime events
    const channel = supabase
      .channel(`dashboard-events-${selectedId}`)
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "events",
          filter: `patient_id=eq.${selectedId}`,
        },
        (payload) => {
          setEvents((prev) => [payload.new as MiraEvent, ...prev]);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [selectedId]);

  const handleSelectPatient = useCallback((id: string) => {
    setSelectedId(id);
    setEvents([]);
    setDismissedAlerts(new Set());
  }, []);

  const handleDismissAlert = useCallback((id: string) => {
    setDismissedAlerts((prev) => new Set(prev).add(id));
  }, []);

  const selectedPatient = patients.find((p) => p.id === selectedId);
  const patientCard = selectedPatient?.card as PatientCard | null;

  const activeAlerts = events.filter(
    (e) => e.severity === "RED" && !dismissedAlerts.has(e.id)
  );

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="w-10 h-10 border-3 border-teal-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="mt-4 text-sm text-muted">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex bg-background overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        patients={patients}
        selectedId={selectedId}
        onSelect={handleSelectPatient}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Alert banner */}
        <AlertBanner alerts={activeAlerts} onDismiss={handleDismissAlert} />

        {selectedPatient ? (
          <div className="flex-1 flex min-h-0">
            {/* Patient detail - left/center */}
            <div className="flex-1 min-w-0 border-r border-border">
              <PatientDetail patient={selectedPatient} card={patientCard} />
            </div>

            {/* Event timeline - right */}
            <div className="w-96 shrink-0 bg-surface border-l border-border">
              <EventTimeline events={events} />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted">
            <div className="text-center">
              <svg className="w-12 h-12 mx-auto mb-4 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
              <p className="text-lg font-medium">Select a resident</p>
              <p className="text-sm mt-1">Choose a resident from the sidebar to view their details</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
