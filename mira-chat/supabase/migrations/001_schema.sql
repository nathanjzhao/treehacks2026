-- Mira backend spine schema
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Patients table
CREATE TABLE IF NOT EXISTS public.patients (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id text UNIQUE,
  display_name text NOT NULL,
  room_number text,
  photo_url text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Patient cards: stores full JSON medical info per patient
CREATE TABLE IF NOT EXISTS public.patient_cards (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  card_json jsonb NOT NULL,
  source text NOT NULL DEFAULT 'manual_entry',
  schema_version int NOT NULL DEFAULT 1,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_patient_cards_patient_latest
  ON public.patient_cards (patient_id, created_at DESC);

-- Events table (the event spine)
CREATE TABLE IF NOT EXISTS public.events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  type text NOT NULL,
  severity text NOT NULL DEFAULT 'GREEN',
  receipt_text text,
  decision jsonb NOT NULL DEFAULT '{}'::jsonb,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
  source text NOT NULL DEFAULT 'backend',
  schema_version int NOT NULL DEFAULT 1,
  idempotency_key text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency_key
  ON public.events (idempotency_key)
  WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_events_patient_created_desc
  ON public.events (patient_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_patient_type
  ON public.events (patient_id, type);

-- Object requests (CV queue)
CREATE TABLE IF NOT EXISTS public.object_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  object_name text NOT NULL,
  context text,
  status text NOT NULL DEFAULT 'PENDING'
    CHECK (status IN ('PENDING', 'FOUND', 'NOT_FOUND')),
  created_at timestamptz NOT NULL DEFAULT now(),
  resolved_at timestamptz,
  resolved_location jsonb,
  confidence float
);

CREATE INDEX IF NOT EXISTS idx_object_requests_patient_status
  ON public.object_requests (patient_id, status);

-- Object state (latest known location per patient/object)
CREATE TABLE IF NOT EXISTS public.object_state (
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  object_name text NOT NULL,
  location jsonb,
  confidence float,
  updated_at timestamptz NOT NULL DEFAULT now(),
  source text,
  UNIQUE (patient_id, object_name)
);

-- Escalations table
CREATE TABLE IF NOT EXISTS public.escalations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  event_id uuid REFERENCES public.events(id) ON DELETE SET NULL,
  channel text NOT NULL,
  target text,
  status text NOT NULL,
  attempt_count int NOT NULL DEFAULT 1,
  response_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_escalations_patient_created_desc
  ON public.escalations (patient_id, created_at DESC);

-- updated_at triggers
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_patients_updated_at
  BEFORE UPDATE ON public.patients
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TRIGGER trg_patient_cards_updated_at
  BEFORE UPDATE ON public.patient_cards
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TRIGGER trg_events_updated_at
  BEFORE UPDATE ON public.events
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TRIGGER trg_escalations_updated_at
  BEFORE UPDATE ON public.escalations
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Disable RLS for hackathon speed
ALTER TABLE public.patients DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.patient_cards DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.events DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.object_requests DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.object_state DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.escalations DISABLE ROW LEVEL SECURITY;
