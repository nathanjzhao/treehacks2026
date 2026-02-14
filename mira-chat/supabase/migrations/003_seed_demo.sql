-- Seed 3 demo patients for Mira

INSERT INTO public.patients (id, display_name, room_number) VALUES
  ('a1b2c3d4-0001-4000-8000-000000000001', 'Margaret Chen', '204'),
  ('a1b2c3d4-0002-4000-8000-000000000002', 'Robert Williams', '112'),
  ('a1b2c3d4-0003-4000-8000-000000000003', 'Helen Garcia', '318')
ON CONFLICT DO NOTHING;

-- Margaret Chen - Type 2 Diabetes, Hypertension, Mild Cognitive Impairment
INSERT INTO public.patient_cards (patient_id, card_json, source) VALUES (
  'a1b2c3d4-0001-4000-8000-000000000001',
  '{
    "display_name": "Margaret Chen",
    "room_number": "204",
    "demographics": { "age_range": "78-82", "sex": "female" },
    "conditions": [
      { "name": "Type 2 Diabetes", "onset_year": 2015 },
      { "name": "Hypertension", "onset_year": 2012 },
      { "name": "Mild Cognitive Impairment", "onset_year": 2023 }
    ],
    "allergies": [
      { "substance": "Penicillin", "reaction": "Rash and hives" },
      { "substance": "Sulfa drugs", "reaction": "Nausea" }
    ],
    "meds": [
      { "name": "Metformin", "strength": "500mg", "schedule_times": ["08:00", "20:00"], "purpose": "Blood sugar control", "with_food": true, "is_critical": true },
      { "name": "Lisinopril", "strength": "10mg", "schedule_times": ["08:00"], "purpose": "Blood pressure", "with_food": false, "is_critical": true },
      { "name": "Donepezil", "strength": "5mg", "schedule_times": ["21:00"], "purpose": "Cognitive support", "with_food": false, "is_critical": false }
    ],
    "contacts": {
      "caregiver_name": "David Chen",
      "caregiver_phone": "+15550201",
      "emergency_phone": "911"
    }
  }'::jsonb,
  'seed'
);

-- Robert Williams - Parkinson's, Osteoarthritis, Depression
INSERT INTO public.patient_cards (patient_id, card_json, source) VALUES (
  'a1b2c3d4-0002-4000-8000-000000000002',
  '{
    "display_name": "Robert Williams",
    "room_number": "112",
    "demographics": { "age_range": "72-76", "sex": "male" },
    "conditions": [
      { "name": "Parkinson''s Disease", "onset_year": 2019 },
      { "name": "Osteoarthritis", "onset_year": 2016 },
      { "name": "Depression", "onset_year": 2021 }
    ],
    "allergies": [
      { "substance": "Codeine", "reaction": "Severe nausea and dizziness" }
    ],
    "meds": [
      { "name": "Carbidopa/Levodopa", "strength": "25/100mg", "schedule_times": ["07:00", "12:00", "17:00", "22:00"], "purpose": "Parkinson''s motor control", "with_food": false, "is_critical": true },
      { "name": "Sertraline", "strength": "50mg", "schedule_times": ["08:00"], "purpose": "Depression management", "with_food": true, "is_critical": false },
      { "name": "Acetaminophen", "strength": "500mg", "schedule_times": [], "purpose": "Pain relief (as needed)", "with_food": false, "is_critical": false }
    ],
    "contacts": {
      "caregiver_name": "Lisa Williams",
      "caregiver_phone": "+15550302",
      "emergency_phone": "911"
    }
  }'::jsonb,
  'seed'
);

-- Helen Garcia - Heart Failure, AFib, CKD Stage 3
INSERT INTO public.patient_cards (patient_id, card_json, source) VALUES (
  'a1b2c3d4-0003-4000-8000-000000000003',
  '{
    "display_name": "Helen Garcia",
    "room_number": "318",
    "demographics": { "age_range": "82-86", "sex": "female" },
    "conditions": [
      { "name": "Heart Failure (NYHA II)", "onset_year": 2018 },
      { "name": "Atrial Fibrillation", "onset_year": 2017 },
      { "name": "Chronic Kidney Disease Stage 3", "onset_year": 2020 }
    ],
    "allergies": [
      { "substance": "Aspirin", "reaction": "GI bleeding" },
      { "substance": "Iodine contrast", "reaction": "Anaphylaxis risk" }
    ],
    "meds": [
      { "name": "Furosemide", "strength": "40mg", "schedule_times": ["07:00"], "purpose": "Fluid management", "with_food": false, "is_critical": true },
      { "name": "Warfarin", "strength": "5mg", "schedule_times": ["18:00"], "purpose": "Blood thinner for AFib", "with_food": false, "is_critical": true },
      { "name": "Metoprolol", "strength": "25mg", "schedule_times": ["08:00", "20:00"], "purpose": "Heart rate control", "with_food": true, "is_critical": true }
    ],
    "contacts": {
      "caregiver_name": "Maria Garcia",
      "caregiver_phone": "+15550403",
      "emergency_phone": "911"
    }
  }'::jsonb,
  'seed'
);
