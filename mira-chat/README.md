# Mira - Assisted Living Intelligence

Voice/text chat assistant + supervisor dashboard for assisted living facilities. Handles object finding (integrates with CV team), medical Q&A, and caregiver escalation.

## Quick Start

```bash
cd mira
npm install
cp .env.example .env.local  # fill in your keys
npm run dev                  # http://localhost:3000
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes | Supabase anon/public key |
| `SUPABASE_URL` | Yes | Same as above (for server) |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (uses gpt-4o-mini) |
| `TWILIO_ACCOUNT_SID` | No | Twilio account SID for SMS |
| `TWILIO_AUTH_TOKEN` | No | Twilio auth token |
| `TWILIO_PHONE_NUMBER` | No | Twilio sender number (E.164) |
| `TWILIO_ALERT_TO` | No | Default SMS recipient (E.164) |

## Database Setup

Run these SQL files in your Supabase SQL Editor (Dashboard > SQL Editor), in order:

1. `supabase/migrations/001_schema.sql` - Creates tables
2. `supabase/migrations/002_enable_realtime.sql` - Enables realtime subscriptions
3. `supabase/migrations/003_seed_demo.sql` - Seeds 3 demo patients

## Pages

- `/` - Landing page
- `/device` - Resident chat interface (voice + text)
- `/dashboard` - Supervisor dashboard (patient management + event timeline)

## API Endpoints

### Events
```bash
# Create event
curl -X POST http://localhost:3000/api/events \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"a1b2c3d4-0001-4000-8000-000000000001","type":"CHAT_USER_UTTERANCE","severity":"GREEN","receipt_text":"Hello Mira"}'

# Get events
curl "http://localhost:3000/api/events?patient_id=a1b2c3d4-0001-4000-8000-000000000001"
```

### Chat
```bash
# Ask a question
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"a1b2c3d4-0001-4000-8000-000000000001","message":"What medications do I take?"}'

# Find an object
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"a1b2c3d4-0001-4000-8000-000000000001","message":"Where are my glasses?"}'
```

### Object Finding (CV Team Integration)
```bash
# CV team reports object found
curl -X POST http://localhost:3000/api/objects/update \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"a1b2c3d4-0001-4000-8000-000000000001","object_name":"glasses","found":true,"location":"on the nightstand in room 204","confidence":0.95}'
```

### Escalation
```bash
curl -X POST http://localhost:3000/api/escalate \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"a1b2c3d4-0001-4000-8000-000000000001","reason":"Resident reported chest pain"}'
```

## Architecture

```
User speaks "Where are my glasses?"
  -> POST /api/chat (intent: find_object)
  -> Creates object_requests row (PENDING)
  -> Emits FIND_OBJECT_REQUESTED event
  -> Returns friendly reply + request_id

CV team listens for FIND_OBJECT_REQUESTED via Supabase Realtime
  -> Runs computer vision
  -> POST /api/objects/update { found: true, location: "nightstand" }
  -> Updates object_requests (FOUND) + upserts object_state
  -> Emits OBJECT_LOCATED event

Device UI receives OBJECT_LOCATED via Supabase Realtime
  -> Matches request_id
  -> Shows system message + announces via TTS
  -> Dashboard timeline updates in real-time
```

## Event Types

| Type | Description |
|------|-------------|
| `CHAT_USER_UTTERANCE` | Resident sent a message |
| `CHAT_ASSISTANT_RESPONSE` | Mira replied |
| `FIND_OBJECT_REQUESTED` | Object search initiated |
| `OBJECT_LOCATED` | Object found by CV team |
| `OBJECT_NOT_FOUND` | Object could not be found |
| `ESCALATED` | Caregiver alert triggered |
