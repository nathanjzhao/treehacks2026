# Audio Integration: Meta Ray-Ban Gen 3 → Mira

This document describes how to wire audio from Meta Ray-Ban Gen 3 glasses through the Mira backend, enabling voice-driven interactions for assisted living residents.

---

## Architecture Overview

```
Ray-Ban Gen 3 (mic)
    │ Bluetooth audio
    ▼
Android Phone (DAT SDK)
    │ captures audio stream
    ▼
┌─────────────────────────────────┐
│  Whisper STT                    │
│  POST /api/voice/transcribe     │
│  (or direct OpenAI Whisper API) │
└────────────┬────────────────────┘
             │ transcribed text
             ▼
┌─────────────────────────────────┐
│  Mira Chat LLM (SSE stream)    │
│  POST /api/chat                 │
│  OpenRouter → gpt-4o-mini       │
│  with function calling:         │
│   • find_object                 │
│   • escalate_to_caregiver       │
│   • lookup_medication           │
└────────────┬────────────────────┘
             │ SSE events + final reply
             ▼
┌─────────────────────────────────┐
│  Response routing:              │
│  • TTS → glasses speaker        │
│  • /stream page (HUD overlay)   │
│  • Dashboard (realtime feed)    │
│  • Android app UI               │
└─────────────────────────────────┘
```

---

## Step 1: Capture Audio from Ray-Ban Gen 3

The glasses stream audio to the phone over Bluetooth. On Android, the glasses mic shows up as a Bluetooth SCO audio source. You do NOT need the DAT SDK for audio capture — standard Android audio APIs work.

### Option A: Standard Android AudioRecord (recommended)

```kotlin
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder

// Start Bluetooth SCO to route glasses mic → phone
val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
audioManager.startBluetoothSco()
audioManager.isBluetoothScoOn = true

// Record from Bluetooth mic
val sampleRate = 16000
val bufferSize = AudioRecord.getMinBufferSize(
    sampleRate,
    AudioFormat.CHANNEL_IN_MONO,
    AudioFormat.ENCODING_PCM_16BIT
)

val recorder = AudioRecord(
    MediaRecorder.AudioSource.DEFAULT,  // or VOICE_COMMUNICATION
    sampleRate,
    AudioFormat.CHANNEL_IN_MONO,
    AudioFormat.ENCODING_PCM_16BIT,
    bufferSize
)

recorder.startRecording()
```

### Option B: MediaRecorder (simpler, outputs file)

```kotlin
val recorder = MediaRecorder().apply {
    setAudioSource(MediaRecorder.AudioSource.DEFAULT)
    setOutputFormat(MediaRecorder.OutputFormat.WEBM)
    setAudioEncoder(MediaRecorder.AudioEncoder.OPUS)
    setOutputFile(outputFile.absolutePath)
    prepare()
    start()
}
```

### Voice Activity Detection (silence auto-stop)

The web app uses these VAD parameters — replicate on Android:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SILENCE_THRESHOLD` | 8 | RMS below this = silence (0-128 scale) |
| `SILENCE_DURATION_MS` | 2200 | 2.2s continuous silence → auto-stop |
| `MIN_SPEECH_DURATION_MS` | 2000 | Must speak 2s before silence detection starts |
| `MAX_RECORDING_MS` | 30000 | 30s hard cap |

Compute RMS from PCM samples:
```kotlin
fun computeRms(buffer: ShortArray, bytesRead: Int): Double {
    var sum = 0.0
    for (i in 0 until bytesRead) {
        val normalized = buffer[i].toDouble() / Short.MAX_VALUE
        sum += normalized * normalized
    }
    return sqrt(sum / bytesRead) * 128  // scale to 0-128
}
```

---

## Step 2: Send Audio to Whisper for Transcription

### Option A: Use Mira's proxy endpoint

```
POST {MIRA_BASE_URL}/api/voice/transcribe
Content-Type: multipart/form-data

Form field: "audio" = <audio file> (webm/opus, mp3, wav, m4a all work)
```

**Response:**
```json
{ "ok": true, "text": "Where are my pills?" }
```

This proxies to OpenAI Whisper (`whisper-1` model). The server needs `OPENAI_API_KEY` in env.

### Option B: Call OpenAI Whisper directly from Android (lower latency)

```kotlin
val client = OkHttpClient()

val audioBody = audioFile.asRequestBody("audio/webm".toMediaType())
val multipart = MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "audio.webm", audioBody)
    .addFormDataPart("model", "whisper-1")
    .addFormDataPart("language", "en")
    .addFormDataPart("response_format", "json")
    .build()

val request = Request.Builder()
    .url("https://api.openai.com/v1/audio/transcriptions")
    .addHeader("Authorization", "Bearer $OPENAI_API_KEY")
    .post(multipart)
    .build()

val response = client.newCall(request).execute()
val json = JSONObject(response.body!!.string())
val transcribedText = json.getString("text")
```

---

## Step 3: Send Transcribed Text to Mira Chat API

The chat endpoint returns an **SSE (Server-Sent Events) stream** with real-time processing steps, then a final result.

### Request

```
POST {MIRA_BASE_URL}/api/chat
Content-Type: application/json

{
  "patient_id": "a1b2c3d4-0001-4000-8000-000000000001",
  "message": "Where are my pills?"
}
```

**Demo patient IDs:**
| Patient | ID | Room |
|---------|------|------|
| Margaret Chen | `a1b2c3d4-0001-4000-8000-000000000001` | 204 |
| Robert Williams | `a1b2c3d4-0002-4000-8000-000000000002` | 112 |
| Helen Garcia | `a1b2c3d4-0003-4000-8000-000000000003` | 318 |

### SSE Response Format

The response is `Content-Type: text/event-stream`. Each event is a line like:

```
data: {"type":"step","index":0,"label":"Interpreting your message","status":"active"}

data: {"type":"step_done","index":0}

data: {"type":"step","index":1,"label":"Reviewing health records","status":"active"}

data: {"type":"step_done","index":1}

data: {"type":"step","index":2,"label":"Recalling conversation","status":"active"}

data: {"type":"step_done","index":2}

data: {"type":"step","index":3,"label":"Thinking","status":"active","detail":"Analyzing intent…"}

data: {"type":"step_done","index":3}

data: {"type":"step","index":4,"label":"Searching for pill organizer","status":"active","searches":["pill organizer"]}

data: {"type":"step_done","index":4}

data: {"type":"step","index":5,"label":"Composing response","status":"active"}

data: {"type":"step_done","index":5}

data: {"type":"result","ok":true,"reply":"I'm looking for your pill organizer now...","action":"FIND_OBJECT","request_id":"uuid-here","object_name":"pill organizer","event_id":"uuid-here"}
```

### Event types

| type | Fields | Description |
|------|--------|-------------|
| `step` | `index`, `label`, `status`, `detail?`, `searches?` | New processing step started |
| `step_done` | `index` | Step at index completed |
| `text` | `chunk` | Streaming text chunk of the final reply (arrives before `result`) |
| `result` | `ok`, `reply`, `action`, `request_id?`, `object_name?`, `citations?`, `event_id` | Final response |

**Note:** When `find_object` is triggered, additional steps appear from the 3D scene explorer (e.g., "Scanning the scene...", "Asking AI to find 'pills'...", "Triangulating 3D position...", "Found it! Flying to object..."). These are real-time status updates from the explorer and will appear between the "Searching for ..." step and the "Composing response" step.

### `action` values in result

| Action | Meaning | Extra fields |
|--------|---------|-------------|
| `ANSWER` | General Q&A response | — |
| `FIND_OBJECT` | Object search initiated | `request_id`, `object_name` |
| `ESCALATE` | Caregiver alerted via SMS | — |

### Android SSE parsing

```kotlin
val client = OkHttpClient()
val body = """{"patient_id":"$patientId","message":"$text"}"""
    .toRequestBody("application/json".toMediaType())

val request = Request.Builder()
    .url("$MIRA_BASE_URL/api/chat")
    .post(body)
    .build()

client.newCall(request).enqueue(object : Callback {
    override fun onResponse(call: Call, response: Response) {
        val reader = response.body!!.source()
        while (!reader.exhausted()) {
            val line = reader.readUtf8Line() ?: break
            if (!line.startsWith("data: ")) continue
            val json = JSONObject(line.removePrefix("data: "))

            when (json.getString("type")) {
                "step" -> {
                    // Show step in UI: json.getString("label")
                    Log.d("Mira", "Step: ${json.getString("label")}")
                }
                "step_done" -> {
                    // Mark step complete
                }
                "result" -> {
                    val reply = json.getString("reply")
                    val action = json.getString("action")
                    // Use reply for TTS, display, etc.
                    handleResult(reply, action, json)
                }
            }
        }
    }

    override fun onFailure(call: Call, e: IOException) {
        Log.e("Mira", "Chat request failed", e)
    }
})
```

---

## Step 4: Route the Response

Once you have `reply` from the `result` event:

### 4a. Text-to-Speech (play through glasses speaker)

```
POST {MIRA_BASE_URL}/api/voice/tts
Content-Type: application/json

{ "text": "I found your pill organizer on the garden table." }
```

**Response:** Raw `audio/mpeg` bytes. Play directly:

```kotlin
val ttsBody = """{"text":"$reply"}"""
    .toRequestBody("application/json".toMediaType())

val ttsRequest = Request.Builder()
    .url("$MIRA_BASE_URL/api/voice/tts")
    .post(ttsBody)
    .build()

val ttsResponse = client.newCall(ttsRequest).execute()
val audioBytes = ttsResponse.body!!.bytes()

// Write to temp file and play via MediaPlayer
val tempFile = File.createTempFile("mira_tts", ".mp3", cacheDir)
tempFile.writeBytes(audioBytes)

MediaPlayer().apply {
    setDataSource(tempFile.absolutePath)
    // Route to Bluetooth speaker (glasses)
    setAudioAttributes(AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANT)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build())
    prepare()
    start()
}
```

Uses ElevenLabs (Alice voice, `eleven_turbo_v2_5` model). Needs `ELEVENLABS_API_KEY` in server env.

### 4b. Send steps + reply to /stream page (HUD overlay)

The `/stream` page at `{MIRA_BASE_URL}/stream` shows a HUD overlay for demo purposes. Currently it uses scripted demo chains. To wire it to real data:

1. Forward the SSE step events from `/api/chat` to the `/stream` page via WebSocket or shared state
2. Or: have the Android app POST to a relay endpoint that the `/stream` page subscribes to
3. Simplest for hackathon: have the `/stream` page also call `/api/chat` directly (it already has Web Speech API built in)

### 4c. Dashboard updates automatically

The dashboard at `{MIRA_BASE_URL}/dashboard` subscribes to Supabase Realtime on the `events` table. All chat messages, object searches, and escalations appear automatically — no extra wiring needed.

---

## Step 5: Object Finding Flow (end-to-end)

When the LLM decides to search for an object, the chat API handles everything inline:

1. Chat API calls `find_object` tool → creates `object_requests` row (status: PENDING)
2. Chat API emits `FIND_OBJECT_REQUESTED` event
3. Chat API connects to the **Explorer 3D viewer** via WebSocket (`ws://localhost:8080/ws/chat`)
4. Sends `{"message": "find the pill organizer"}` to the explorer
5. Explorer scans the 3D point cloud scene, sends status updates back:
   - `{"role": "status", "content": "Scanning the scene..."}`
   - `{"role": "status", "content": "Asking AI to find 'pill organizer'..."}`
   - `{"role": "status", "content": "Triangulating 3D position..."}`
   - `{"role": "status", "content": "Found it! Flying to object..."}`
6. These statuses stream to the device/stream UI as SSE step events in real time
7. Explorer returns final result: `{"role": "assistant", "content": "Found: pill organizer on the dresser (seen in 3 views)"}`
8. Chat API updates `object_requests` → FOUND or NOT_FOUND
9. LLM composes a natural response using the explorer's result
10. On the second monitor, the Explorer 3D viewer flies the camera to the object and shows a pulsing marker

**The explorer must be running** at `ws://localhost:8080/ws/chat` (configurable via `EXPLORER_WS_URL` env var). If it's not running, the chat API gracefully falls back with a "could not connect" message and the LLM still responds.

```bash
# Start the explorer (from repo root):
cd explorer && python viewer.py /path/to/scene.glb --port 8080 --viser-port 8081
```

**Legacy `/api/objects/update` endpoint** still exists if you need to report object locations from an external CV pipeline, but the primary flow now uses the direct explorer integration above.

---

## Environment Variables

Server-side (`.env.local` on the Next.js app):

```bash
# Required
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
OPENROUTER_API_KEY=sk-or-v1-...        # LLM (gpt-4o-mini)
OPENAI_API_KEY=sk-...                   # Whisper STT
ELEVENLABS_API_KEY=...                  # TTS

# Optional
TWILIO_ACCOUNT_SID=...                  # SMS escalation
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1XXXXXXXXXX
TWILIO_ALERT_TO=+1XXXXXXXXXX
```

Android-side (if calling Whisper directly):
```
OPENAI_API_KEY=sk-...
MIRA_BASE_URL=https://your-deployed-url.vercel.app  # or local IP for dev
```

---

## Quick Test (curl)

```bash
BASE=http://localhost:3000

# 1. Transcribe audio
curl -X POST $BASE/api/voice/transcribe \
  -F "audio=@test.webm"

# 2. Chat (SSE stream)
curl -N -X POST $BASE/api/chat \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"a1b2c3d4-0001-4000-8000-000000000001","message":"Where are my pills?"}'

# 3. TTS
curl -X POST $BASE/api/voice/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello Margaret"}' \
  --output test.mp3

# 4. Report object found (CV team)
curl -X POST $BASE/api/objects/update \
  -H "Content-Type: application/json" \
  -d '{"request_id":"<from-chat-result>","patient_id":"a1b2c3d4-0001-4000-8000-000000000001","object_name":"pill organizer","found":true,"location":"Garden table","confidence":0.94}'
```

---

## File Map (key files in the chat branch)

```
app/
├── api/
│   ├── chat/route.ts          # Main chat endpoint (SSE stream, function calling)
│   ├── voice/
│   │   ├── transcribe/route.ts # Whisper STT proxy
│   │   └── tts/route.ts        # ElevenLabs TTS
│   ├── objects/
│   │   ├── request/route.ts    # Create object search request
│   │   └── update/route.ts     # CV team reports result
│   ├── escalate/route.ts       # Manual caregiver escalation
│   ├── events/route.ts         # Event CRUD
│   └── patients/route.ts       # List patients
├── device/page.tsx             # Resident chat UI (web)
├── dashboard/page.tsx          # Supervisor dashboard
├── stream/page.tsx             # AR glasses HUD overlay (demo)
└── page.tsx                    # Landing page
lib/
├── event-spine.ts              # Event logging (appendEvent, getRecentEvents)
├── supabase-server.ts          # Server Supabase client
├── supabase/client.ts          # Browser Supabase client
└── twilio.ts                   # SMS sending
```
