# Integration Guide: Meta Ray-Ban Gen 3 → Mira

How to connect the Ray-Ban glasses to the Mira backend for voice-driven interactions.

---

## Architecture

```
Ray-Ban Gen 3 (mic)
    │ Bluetooth audio
    ▼
Android Phone
    │ captures audio, sends to Whisper
    ▼
OpenAI Whisper API ──→ transcript text
    │
    ▼
Mira Backend (POST /api/chat)
    │ SSE stream with steps + reply
    │
    ├──→ Android app (parse SSE, get reply text)
    │        │
    │        ▼
    │    POST /api/voice/tts ──→ audio bytes ──→ glasses speaker
    │
    ├──→ /stream page (HUD overlay, mirrors chat via Supabase Realtime)
    │
    ├──→ /dashboard (auto-updates via Supabase Realtime)
    │
    └──→ Explorer 3D viewer (triggered automatically on find_object)
             └──→ second monitor shows camera flying to object
```

**3 network hops:** Android → Whisper API, Android → Mira `/api/chat`, Android → Mira `/api/voice/tts`

---

## Step 1: Capture Audio from Ray-Ban Gen 3

The glasses stream audio to the phone over Bluetooth. On Android, the glasses mic shows up as a Bluetooth SCO audio source.

### Standard Android AudioRecord

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

### Voice Activity Detection (silence auto-stop)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SILENCE_THRESHOLD` | 8 | RMS below this = silence (0-128 scale) |
| `SILENCE_DURATION_MS` | 2200 | 2.2s continuous silence → auto-stop |
| `MIN_SPEECH_DURATION_MS` | 2000 | Must speak 2s before silence detection starts |
| `MAX_RECORDING_MS` | 30000 | 30s hard cap |

---

## Step 2: Transcribe with Whisper (on Android)

The Android app calls OpenAI Whisper directly, then sends the transcript to Mira.

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
// Now send transcribedText to Mira's /api/chat (Step 3)
```

---

## Step 3: Send Transcript to Mira Chat API

POST the transcript text to `/api/chat`. The response is an **SSE (Server-Sent Events) stream** with real-time processing steps, streaming text, then a final result.

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

The response is `Content-Type: text/event-stream`. Each event is a line:

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

// If find_object triggered, explorer sub-steps appear here:
data: {"type":"step","index":5,"label":"Scanning the scene...","status":"active"}
data: {"type":"step_done","index":5}
data: {"type":"step","index":6,"label":"Asking AI to find 'pill organizer'...","status":"active"}
data: {"type":"step_done","index":6}
data: {"type":"step","index":7,"label":"Triangulating 3D position...","status":"active"}
data: {"type":"step_done","index":7}
data: {"type":"step","index":8,"label":"Found it! Flying to object...","status":"active"}
data: {"type":"step_done","index":8}

data: {"type":"step","index":9,"label":"Composing response","status":"active"}

// Streaming text chunks of the reply (arrive as the LLM generates):
data: {"type":"text","chunk":"I found your "}
data: {"type":"text","chunk":"pill organizer! "}
data: {"type":"text","chunk":"It's on the dresser."}

data: {"type":"step_done","index":9}

data: {"type":"result","ok":true,"reply":"I found your pill organizer! It's on the dresser.","action":"FIND_OBJECT","request_id":"uuid","object_name":"pill organizer","event_id":"uuid"}
```

### Event types

| type | Fields | Description |
|------|--------|-------------|
| `step` | `index`, `label`, `status`, `detail?`, `searches?` | Processing step started |
| `step_done` | `index` | Step completed |
| `text` | `chunk` | Streaming text chunk of the reply (concatenate all chunks = full reply) |
| `result` | `ok`, `reply`, `action`, `request_id?`, `object_name?`, `citations?`, `event_id` | Final response with complete reply |

### `action` values in result

| Action | Meaning | Extra fields |
|--------|---------|-------------|
| `ANSWER` | General Q&A or medical info response | `citations?` (if Perplexity Sonar was used) |
| `FIND_OBJECT` | Object found/searched in 3D scene | `request_id`, `object_name` |
| `ESCALATE` | Caregiver alerted via SMS | — |

### What the Android app needs to do

For the **minimum viable integration**, the Android app only needs to care about `result` events:

```kotlin
val client = OkHttpClient.Builder()
    .readTimeout(60, TimeUnit.SECONDS)  // explorer search can take up to 45s
    .build()

val body = """{"patient_id":"$patientId","message":"$transcribedText"}"""
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
                "result" -> {
                    val reply = json.getString("reply")
                    val action = json.getString("action")
                    // Send reply to TTS (Step 4)
                    playTTS(reply)
                }
            }
        }
    }

    override fun onFailure(call: Call, e: IOException) {
        Log.e("Mira", "Chat request failed", e)
    }
})
```

**Important:** Set a read timeout of at least 60 seconds. When `find_object` is triggered, the explorer 3D search can take 10-30 seconds before returning a result.

---

## Step 4: Play Response via TTS

Once you have `reply` from the `result` event, send it to the TTS endpoint:

```
POST {MIRA_BASE_URL}/api/voice/tts
Content-Type: application/json

{ "text": "I found your pill organizer on the dresser." }
```

**Response:** Raw `audio/mpeg` bytes (ElevenLabs, Alice voice). Play directly through the glasses speaker:

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
    setAudioAttributes(AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANT)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build())
    prepare()
    start()
}
```

---

## Step 5: What Happens Behind the Scenes

You don't need to implement any of this — it's handled by the Mira backend automatically. This is just for context.

### General Q&A / Medical Info
1. LLM detects intent → calls `lookup_medication` or `search_medical_info` (Perplexity Sonar)
2. Returns answer with optional citations

### Object Finding
1. LLM detects "find object" intent → calls `find_object` tool
2. Mira backend connects to the **Explorer 3D viewer** via WebSocket (`ws://localhost:8080/ws/chat`)
3. Explorer scans the 3D point cloud, sends status updates that stream to the device as SSE steps
4. Explorer returns result → LLM composes a natural response
5. On the second demo monitor, the Explorer flies the camera to the object with a pulsing marker

### Caregiver Escalation
1. LLM detects emergency → calls `escalate_to_caregiver`
2. SMS sent via Twilio to the caregiver
3. Event logged to dashboard

### Automatic UI Updates
- `/stream` page (HUD overlay on first monitor) mirrors the chat via Supabase Realtime — no wiring needed
- `/dashboard` page auto-updates from the events table — no wiring needed

---

## Environment Variables

**Android app needs:**
```
OPENAI_API_KEY=sk-...                   # For Whisper STT
MIRA_BASE_URL=http://<computer-ip>:3000 # Mira backend (local IP on same WiFi)
```

**Mira server (`.env.local`):**
```bash
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# LLM + Search
OPENROUTER_API_KEY=sk-or-v1-...        # gpt-4o-mini + Perplexity Sonar

# Voice
ELEVENLABS_API_KEY=...                  # TTS

# Escalation (optional)
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1XXXXXXXXXX
TWILIO_ALERT_TO=+1XXXXXXXXXX

# Explorer (optional, defaults to ws://localhost:8080/ws/chat)
EXPLORER_WS_URL=ws://localhost:8080/ws/chat
```

---

## Quick Test (curl)

```bash
BASE=http://localhost:3000

# Chat (SSE stream) — this is the main endpoint your app calls
curl -N -X POST $BASE/api/chat \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"a1b2c3d4-0001-4000-8000-000000000001","message":"Where are my pills?"}'

# TTS
curl -X POST $BASE/api/voice/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello Margaret"}' \
  --output test.mp3
```

---

## Summary: What the Android App Does

```
1. Record audio from Ray-Ban mic (Bluetooth SCO)
2. Send audio to OpenAI Whisper API → get transcript
3. POST transcript to MIRA_BASE_URL/api/chat → parse SSE, get reply from "result" event
4. POST reply text to MIRA_BASE_URL/api/voice/tts → get audio bytes → play on glasses speaker
```

That's it. Everything else (object finding, escalation, dashboard, stream page) is handled by the Mira backend.
