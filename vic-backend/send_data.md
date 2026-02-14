# Send Data Documentation

This document describes how the iOS app sends memory captures to the backend.

---

## Overview

The iOS app sends data to the backend via two mechanisms:
1. **HTTP Upload** - `POST /upload/{capture_id}` for media files
2. **WebSocket** - `/ws/ios/{user_id}` for capture metadata and real-time sync

---

## Endpoints

### 1. File Upload: `POST /upload/{capture_id}`

Upload media files (photos, audio) to Cloud Storage.

**Request:**
```
POST /upload/{capture_id}
Content-Type: multipart/form-data

file: <binary file data>
```

**Response:**
```json
{
  "status": "success",
  "url": "https://storage.googleapis.com/reality-hack-2026-raw-media/memories/{capture_id}/photo.jpg",
  "captureId": "abc-123-def-456"
}
```

**Error Response:**
```json
{
  "status": "error",
  "error": "upload failed: <details>"
}
```

---

### 2. iOS WebSocket: `/ws/ios/{user_id}`

Real-time connection for sending capture metadata and receiving acknowledgments.

#### Connection
```
WebSocket: wss://{host}/ws/ios/{user_id}
```

#### Send: Memory Capture
```json
{
  "type": "memory_capture",
  "id": "abc-123-def-456",
  "timestamp": "2026-01-24T10:30:00-05:00",
  "photoURL": "https://storage.googleapis.com/reality-hack-2026-raw-media/memories/abc-123/photo.jpg",
  "audioURL": "https://storage.googleapis.com/reality-hack-2026-raw-media/memories/abc-123/audio.m4a",
  "transcription": "Just met with John at the coffee shop to discuss the project."
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Must be `"memory_capture"` |
| `id` | No | Unique capture ID (auto-generated if omitted) |
| `timestamp` | Yes | ISO 8601 timestamp |
| `photoURL` | No | URL to uploaded photo |
| `audioURL` | No | URL to uploaded audio |
| `transcription` | No | Transcribed text from audio |

#### Receive: Acknowledgment
```json
{
  "type": "ack",
  "status": "received",
  "captureId": "abc-123-def-456",
  "timestamp": "2026-01-24T15:30:00Z"
}
```

#### Receive: Processing Complete (broadcast)
```json
{
  "type": "memory_processed",
  "date": "2026-01-24",
  "captureId": "abc-123-def-456"
}
```

#### Error Response
```json
{
  "ok": false,
  "error": "failed_to_save",
  "detail": "error message"
}
```

---

## Complete Flow

```
iOS App                         Backend                      Cloud Storage
   |                               |                              |
   |-- POST /upload/{id} --------->|                              |
   |   (photo file)                |-- Upload to GCS ------------>|
   |                               |                              |
   |<-- {"status":"success",...} --|                              |
   |                               |                              |
   |== WebSocket /ws/ios/{user_id} ==|                            |
   |                               |                              |
   |-- {"type":"memory_capture"} ->|                              |
   |                               |-- Save to Firestore          |
   |<-- {"type":"ack",...} --------|                              |
   |                               |                              |
   |                               |-- [Async] Gemini Analysis    |
   |                               |-- [Async] Update Contacts    |
   |                               |-- [Async] Check Condensation |
   |                               |                              |
   |<-- {"type":"memory_processed"}|  (when analysis complete)    |
```

---

## Data Storage

After a capture is sent:
1. Media stored in `reality-hack-2026-raw-media` bucket
2. Metadata stored in Firestore `memory_captures` collection
3. Gemini analyzes the capture asynchronously
4. Contacts updated if names/faces detected
5. Hourly condensation triggered if 1+ hour since last

See `storing.md` for detailed schema information.
