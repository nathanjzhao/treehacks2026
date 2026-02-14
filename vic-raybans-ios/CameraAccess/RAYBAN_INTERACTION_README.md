# 🕶️ Ray-Ban Meta AI Companion: Interaction Guide

This project transforms Ray-Ban Meta Smart Glasses into a **proactive memory assistant** for person-centered care. By combining real-time computer vision, spatial awareness (Geofencing), and a Gemini-powered memory backend, we provide a "second brain" that assists users through their glasses' audio and camera.

---

## 🎮 Core Interaction Framework

The system utilizes a balanced "Active vs. Passive" interaction model:

### 1. 🎤 Active Querying (The "Second Brain")
*   **Interaction**: Long-press the central Query Button on the mobile interface.
*   **Workflow**:
    1.  **Capture**: The app takes a snapshot from the glasses' POV.
    2.  **Transcription**: Local on-device STT captures your question (e.g., "Who am I talking to?" or "Where are my glasses?").
    3.  **Context Injection**: The app automatically attaches your **Current GPS Coordinates** and **Visual Snapshot** to the query.
    4.  **AI Response**: The backend processes the memory history and streams an answer back.
    5.  **Multi-Modal Feedback**: The answer is read aloud into the glasses' speakers via high-quality TTS while appearing as a **Cyan HUD Caption** on the phone for caregivers to see.

### 2. 📍 Spatial Memory (Geofencing)
*   **Interaction**: Tap the Map Pin icon (📍) to "Tag" a location.
*   **Workflow**:
    1.  **Drop Pin**: Marks your current precise GPS coordinates (accurate to ~15-20 meters).
    2.  **Visual Confirmation**: A temporary "Updated Location" HUD appears with your exact Lat/Lon.
    3.  **Passive Monitoring**: The app creates an invisible 20m "Memory Zone".
    4.  **Audio Triggers**: 
        *   **Entering**: "Welcome back to your saved location."
        *   **Exiting**: An urgent alert sounds: "You have left your saved location. Did you forget your keys?"
*   **Reset**: Use the 🗑️ Trash icon to clear all active geofenced memories.

---

## 📱 Visual HUD (Heads-Up Display)

Since the user is wearing the glasses, the phone acts as a **Companion Dashboard** for caregivers or for the user's reference:

| Indicator | Meaning |
| :--- | :--- |
| **White Text (Bottom)** | Live transcription of what the wearer is saying. |
| **Cyan Bubble (Bottom)** | Proactive AI responses or Location Status. |
| **Black Bubble (Top)** | Connection & Streaming status. |
| **Red Pulse** | Recording state (Glasses are actively listening/capturing). |

---

## 🛠️ Technical Implementation Details

### Spatial Awareness (`GeofenceManager.swift`)
The system uses `CLCircularRegion` with a specialized **20-meter radius** (optimized for hackathon indoor/small space demonstration). It utilizes `CLLocationManager` with "Always" permissions to ensure background tracking even if the phone screen is off.

### Memory WebSocket (`QueryWebSocketClient.swift`)
Communication is handled via a high-speed WebSocket that supports:
*   **Base64 Image Uploads**: For visual verification.
*   **GPS Enrichment**: Every text query includes `latitude` and `longitude`.
*   **Clarification Loops**: If the AI needs more info, it prompts a "Clarification Needed" UI state.

### Audio Pipeline (`TTSManager.swift`)
Uses `AVSpeechSynthesizer` configured to route audio directly to the Meta Wearables Bluetooth channel, providing a seamless "voice in your ear" experience.

---

## 🚀 How to Demo
1.  **Connect**: Power on the Ray-Ban Meta glasses and pair through the "Connect" screen.
2.  **Tag**: Walk to a specific point (e.g., the "Home" area) and tap 📍.
3.  **Leave**: Walk 20 meters away. Wait for the glasses to say "You have left your saved location."
4.  **Query**: Hold the Query button and ask "Where am I?" — The AI will use the newly synced GPS data to provide context.

---
*Created for RealityHacks 2026 - Empowering Seniors through Wearable AI.*
