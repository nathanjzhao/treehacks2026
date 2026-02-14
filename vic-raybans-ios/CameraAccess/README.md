# 👓 Ray-Ban Meta: Memory Stream Frontend

This repository contains the iOS implementation for the Ray-Ban Meta Smart Glasses integration, built for **RealityHacks 2026**. It serves as a high-fidelity companion dashboard and processing hub that bridges the Meta Wearables SDK with a Gemini-powered memory backend.

---

## 🏗️ System Architecture

The frontend follows an event-driven **MVVM** pattern, optimized for low-latency multi-modal interactions (Vision + Voice).

### 1. Device Interfacing (Meta DAT)
Utilizes the **Meta Wearables Device Access Toolkit (DAT)** for secure pairing and data exfiltration:
*   **Media Streaming**: Leveraging `MWDATCamera` to handle RTSP-like H.264 video streams from the head-mounted camera.
*   **Asynchronous Capture**: Precise frame-by-frame capture triggered by user queries, ensuring the LLM receives the exact visual POV of the wearer.

### 2. Multi-Modal Query Pipeline
A duplex communication loop designed for near-instantaneous AI feedback:
*   **Input**: Synchronized capture of **Head-POV Image** and **Local STT (Speech-to-Text)** via Apple's `SFSpeechRecognizer`.
*   **Transport**: Custom WebSocket (`QueryWebSocketClient`) architecture sending JSON payloads:
    ```json
    {
      "text": "What is the name of the person in front of me?",
      "imageURL": "https://gcs-bucket/query_123.jpg",
      "latitude": 42.3601,
      "longitude": -71.0942
    }
    ```
*   **Output Loop**: Intelligent routing of responses. The backend can return either a text `answer` (read via local `AVSpeechSynthesizer`) or a base64-encoded `audio` blob for direct playback, bypassing TTS latency.

### 3. Voice UI (VUI) & HUD
*   **Bidirectional Audio**: Manages `AVAudioSession` categories to ensure the glasses' speakers remain high-priority for AI responses while maintaining background microphone sensitivity for continuous transcription.
*   **Contextual HUD**: A SwiftUI-based overlay that displays a "Caregiver View"—live transcriptions of the user's speech and cyan-coded overlays of the AI's internal reasoning/response.

---

## 🛠️ Technical Specifications

*   **SDK**: Meta Wearables Core + Camera Framework (2025/26 Beta).
*   **Networking**: WSS (WebSocket Secure) with auto-reconnection logic and GCS-bucket image upload pipeline.
*   **Voice Engine**: Combined `Speech` framework (STT) and `AVFoundation` (TTS) with specialized routing for wearable Bluetooth profiles.
*   **State Management**: `@StateObject` driven view models ensuring reactive UI updates across the query lifecycle (Listening -> Processing -> Responding).

---

## 🚧 Roadmap & Experimental Features

### Location-Based Context (In Progress)
While `CoreLocation` permissions and `GeofenceManager` patterns are scaffolded, full spatial-memory integration is under active development. Current focus is on:
*   **Coordinate Enrichment**: Attaching Geo-stamps to visual memories for spatial search ("Where was I when I saw X?").
*   **Proximity Triggers**: Developing a localized alert system for safe-zone departures (Experimental).

---

## 🚀 Deployment

1.  **Hardware**: Requires Ray-Ban Meta Smart Glasses in Developer Mode.
2.  **Configuration**: Set the `baseURL` in `QueryWebSocketClient.swift` to point to the memory-backend instance.
3.  **Build**: Use the `CameraAccess` scheme. Must be deployed to a physical device for camera API access.

---
*Technical Reference for RealityHacks 2026 - Team Enhance.*
