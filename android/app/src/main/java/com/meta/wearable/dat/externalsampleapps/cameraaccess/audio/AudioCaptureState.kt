/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

/**
 * Represents the current state of the audio recording system.
 *
 * State machine flow:
 * STOPPED → LISTENING
 *     ↓
 *     ├─→ WAKE_WORD_DETECTED → RECORDING (query) → TRANSCRIBING → SENDING_QUERY → LISTENING
 *     │
 *     └─→ SPEECH_DETECTED → RECORDING (passive) → TRANSCRIBING → SENDING_CONTEXT → LISTENING
 */
sealed class AudioCaptureState {
    /** Audio system is completely stopped */
    object Stopped : AudioCaptureState()

    /** Continuously listening for wake words and speech */
    object Listening : AudioCaptureState()

    /** Wake word detected, transitioning to query recording */
    data class WakeWordDetected(
        val keyword: String,
        val timestamp: Long = System.currentTimeMillis()
    ) : AudioCaptureState()

    /** Speech detected without wake word, transitioning to passive recording */
    data class SpeechDetected(
        val timestamp: Long = System.currentTimeMillis()
    ) : AudioCaptureState()

    /** Currently recording audio (query or passive mode) */
    data class Recording(
        val mode: RecordingMode,
        val durationMs: Long,
        val startTimestamp: Long = System.currentTimeMillis()
    ) : AudioCaptureState()

    /** Transcribing recorded audio via Whisper API */
    data class Transcribing(
        val mode: RecordingMode,
        val audioSizeBytes: Int
    ) : AudioCaptureState()

    /** Sending query to Mira chat API (wake word mode only) */
    data class SendingQuery(
        val transcription: String,
        val confidence: Double
    ) : AudioCaptureState()

    /** Sending passive context to memory backend (passive mode only) */
    data class SendingContext(
        val transcription: String,
        val confidence: Double
    ) : AudioCaptureState()

    /** Playing TTS response (wake word mode only) */
    data class PlayingTTS(
        val reply: String
    ) : AudioCaptureState()

    /** Error occurred during audio processing */
    data class Error(
        val message: String,
        val exception: Throwable? = null
    ) : AudioCaptureState()
}

/**
 * Recording mode for dual-mode operation
 */
enum class RecordingMode {
    /** Query mode: Wake word detected, send to Mira chat API */
    QUERY,

    /** Passive mode: Speech detected, send to memory backend */
    PASSIVE
}

/**
 * Statistics for audio recording system
 */
data class AudioStats(
    val wakeWordDetections: Int = 0,
    val passiveContexts: Int = 0,
    val transcriptionsCompleted: Int = 0,
    val transcriptionsFailed: Int = 0,
    val queriesSent: Int = 0,
    val queriesFailed: Int = 0,
    val averageRecordingDurationMs: Long = 0,
    val averageConfidence: Double = 0.0,
    val uptimeSeconds: Long = 0,
    val lastWakeWordTime: Long? = null,
    val lastPassiveContextTime: Long? = null
)
