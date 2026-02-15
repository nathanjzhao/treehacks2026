/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import ai.picovoice.porcupine.Porcupine
import com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingLogger
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Core orchestrator for dual-mode audio recording.
 *
 * Manages:
 * - Bluetooth SCO audio routing for Ray-Ban glasses microphone
 * - Continuous audio capture (16kHz mono PCM)
 * - Dual detection: Porcupine wake words + VAD for passive speech
 * - Recording modes: Query (wake word) vs Passive (any speech)
 * - Transcription via Whisper API
 * - Query handling via Mira chat API (with TTS)
 * - Passive context logging via WebSocket
 */
class AudioRecordingManager(
    private val context: Context,
    private val configuration: AudioConfiguration
) {
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // State management
    private val _state = MutableStateFlow<AudioCaptureState>(AudioCaptureState.Stopped)
    val state: StateFlow<AudioCaptureState> = _state.asStateFlow()

    private val _stats = MutableStateFlow(AudioStats())
    val stats: StateFlow<AudioStats> = _stats.asStateFlow()

    // Audio components
    private var bluetoothScoManager: BluetoothScoManager? = null
    private var audioRecord: AudioRecord? = null
    private var porcupine: Porcupine? = null

    // Services
    private var whisperService: WhisperTranscriptionService? = null
    private var miraChatService: MiraChatService? = null
    private var audioStreamDestination: AudioStreamDestination? = null

    // Recording state
    private var audioProcessingJob: Job? = null
    private var isRunning = false
    private val recordingBuffers = mutableListOf<ShortArray>()
    private var recordingStartTime = 0L
    private var silenceStartTime = 0L
    private var currentRecordingMode: RecordingMode? = null

    companion object {
        private const val TAG = "AudioRecordingManager"
        private const val SAMPLE_RATE = 16000 // Required by Porcupine
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val FRAME_SIZE = 512 // Porcupine frame size
        private const val USER_ID = "demo_user" // TODO: Get from user settings
    }

    /**
     * Initialize audio recording system.
     */
    fun initialize(): Boolean {
        try {
            StreamingLogger.info(TAG, "Initializing audio recording system")

            // Validate configuration
            if (configuration.porcupineAccessKey.isBlank()) {
                StreamingLogger.error(TAG, "Porcupine access key is required")
                return false
            }
            if (configuration.openaiApiKey.isBlank()) {
                StreamingLogger.error(TAG, "OpenAI API key is required")
                return false
            }

            // Initialize Bluetooth SCO
            bluetoothScoManager = BluetoothScoManager(context)
            if (!bluetoothScoManager!!.startBluetoothSco()) {
                StreamingLogger.error(TAG, "Failed to start Bluetooth SCO")
                return false
            }

            // Initialize AudioRecord
            val minBufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT
            )
            val bufferSize = minBufferSize * 4 // Larger buffer for stability

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )

            if (audioRecord!!.state != AudioRecord.STATE_INITIALIZED) {
                StreamingLogger.error(TAG, "Failed to initialize AudioRecord")
                return false
            }

            // Initialize Porcupine with custom wake words
            porcupine = initializePorcupine(configuration.porcupineAccessKey)
            if (porcupine == null) {
                StreamingLogger.error(TAG, "Failed to initialize Porcupine")
                return false
            }

            // Initialize services
            whisperService = WhisperTranscriptionService(configuration.openaiApiKey)
            miraChatService = MiraChatService(context, configuration.miraBaseUrl)

            // Initialize passive context destination
            if (configuration.passiveContextEnabled) {
                audioStreamDestination = AudioStreamDestination(USER_ID)
                audioStreamDestination?.connect()
            }

            StreamingLogger.info(TAG, "Audio recording system initialized")
            return true

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Initialization error: ${e.message}")
            return false
        }
    }

    /**
     * Initialize Porcupine wake word detector.
     */
    private fun initializePorcupine(accessKey: String): Porcupine? {
        return try {
            val assetManager = context.assets
            val wakeWordFiles = mutableListOf<String>()

            // Check for custom wake word files in assets
            try {
                val assetFiles = assetManager.list("") ?: emptyArray()
                if ("Hey-Mira_en_android_v4_0_0.ppn" in assetFiles) {
                    wakeWordFiles.add("Hey-Mira_en_android_v4_0_0.ppn")
                }
                // Optional: check for additional wake word variants
                if ("hey_mir_android.ppn" in assetFiles) {
                    wakeWordFiles.add("hey_mir_android.ppn")
                }
            } catch (e: Exception) {
                StreamingLogger.error(TAG, "Failed to list assets: ${e.message}")
            }

            val builder = Porcupine.Builder().setAccessKey(accessKey)

            if (wakeWordFiles.isNotEmpty()) {
                // Use custom wake word files
                StreamingLogger.info(TAG, "Using custom wake words: ${wakeWordFiles.joinToString()}")
                builder.setKeywordPaths(wakeWordFiles.toTypedArray())
                    .setSensitivities(FloatArray(wakeWordFiles.size) {
                        configuration.settings.wakeWordSensitivity
                    })
            } else {
                // Fallback to built-in keyword for testing
                StreamingLogger.info(TAG, "No custom wake words found, using built-in fallback")
                builder.setKeywords(arrayOf(Porcupine.BuiltInKeyword.HEY_SIRI))
                    .setSensitivities(floatArrayOf(configuration.settings.wakeWordSensitivity))
            }

            builder.build(context)
        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Failed to initialize Porcupine: ${e.message}")
            null
        }
    }

    /**
     * Start audio recording and processing.
     */
    fun start() {
        if (isRunning) {
            StreamingLogger.info(TAG, "Audio recording already running")
            return
        }

        try {
            audioRecord?.startRecording()
            isRunning = true
            _state.value = AudioCaptureState.Listening

            // Start audio processing loop
            audioProcessingJob = scope.launch {
                processAudioLoop()
            }

            StreamingLogger.info(TAG, "Audio recording started")

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Failed to start recording: ${e.message}")
            _state.value = AudioCaptureState.Error("Failed to start: ${e.message}")
        }
    }

    /**
     * Main audio processing loop with dual detection.
     */
    private suspend fun processAudioLoop() {
        val frame = ShortArray(FRAME_SIZE)
        val startTime = System.currentTimeMillis()

        while (isRunning) {
            try {
                // Read audio frame
                val samplesRead = audioRecord?.read(frame, 0, FRAME_SIZE) ?: 0
                if (samplesRead <= 0) {
                    delay(10)
                    continue
                }

                // Update uptime
                val uptimeSeconds = (System.currentTimeMillis() - startTime) / 1000
                _stats.value = _stats.value.copy(uptimeSeconds = uptimeSeconds)

                // Process frame based on current state
                when (_state.value) {
                    is AudioCaptureState.Listening -> {
                        // Dual detection: wake word + passive speech
                        handleListeningState(frame)
                    }
                    is AudioCaptureState.Recording -> {
                        // Continue recording until silence detected
                        handleRecordingState(frame)
                    }
                    else -> {
                        // Do nothing in other states
                    }
                }

            } catch (e: Exception) {
                StreamingLogger.error(TAG, "Audio processing error: ${e.message}")
                delay(100)
            }
        }
    }

    /**
     * Handle listening state: detect wake words and passive speech.
     */
    private suspend fun handleListeningState(frame: ShortArray) {
        // 1. Check for wake word (Porcupine)
        val keywordIndex = porcupine?.process(frame) ?: -1
        if (keywordIndex >= 0) {
            // Wake word detected -> Query mode
            val keyword = "hey Mira" // TODO: Get from keyword index
            StreamingLogger.info(TAG, "Wake word detected: $keyword (mode: QUERY)")

            _state.value = AudioCaptureState.WakeWordDetected(keyword)
            _stats.value = _stats.value.copy(
                wakeWordDetections = _stats.value.wakeWordDetections + 1,
                lastWakeWordTime = System.currentTimeMillis()
            )

            startRecording(RecordingMode.QUERY)
            return
        }

        // 2. Check for passive speech (VAD)
        if (configuration.passiveContextEnabled) {
            val rms = calculateRMS(frame)
            if (rms > configuration.settings.silenceThresholdRMS) {
                // Speech detected (no wake word) -> Passive mode
                StreamingLogger.debug(TAG, "Passive speech detected (mode: PASSIVE)")

                _state.value = AudioCaptureState.SpeechDetected()
                _stats.value = _stats.value.copy(
                    passiveContexts = _stats.value.passiveContexts + 1,
                    lastPassiveContextTime = System.currentTimeMillis()
                )

                startRecording(RecordingMode.PASSIVE)
            }
        }
    }

    /**
     * Handle recording state: accumulate audio until silence detected.
     */
    private suspend fun handleRecordingState(frame: ShortArray) {
        recordingBuffers.add(frame.copyOf())

        val recordingDuration = System.currentTimeMillis() - recordingStartTime
        _state.value = AudioCaptureState.Recording(
            mode = currentRecordingMode!!,
            durationMs = recordingDuration
        )

        // Check for silence
        val isSilent = detectSilence(frame)

        if (isSilent) {
            if (silenceStartTime == 0L) {
                silenceStartTime = System.currentTimeMillis()
            } else {
                val silenceDuration = System.currentTimeMillis() - silenceStartTime

                if (silenceDuration >= configuration.settings.silenceTimeoutMs) {
                    // Silence detected -> stop recording
                    if (recordingDuration >= configuration.settings.minRecordingDurationMs) {
                        stopRecording()
                    } else {
                        // Too short, discard
                        StreamingLogger.debug(TAG, "Recording too short, discarding")
                        resetRecording()
                        _state.value = AudioCaptureState.Listening
                    }
                }
            }
        } else {
            silenceStartTime = 0L // Reset silence timer
        }

        // Max duration guard
        if (recordingDuration >= configuration.settings.maxRecordingDurationMs) {
            StreamingLogger.info(TAG, "Max recording duration reached")
            stopRecording()
        }
    }

    /**
     * Start recording audio.
     */
    private fun startRecording(mode: RecordingMode) {
        recordingBuffers.clear()
        recordingStartTime = System.currentTimeMillis()
        silenceStartTime = 0L
        currentRecordingMode = mode

        _state.value = AudioCaptureState.Recording(mode, 0)
    }

    /**
     * Stop recording and process audio.
     */
    private suspend fun stopRecording() {
        val mode = currentRecordingMode ?: return
        val duration = System.currentTimeMillis() - recordingStartTime

        StreamingLogger.info(TAG, "Recording stopped (mode: $mode, duration: ${duration}ms)")

        // Convert buffers to PCM byte array
        val pcmData = convertBuffersToPcm(recordingBuffers)
        _state.value = AudioCaptureState.Transcribing(mode, pcmData.size)

        // Transcribe audio
        val transcriptionResult = whisperService?.transcribe(pcmData)

        if (transcriptionResult?.isSuccess == true) {
            val transcription = transcriptionResult.getOrNull()!!
            _stats.value = _stats.value.copy(
                transcriptionsCompleted = _stats.value.transcriptionsCompleted + 1
            )

            // Process based on mode
            when (mode) {
                RecordingMode.QUERY -> handleQueryResult(transcription)
                RecordingMode.PASSIVE -> handlePassiveContext(transcription)
            }
        } else {
            StreamingLogger.error(TAG, "Transcription failed: ${transcriptionResult?.exceptionOrNull()?.message}")
            _stats.value = _stats.value.copy(
                transcriptionsFailed = _stats.value.transcriptionsFailed + 1
            )
            resetRecording()
            _state.value = AudioCaptureState.Listening
        }
    }

    /**
     * Handle query result (wake word mode).
     */
    private suspend fun handleQueryResult(transcription: TranscriptionResult) {
        _state.value = AudioCaptureState.SendingQuery(
            transcription.text,
            transcription.confidence
        )

        // Send query to Mira
        val chatResult = miraChatService?.sendQuery(
            configuration.patientId,
            transcription.text
        )

        if (chatResult?.isSuccess == true) {
            val result = chatResult.getOrNull()!!
            _stats.value = _stats.value.copy(
                queriesSent = _stats.value.queriesSent + 1
            )

            // Play TTS response
            _state.value = AudioCaptureState.PlayingTTS(result.reply)
            miraChatService?.playTTS(result.reply)

        } else {
            StreamingLogger.error(TAG, "Query failed: ${chatResult?.exceptionOrNull()?.message}")
            _stats.value = _stats.value.copy(
                queriesFailed = _stats.value.queriesFailed + 1
            )
        }

        resetRecording()
        _state.value = AudioCaptureState.Listening
    }

    /**
     * Handle passive context (passive mode).
     */
    private suspend fun handlePassiveContext(transcription: TranscriptionResult) {
        _state.value = AudioCaptureState.SendingContext(
            transcription.text,
            transcription.confidence
        )

        // Send to memory backend (silent logging)
        audioStreamDestination?.sendPassiveContext(
            transcription.text,
            transcription.confidence,
            transcription.duration
        )

        resetRecording()
        _state.value = AudioCaptureState.Listening
    }

    /**
     * Reset recording state.
     */
    private fun resetRecording() {
        recordingBuffers.clear()
        recordingStartTime = 0L
        silenceStartTime = 0L
        currentRecordingMode = null
    }

    /**
     * Detect silence in audio frame using RMS threshold.
     */
    private fun detectSilence(frame: ShortArray): Boolean {
        val rms = calculateRMS(frame)
        return rms < configuration.settings.silenceThresholdRMS
    }

    /**
     * Calculate RMS (Root Mean Square) of audio frame.
     */
    private fun calculateRMS(frame: ShortArray): Double {
        return sqrt(frame.map { it.toDouble().pow(2) }.average())
    }

    /**
     * Convert recording buffers to PCM byte array.
     */
    private fun convertBuffersToPcm(buffers: List<ShortArray>): ByteArray {
        val totalSamples = buffers.sumOf { it.size }
        val pcmData = ByteArray(totalSamples * 2)
        var offset = 0

        for (buffer in buffers) {
            for (sample in buffer) {
                pcmData[offset++] = (sample.toInt() and 0xFF).toByte()
                pcmData[offset++] = ((sample.toInt() shr 8) and 0xFF).toByte()
            }
        }

        return pcmData
    }

    /**
     * Stop audio recording.
     */
    fun stop() {
        if (!isRunning) {
            return
        }

        try {
            isRunning = false
            audioProcessingJob?.cancel()

            audioRecord?.stop()
            _state.value = AudioCaptureState.Stopped

            StreamingLogger.info(TAG, "Audio recording stopped")

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Failed to stop recording: ${e.message}")
        }
    }

    /**
     * Clean up resources.
     */
    fun cleanup() {
        stop()

        audioRecord?.release()
        audioRecord = null

        porcupine?.delete()
        porcupine = null

        bluetoothScoManager?.cleanup()
        bluetoothScoManager = null

        audioStreamDestination?.cleanup()
        audioStreamDestination = null

        scope.cancel()

        StreamingLogger.info(TAG, "Audio recording manager cleaned up")
    }
}
