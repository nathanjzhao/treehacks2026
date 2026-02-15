/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Service for transcribing audio using OpenAI Whisper API.
 */
class WhisperTranscriptionService(
    private val apiKey: String
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    companion object {
        private const val TAG = "WhisperTranscription"
        private const val WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    }

    /**
     * Transcribe audio data using Whisper API.
     *
     * @param audioData Raw PCM audio data (16kHz, mono, 16-bit)
     * @return Result containing transcription or error
     */
    suspend fun transcribe(audioData: ByteArray): Result<TranscriptionResult> = withContext(Dispatchers.IO) {
        try {
            StreamingLogger.info(TAG, "Starting transcription (${audioData.size} bytes)")

            // Convert PCM to WAV format
            val wavData = convertPcmToWav(audioData, sampleRate = 16000, channels = 1, bitsPerSample = 16)
            StreamingLogger.debug(TAG, "Converted to WAV format (${wavData.size} bytes)")

            // Create multipart request body
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    "audio.wav",
                    wavData.toRequestBody("audio/wav".toMediaType())
                )
                .addFormDataPart("model", "whisper-1")
                .addFormDataPart("language", "en")
                .addFormDataPart("response_format", "verbose_json")
                .build()

            // Create request
            val request = Request.Builder()
                .url(WHISPER_API_URL)
                .addHeader("Authorization", "Bearer $apiKey")
                .post(requestBody)
                .build()

            // Execute request
            val response = client.newCall(request).execute()

            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                StreamingLogger.error(TAG, "Transcription failed: HTTP ${response.code} - $errorBody")
                return@withContext Result.failure(
                    Exception("Transcription failed: HTTP ${response.code} - $errorBody")
                )
            }

            // Parse response
            val responseBody = response.body?.string()
                ?: return@withContext Result.failure(Exception("Empty response body"))

            val whisperResponse = json.decodeFromString<WhisperResponse>(responseBody)

            // Calculate confidence from segments
            val confidence = calculateConfidence(whisperResponse.segments)

            val result = TranscriptionResult(
                text = whisperResponse.text,
                duration = whisperResponse.duration ?: 0.0,
                confidence = confidence,
                language = whisperResponse.language ?: "en",
                segments = whisperResponse.segments?.map {
                    TranscriptionSegment(
                        text = it.text,
                        start = it.start,
                        end = it.end,
                        confidence = 1.0 - it.no_speech_prob
                    )
                } ?: emptyList()
            )

            StreamingLogger.info(
                TAG,
                "Transcription completed: \"${result.text}\" (confidence: ${"%.2f".format(confidence * 100)}%)"
            )

            Result.success(result)

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Transcription error: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Calculate average confidence from Whisper segments.
     */
    private fun calculateConfidence(segments: List<WhisperSegment>?): Double {
        if (segments.isNullOrEmpty()) {
            return 0.8 // Default confidence if no segments
        }

        // Confidence = 1.0 - no_speech_prob (averaged across segments)
        val avgNoSpeechProb = segments.map { it.no_speech_prob }.average()
        return (1.0 - avgNoSpeechProb).coerceIn(0.0, 1.0)
    }

    /**
     * Convert raw PCM audio data to WAV format.
     *
     * @param pcmData Raw PCM data (16-bit little-endian samples)
     * @param sampleRate Sample rate in Hz (e.g., 16000)
     * @param channels Number of channels (1 = mono, 2 = stereo)
     * @param bitsPerSample Bits per sample (typically 16)
     * @return WAV file data with proper header
     */
    private fun convertPcmToWav(
        pcmData: ByteArray,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ): ByteArray {
        val output = ByteArrayOutputStream()

        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val dataSize = pcmData.size
        val fileSize = 36 + dataSize

        // Write WAV header
        output.write("RIFF".toByteArray())
        output.write(intToLittleEndian(fileSize))
        output.write("WAVE".toByteArray())

        // Write fmt chunk
        output.write("fmt ".toByteArray())
        output.write(intToLittleEndian(16)) // Subchunk1Size (16 for PCM)
        output.write(shortToLittleEndian(1)) // AudioFormat (1 = PCM)
        output.write(shortToLittleEndian(channels))
        output.write(intToLittleEndian(sampleRate))
        output.write(intToLittleEndian(byteRate))
        output.write(shortToLittleEndian(blockAlign))
        output.write(shortToLittleEndian(bitsPerSample))

        // Write data chunk
        output.write("data".toByteArray())
        output.write(intToLittleEndian(dataSize))
        output.write(pcmData)

        return output.toByteArray()
    }

    /**
     * Convert int to little-endian byte array.
     */
    private fun intToLittleEndian(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte(),
            ((value shr 16) and 0xFF).toByte(),
            ((value shr 24) and 0xFF).toByte()
        )
    }

    /**
     * Convert short to little-endian byte array.
     */
    private fun shortToLittleEndian(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte()
        )
    }
}

/**
 * Whisper API response (verbose_json format).
 */
@Serializable
data class WhisperResponse(
    val text: String,
    val duration: Double? = null,
    val language: String? = null,
    val segments: List<WhisperSegment>? = null
)

/**
 * Whisper API segment.
 */
@Serializable
data class WhisperSegment(
    val text: String,
    val start: Double,
    val end: Double,
    val no_speech_prob: Double
)

/**
 * Transcription result with metadata.
 */
data class TranscriptionResult(
    val text: String,
    val duration: Double,
    val confidence: Double,
    val language: String,
    val segments: List<TranscriptionSegment>
)

/**
 * Transcription segment with confidence.
 */
data class TranscriptionSegment(
    val text: String,
    val start: Double,
    val end: Double,
    val confidence: Double
)
