/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Service for handling wake word queries via Mira chat API.
 *
 * Supports:
 * - Sending queries to /api/chat endpoint
 * - Parsing Server-Sent Events (SSE) response
 * - TTS playback via /api/voice/tts
 */
class MiraChatService(
    private val context: Context,
    private val miraBaseUrl: String
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS) // Longer timeout for explorer searches
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    companion object {
        private const val TAG = "MiraChatService"
    }

    /**
     * Send query to Mira chat API and parse SSE response.
     *
     * @param patientId Patient ID for context
     * @param message User query text
     * @return Result containing chat result or error
     */
    suspend fun sendQuery(
        patientId: String,
        message: String
    ): Result<MiraChatResult> = withContext(Dispatchers.IO) {
        try {
            val url = "$miraBaseUrl/api/chat"
            StreamingLogger.info(TAG, "=== SENDING QUERY TO MIRA ===")
            StreamingLogger.info(TAG, "URL: $url")
            StreamingLogger.info(TAG, "Patient ID: $patientId")
            StreamingLogger.info(TAG, "Message: \"$message\"")

            // Create request body
            val requestJson = """{"patient_id":"$patientId","message":"$message"}"""
            val requestBody = requestJson.toRequestBody("application/json".toMediaType())

            // Create request
            val request = Request.Builder()
                .url(url)
                .post(requestBody)
                .build()

            StreamingLogger.info(TAG, "Executing HTTP request...")

            // Execute request
            val response = client.newCall(request).execute()

            StreamingLogger.info(TAG, "Response received: HTTP ${response.code}")

            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                StreamingLogger.error(TAG, "=== QUERY FAILED ===")
                StreamingLogger.error(TAG, "HTTP ${response.code} - $errorBody")
                StreamingLogger.error(TAG, "URL: $url")
                return@withContext Result.failure(
                    Exception("Query failed: HTTP ${response.code} - $errorBody")
                )
            }

            // Parse SSE stream
            val source = response.body!!.source()
            var reply = ""
            var action = "ANSWER"
            var requestId: String? = null
            var objectName: String? = null
            val steps = mutableListOf<String>()

            while (!source.exhausted()) {
                val line = source.readUtf8Line() ?: break

                // Skip empty lines and non-data lines
                if (!line.startsWith("data: ")) continue

                val jsonStr = line.removePrefix("data: ")

                // Skip [DONE] marker
                if (jsonStr.trim() == "[DONE]") break

                try {
                    val jsonObj = json.decodeFromString<JsonObject>(jsonStr)
                    val type = jsonObj["type"]?.jsonPrimitive?.content ?: continue

                    when (type) {
                        "step" -> {
                            val label = jsonObj["label"]?.jsonPrimitive?.content
                            if (label != null) {
                                steps.add(label)
                                StreamingLogger.info(TAG, "Step: $label")
                            }
                        }
                        "text" -> {
                            val chunk = jsonObj["chunk"]?.jsonPrimitive?.content ?: ""
                            reply += chunk
                        }
                        "result" -> {
                            // Final result event
                            reply = jsonObj["reply"]?.jsonPrimitive?.content ?: reply
                            action = jsonObj["action"]?.jsonPrimitive?.content ?: "ANSWER"
                            requestId = jsonObj["request_id"]?.jsonPrimitive?.content
                            objectName = jsonObj["object_name"]?.jsonPrimitive?.content

                            StreamingLogger.info(TAG, "Result: action=$action, reply=\"$reply\"")
                            break
                        }
                    }
                } catch (e: Exception) {
                    StreamingLogger.error(TAG, "Failed to parse SSE line: $line - ${e.message}")
                }
            }

            if (reply.isEmpty()) {
                StreamingLogger.error(TAG, "No reply received from Mira after parsing SSE stream")
                return@withContext Result.failure(Exception("No reply received from Mira"))
            }

            val result = MiraChatResult(
                reply = reply,
                action = action,
                requestId = requestId,
                objectName = objectName,
                steps = steps
            )

            StreamingLogger.info(TAG, "=== QUERY SUCCESS ===")
            StreamingLogger.info(TAG, "Reply: \"$reply\"")
            StreamingLogger.info(TAG, "Action: $action")

            Result.success(result)

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "=== QUERY EXCEPTION ===")
            StreamingLogger.error(TAG, "Exception: ${e.javaClass.simpleName}")
            StreamingLogger.error(TAG, "Message: ${e.message}")
            StreamingLogger.error(TAG, "URL: $miraBaseUrl/api/chat")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Play TTS response on glasses speaker.
     *
     * @param reply Text to convert to speech
     * @return Result indicating success or failure
     */
    suspend fun playTTS(reply: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            StreamingLogger.info(TAG, "Requesting TTS for: \"$reply\"")

            // Create request body
            val requestJson = """{"text":"$reply"}"""
            val requestBody = requestJson.toRequestBody("application/json".toMediaType())

            // Create request
            val request = Request.Builder()
                .url("$miraBaseUrl/api/voice/tts")
                .post(requestBody)
                .build()

            // Execute request
            val response = client.newCall(request).execute()

            if (!response.isSuccessful) {
                StreamingLogger.error(TAG, "TTS failed: HTTP ${response.code}")
                return@withContext Result.failure(Exception("TTS failed: HTTP ${response.code}"))
            }

            // Get audio bytes
            val audioBytes = response.body!!.bytes()
            StreamingLogger.info(TAG, "Received TTS audio (${audioBytes.size} bytes)")

            // Write to temp file
            val tempFile = File.createTempFile("mira_tts_", ".mp3", context.cacheDir)
            tempFile.writeBytes(audioBytes)

            // Play via MediaPlayer
            val mediaPlayer = MediaPlayer().apply {
                setDataSource(tempFile.absolutePath)
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANT)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                prepare()
            }

            StreamingLogger.info(TAG, "Playing TTS audio...")
            mediaPlayer.start()

            // Wait for playback to complete
            while (mediaPlayer.isPlaying) {
                delay(100)
            }

            mediaPlayer.release()
            tempFile.delete()

            StreamingLogger.info(TAG, "TTS playback completed")
            Result.success(Unit)

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "TTS playback error: ${e.message}")
            Result.failure(e)
        }
    }
}

/**
 * Mira chat API request.
 */
@Serializable
data class MiraChatRequest(
    val patient_id: String,
    val message: String
)

/**
 * Mira chat API result.
 */
data class MiraChatResult(
    val reply: String,
    val action: String, // ANSWER, FIND_OBJECT, ESCALATE
    val requestId: String?,
    val objectName: String?,
    val steps: List<String> = emptyList()
)
