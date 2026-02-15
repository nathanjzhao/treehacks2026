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
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.time.Instant
import java.time.format.DateTimeFormatter
import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * WebSocket destination for sending passive context transcriptions to memory backend.
 *
 * Passive context is speech detected without wake words, used for silent logging
 * to build conversation context without interactive responses.
 */
class AudioStreamDestination(
    private val userId: String
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS) // No timeout for WebSocket
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    private var webSocket: WebSocket? = null
    private var isConnected = false

    companion object {
        private const val TAG = "AudioStreamDestination"
        private const val MEMORY_BACKEND_WS_URL = "wss://memory-backend-328251955578.us-east1.run.app/ws/android"
    }

    /**
     * Connect to memory backend WebSocket.
     */
    fun connect(): Boolean {
        if (isConnected) {
            StreamingLogger.info(TAG, "Already connected to memory backend")
            return true
        }

        try {
            val wsUrl = "$MEMORY_BACKEND_WS_URL/$userId"
            StreamingLogger.info(TAG, "Connecting to memory backend: $wsUrl")

            val request = Request.Builder()
                .url(wsUrl)
                .build()

            webSocket = client.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    isConnected = true
                    StreamingLogger.info(TAG, "WebSocket connected to memory backend")
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    StreamingLogger.debug(TAG, "Received message: $text")
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    isConnected = false
                    StreamingLogger.error(TAG, "WebSocket error: ${t.message}")
                }

                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                    StreamingLogger.info(TAG, "WebSocket closing: $code - $reason")
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    isConnected = false
                    StreamingLogger.info(TAG, "WebSocket closed: $code - $reason")
                }
            })

            return true

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Failed to connect: ${e.message}")
            return false
        }
    }

    /**
     * Send passive context transcription to memory backend.
     *
     * @param transcription Transcribed text
     * @param confidence Transcription confidence (0.0-1.0)
     * @param durationSeconds Audio duration in seconds
     * @return Result indicating success or failure
     */
    suspend fun sendPassiveContext(
        transcription: String,
        confidence: Double,
        durationSeconds: Double
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            if (!isConnected) {
                StreamingLogger.error(TAG, "Not connected to memory backend")
                return@withContext Result.failure(Exception("Not connected to memory backend"))
            }

            if (transcription.isBlank()) {
                StreamingLogger.debug(TAG, "Skipping empty transcription")
                return@withContext Result.success(Unit)
            }

            // Create passive context message
            val message = PassiveContextMessage(
                type = "passive_context",
                id = UUID.randomUUID().toString(),
                timestamp = Instant.now().toString(),
                transcription = transcription,
                confidence = confidence,
                durationSeconds = durationSeconds
            )

            val messageJson = json.encodeToString(message)

            // Send via WebSocket
            val success = webSocket?.send(messageJson) ?: false

            if (success) {
                StreamingLogger.info(
                    TAG,
                    "Sent passive context: \"$transcription\" (confidence: ${"%.2f".format(confidence * 100)}%)"
                )
                Result.success(Unit)
            } else {
                StreamingLogger.error(TAG, "Failed to send passive context")
                Result.failure(Exception("Failed to send message"))
            }

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Error sending passive context: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Disconnect from memory backend.
     */
    fun disconnect() {
        try {
            webSocket?.close(1000, "Client disconnecting")
            webSocket = null
            isConnected = false
            StreamingLogger.info(TAG, "Disconnected from memory backend")
        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Error disconnecting: ${e.message}")
        }
    }

    /**
     * Check if connected to memory backend.
     */
    fun isConnected(): Boolean = isConnected

    /**
     * Clean up resources.
     */
    fun cleanup() {
        disconnect()
        client.dispatcher.executorService.shutdown()
    }
}

/**
 * Passive context message format for memory backend.
 */
@Serializable
data class PassiveContextMessage(
    val type: String, // "passive_context"
    val id: String, // UUID
    val timestamp: String, // ISO 8601
    val transcription: String,
    val confidence: Double,
    val durationSeconds: Double
)
