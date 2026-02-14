/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CloudStreamDestination - Cloud Backend Streaming
//
// Handles streaming to cloud backend via HTTP upload and WebSocket metadata.
// Matches iOS implementation pattern from GCPUploader.swift and MemoryCaptureWebSocketClient.swift

package com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming

import android.graphics.Rect
import android.graphics.YuvImage
import android.graphics.ImageFormat
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

@Serializable
data class UploadResponse(
    val url: String
)

@Serializable
data class WebSocketMessage(
    val type: String,
    val id: String,
    val timestamp: String,
    val photoURL: String,
    val audioURL: String? = null,
    val transcription: String = ""
)

class CloudStreamDestination(
    private val baseUrl: String,
    private val userId: String,
    private val captureIntervalMs: Long
) {
    private val TAG = "CloudStream"

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    private var websocket: WebSocket? = null
    private val json = Json { ignoreUnknownKeys = true }

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("UTC")
    }

    init {
        connectWebSocket()
    }

    /**
     * Send a frame to cloud backend
     */
    suspend fun sendFrame(
        frameData: ByteArray,
        width: Int,
        height: Int,
        timestamp: Long
    ): Result<String> {
        val captureId = UUID.randomUUID().toString()

        StreamingLogger.debug(TAG, "Capturing frame for cloud upload (captureId: $captureId)")

        return try {
            // Convert I420 to JPEG with higher quality for cloud storage
            val jpegData = convertToJpeg(frameData, width, height, quality = 80)

            // Step 1: Upload to backend
            val photoURL = uploadToBackend(captureId, jpegData).getOrThrow()

            // Step 2: Send WebSocket metadata
            sendWebSocketMessage(captureId, photoURL, timestamp)

            StreamingLogger.info(TAG, "Cloud upload successful: $photoURL")
            Result.success(photoURL)

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Cloud upload failed: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Upload JPEG to backend via HTTP POST
     */
    private suspend fun uploadToBackend(captureId: String, jpegData: ByteArray): Result<String> {
        return try {
            val requestBody = jpegData.toRequestBody("image/jpeg".toMediaType())

            val request = Request.Builder()
                .url("$baseUrl/upload/$captureId")
                .post(requestBody)
                .build()

            val response = executeRequest(request)

            if (response.isSuccessful) {
                val responseBody = response.body?.string() ?: throw Exception("Empty response body")
                val uploadResponse = json.decodeFromString<UploadResponse>(responseBody)
                StreamingLogger.info(TAG, "Upload successful for $captureId: ${uploadResponse.url}")
                Result.success(uploadResponse.url)
            } else {
                val error = "HTTP ${response.code}: ${response.message}"
                StreamingLogger.error(TAG, "Upload failed for $captureId: $error")
                Result.failure(Exception(error))
            }
        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Upload error for $captureId: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Send metadata via WebSocket
     */
    private fun sendWebSocketMessage(captureId: String, photoURL: String, timestamp: Long) {
        val timestampStr = dateFormat.format(Date(timestamp))

        val message = WebSocketMessage(
            type = "memory_capture",
            id = captureId,
            timestamp = timestampStr,
            photoURL = photoURL,
            audioURL = null,
            transcription = ""
        )

        val messageJson = json.encodeToString(message)

        val sent = websocket?.send(messageJson) ?: false
        if (sent) {
            StreamingLogger.debug(TAG, "WebSocket message sent for $captureId")
        } else {
            StreamingLogger.warning(TAG, "Failed to send WebSocket message for $captureId (websocket not connected)")
            // Try to reconnect
            connectWebSocket()
        }
    }

    /**
     * Connect to WebSocket endpoint
     */
    private fun connectWebSocket() {
        try {
            val wsUrl = baseUrl.replace("https://", "wss://").replace("http://", "ws://")
            val request = Request.Builder()
                .url("$wsUrl/ws/android/$userId")
                .build()

            websocket = httpClient.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    StreamingLogger.info(TAG, "WebSocket connected")
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    StreamingLogger.debug(TAG, "WebSocket message received: $text")
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    StreamingLogger.error(TAG, "WebSocket error: ${t.message}")
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    StreamingLogger.warning(TAG, "WebSocket closed: $reason")
                }
            })
        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Failed to connect WebSocket: ${e.message}")
        }
    }

    /**
     * Convert I420 raw data to JPEG
     */
    private fun convertToJpeg(i420Data: ByteArray, width: Int, height: Int, quality: Int): ByteArray {
        // Convert I420 to NV21 (Android YuvImage format)
        val nv21 = convertI420toNV21(i420Data, width, height)

        // Compress to JPEG
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), quality, outputStream)

        return outputStream.toByteArray()
    }

    /**
     * Convert I420 (YYYYYYYY:UUVV) to NV21 (YYYYYYYY:VUVU)
     */
    private fun convertI420toNV21(input: ByteArray, width: Int, height: Int): ByteArray {
        val output = ByteArray(input.size)
        val size = width * height
        val quarter = size / 4

        input.copyInto(output, 0, 0, size) // Y is the same

        for (n in 0 until quarter) {
            output[size + n * 2] = input[size + quarter + n] // V first
            output[size + n * 2 + 1] = input[size + n] // U second
        }
        return output
    }

    /**
     * Execute HTTP request with coroutine support
     */
    private suspend fun executeRequest(request: Request): Response = suspendCoroutine { continuation ->
        httpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                continuation.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                continuation.resume(response)
            }
        })
    }

    /**
     * Close WebSocket connection
     */
    fun close() {
        websocket?.close(1000, "Streaming disabled")
        websocket = null
        StreamingLogger.info(TAG, "WebSocket closed")
    }
}
