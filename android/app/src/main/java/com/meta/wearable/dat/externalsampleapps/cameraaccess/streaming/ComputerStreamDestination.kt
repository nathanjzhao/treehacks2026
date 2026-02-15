/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ComputerStreamDestination - Binary HTTP Streaming
//
// Handles streaming to computer via HTTP POST with binary protocol.
// Uses binary protocol: [4-byte header length][JSON metadata][JPEG data]

package com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming

import android.graphics.Rect
import android.graphics.YuvImage
import android.graphics.ImageFormat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Semaphore
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class ComputerStreamDestination(
    private val endpoint: String,
    private val port: Int,
    private val targetFps: Int,
    private val jpegQuality: Int
) {
    private val TAG = "ComputerStream"

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .connectionPool(ConnectionPool(5, 30, TimeUnit.SECONDS))
        .retryOnConnectionFailure(false)
        .build()

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("UTC")
    }

    private var framesSent = AtomicInteger(0)
    private var connected = false

    init {
        android.util.Log.w(TAG, "🚀 ComputerStreamDestination initialized (Binary HTTP):")
        android.util.Log.w(TAG, "   Protocol: HTTP POST with binary frames")
        android.util.Log.w(TAG, "   Target FPS: $targetFps")
        android.util.Log.w(TAG, "   JPEG Quality: $jpegQuality%")
        android.util.Log.w(TAG, "   Endpoint: http://$endpoint:$port/api/stream/ws")
    }

    /**
     * Connect (mark as ready to send)
     */
    fun connect(scope: CoroutineScope): Result<Unit> {
        android.util.Log.i(TAG, "✅ Binary streaming ready")
        connected = true
        return Result.success(Unit)
    }

    /**
     * Disconnect
     */
    fun disconnect() {
        android.util.Log.i(TAG, "Disconnecting")
        connected = false
    }

    /**
     * Send a frame via HTTP POST with binary protocol
     * Note: Rate limiting is handled in VideoStreamingManager before this is called
     */
    suspend fun sendFrame(
        frameData: ByteArray,
        width: Int,
        height: Int,
        timestamp: Long,
        frameNumber: Int
    ): Result<Unit> {
        // Convert I420 to JPEG
        val jpegData = convertToJpeg(frameData, width, height)
        return sendJpegFrame(jpegData, width, height, timestamp, frameNumber)
    }

    /**
     * Send a pre-encoded JPEG frame via HTTP POST with binary protocol.
     * Use this when JPEG encoding is done once upstream and shared across destinations.
     */
    suspend fun sendJpegFrame(
        jpegData: ByteArray,
        width: Int,
        height: Int,
        timestamp: Long,
        frameNumber: Int
    ): Result<Unit> {
        val currentFrames = framesSent.incrementAndGet()

        StreamingLogger.debug(TAG, "Sending frame #$frameNumber (${jpegData.size} bytes)")

        return try {
            if (!connected) {
                return Result.failure(Exception("Not connected"))
            }

            // Build metadata JSON
            val metadata = buildString {
                append("{\"type\":\"frame\",")
                append("\"timestamp\":\"${dateFormat.format(Date(timestamp))}\",")
                append("\"frame_number\":$frameNumber,")
                append("\"width\":$width,\"height\":$height,")
                append("\"jpeg_size\":${jpegData.size}}")
            }

            // Wire format: [4-byte length] + [JSON] + [JPEG]
            val metadataBytes = metadata.toByteArray(Charsets.UTF_8)
            val wireFrame = ByteBuffer.allocate(4 + metadataBytes.size + jpegData.size)
                .putInt(metadataBytes.size)
                .put(metadataBytes)
                .put(jpegData)
                .array()

            // Send via HTTP POST
            val request = Request.Builder()
                .url("http://$endpoint:$port/api/stream/ws")
                .post(wireFrame.toRequestBody("application/octet-stream".toMediaType()))
                .build()

            val startTime = System.currentTimeMillis()
            val response = executeRequest(request)
            val latency = System.currentTimeMillis() - startTime

            if (response.isSuccessful) {
                // Log every 30th successful frame
                if (currentFrames % 30 == 0) {
                    android.util.Log.i(TAG, "📊 Binary HTTP: $currentFrames frames sent, Frame #$frameNumber, ${jpegData.size} bytes, ${latency}ms")
                }
                Result.success(Unit)
            } else {
                val error = "HTTP ${response.code}: ${response.message}"
                StreamingLogger.error(TAG, "Failed to send frame #$frameNumber: $error")
                Result.failure(Exception(error))
            }
        } catch (e: IOException) {
            StreamingLogger.error(TAG, "Network error sending frame #$frameNumber: ${e.message}")
            Result.failure(e)
        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Error sending frame #$frameNumber: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Execute HTTP request with coroutine support
     */
    private suspend fun executeRequest(request: Request): Response = suspendCoroutine { continuation ->
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                continuation.resumeWithException(e)
            }

            override fun onResponse(call: Call, response: Response) {
                continuation.resume(response)
            }
        })
    }

    /**
     * Convert I420 raw data to JPEG
     */
    private fun convertToJpeg(i420Data: ByteArray, width: Int, height: Int): ByteArray {
        // Convert I420 to NV21 (Android YuvImage format)
        val nv21 = convertI420toNV21(i420Data, width, height)

        // Compress to JPEG
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), jpegQuality, outputStream)

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

}
