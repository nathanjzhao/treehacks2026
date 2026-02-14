/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ComputerStreamDestination - Direct IP HTTP Streaming
//
// Handles streaming to computer via phone hotspot for VGGT 3D reconstruction.
// Implements frame rate limiting, JPEG compression, and connection health monitoring.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming

import android.graphics.Rect
import android.graphics.YuvImage
import android.graphics.ImageFormat
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

class ComputerStreamDestination(
    private val endpoint: String,
    private val port: Int,
    private val targetFps: Int,
    private val jpegQuality: Int
) {
    private val TAG = "ComputerStream"

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .writeTimeout(3, TimeUnit.SECONDS)
        .readTimeout(3, TimeUnit.SECONDS)
        .build()

    private var lastFrameTime = 0L
    private val frameIntervalMs = 1000L / targetFps

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("UTC")
    }

    /**
     * Send a frame to the computer endpoint
     */
    suspend fun sendFrame(
        frameData: ByteArray,
        width: Int,
        height: Int,
        timestamp: Long,
        frameNumber: Int
    ): Result<Unit> {
        // Frame rate limiting
        val now = System.currentTimeMillis()
        if (now - lastFrameTime < frameIntervalMs) {
            StreamingLogger.debug(TAG, "Skipping frame #$frameNumber (rate limiting)")
            return Result.success(Unit)
        }
        lastFrameTime = now

        StreamingLogger.debug(TAG, "Sending frame #$frameNumber (${frameData.size} bytes)")

        return try {
            // Convert I420 to JPEG
            val jpegData = convertToJpeg(frameData, width, height)

            // Build multipart request
            val request = buildMultipartRequest(jpegData, timestamp, frameNumber, width, height)

            // Send request
            val startTime = System.currentTimeMillis()
            val response = executeRequest(request)
            val latency = System.currentTimeMillis() - startTime

            if (response.isSuccessful) {
                StreamingLogger.info(
                    TAG,
                    "Frame #$frameNumber sent successfully (${response.code}, ${latency}ms, ${jpegData.size} bytes)"
                )
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
            StreamingLogger.error(TAG, "Unexpected error sending frame #$frameNumber: ${e.message}")
            Result.failure(e)
        }
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

    /**
     * Build multipart/form-data HTTP request
     */
    private fun buildMultipartRequest(
        jpegData: ByteArray,
        timestamp: Long,
        frameNumber: Int,
        width: Int,
        height: Int
    ): Request {
        val timestampStr = dateFormat.format(Date(timestamp))

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image",
                "frame.jpg",
                jpegData.toRequestBody("image/jpeg".toMediaType())
            )
            .addFormDataPart("timestamp", timestampStr)
            .addFormDataPart("frame_number", frameNumber.toString())
            .addFormDataPart("width", width.toString())
            .addFormDataPart("height", height.toString())
            .build()

        return Request.Builder()
            .url("http://$endpoint:$port/frame")
            .post(requestBody)
            .build()
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
}
