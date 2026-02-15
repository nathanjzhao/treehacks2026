/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// VideoStreamingManager - Multi-Destination Streaming Orchestrator
//
// Orchestrates video frame distribution to multiple destinations:
// - Computer (for VGGT 3D reconstruction) via direct IP connection
// - Cloud backend (for storage and analysis) via HTTP/WebSocket
//
// Manages frame sampling, statistics tracking, and connection lifecycle.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming

import android.app.Application
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Frame data container for internal processing
 */
data class FrameData(
    val data: ByteArray,
    val width: Int,
    val height: Int,
    val timestamp: Long,
    val frameNumber: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as FrameData

        if (!data.contentEquals(other.data)) return false
        if (width != other.width) return false
        if (height != other.height) return false
        if (timestamp != other.timestamp) return false
        if (frameNumber != other.frameNumber) return false

        return true
    }

    override fun hashCode(): Int {
        var result = data.contentHashCode()
        result = 31 * result + width
        result = 31 * result + height
        result = 31 * result + timestamp.hashCode()
        result = 31 * result + frameNumber
        return result
    }
}

class VideoStreamingManager(
    private val context: Application,
    private val coroutineScope: CoroutineScope
) {
    private val TAG = "VideoStreamingManager"

    // Streaming destinations
    private var computerDestination: ComputerStreamDestination? = null
    private var computer2Destination: ComputerStreamDestination? = null
    private var cloudDestination: CloudStreamDestination? = null

    // Frame flow channels with overflow handling
    private val computerFrameFlow = MutableSharedFlow<FrameData>(
        replay = 0,
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    private val computer2FrameFlow = MutableSharedFlow<FrameData>(
        replay = 0,
        extraBufferCapacity = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    private val cloudFrameFlow = MutableSharedFlow<FrameData>(
        replay = 0,
        extraBufferCapacity = 3,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    // Statistics tracking
    private val frameCounter = AtomicInteger(0)
    private val droppedFrameCounter = AtomicInteger(0)
    private val startTime = AtomicLong(0)

    // Per-destination frame counters for FPS calculation
    private val computerFrameCounter = AtomicInteger(0)
    private val computer2FrameCounter = AtomicInteger(0)
    private val cloudFrameCounter = AtomicInteger(0)
    private val lastComputerFrameCount = AtomicInteger(0)
    private val lastComputer2FrameCount = AtomicInteger(0)
    private val lastCloudFrameCount = AtomicInteger(0)

    // Bandwidth tracking
    private val computerBytesTransferred = AtomicLong(0)
    private val lastComputerBytes = AtomicLong(0)
    private val computer2BytesTransferred = AtomicLong(0)
    private val lastComputer2Bytes = AtomicLong(0)

    // Latency tracking (moving average)
    private val computerLatencySum = AtomicLong(0)
    private val computerLatencyCount = AtomicInteger(0)
    private val computer2LatencySum = AtomicLong(0)
    private val computer2LatencyCount = AtomicInteger(0)

    // Concurrency control for uploads
    private var computerUploadSemaphore: Semaphore? = null
    private var computer2UploadSemaphore: Semaphore? = null

    // Cached JPEG quality for encode-once fan-out
    private var cachedJpegQuality: Int = 70

    private val _statistics = MutableStateFlow(StreamingStats())
    val statistics: StateFlow<StreamingStats> = _statistics.asStateFlow()

    // Collection jobs
    private var computerJob: Job? = null
    private var computer2Job: Job? = null
    private var cloudJob: Job? = null
    private var statsJob: Job? = null

    init {
        StreamingLogger.info(TAG, "VideoStreamingManager initialized")
        startStatisticsCollection()
    }

    /**
     * Enable computer streaming to specified endpoint
     */
    fun enableComputerStreaming(endpoint: String, port: Int, targetFps: Int = 7, jpegQuality: Int = 70) {
        StreamingLogger.info(TAG, "Enabling computer streaming: $endpoint:$port @ ${targetFps}fps, quality=${jpegQuality}%")
        android.util.Log.w(TAG, "⚠️ COMPUTER STREAMING CONFIG: targetFps=${targetFps}, jpegQuality=${jpegQuality}, endpoint=$endpoint:$port")

        cachedJpegQuality = jpegQuality

        // Cancel existing job if any
        computerJob?.cancel()

        // Create new destination
        computerDestination = ComputerStreamDestination(
            endpoint = endpoint,
            port = port,
            targetFps = targetFps,
            jpegQuality = jpegQuality
        )
        android.util.Log.i(TAG, "✅ ComputerStreamDestination created with targetFps=$targetFps")

        // Connect
        computerDestination?.connect(coroutineScope)?.onSuccess {
            android.util.Log.i(TAG, "✅ Computer 1 connection initiated")
        }?.onFailure { error ->
            android.util.Log.e(TAG, "❌ Computer 1 connection failed: ${error.message}")
        }

        // Update status
        _statistics.update { it.copy(computerStatus = ConnectionStatus.CONNECTING) }

        // Initialize rate limiting and concurrency control
        val frameIntervalMs = 1000L / targetFps
        val maxConcurrentUploads = 5
        computerUploadSemaphore = Semaphore(maxConcurrentUploads)

        android.util.Log.w(TAG, "⏱️ Rate limiting: Using .sample(${frameIntervalMs}ms) for ${targetFps} FPS")

        // Start collection — frames arrive pre-encoded as JPEG from distributeFrame()
        computerJob = coroutineScope.launch(Dispatchers.IO) {
            try {
                @OptIn(kotlinx.coroutines.FlowPreview::class)
                computerFrameFlow
                    .sample(frameIntervalMs)
                    .collect { frameData ->
                        launch(Dispatchers.IO) {
                            computerUploadSemaphore?.withPermit {
                                val sendStartTime = System.currentTimeMillis()

                                // frameData.data is already JPEG-encoded
                                computerDestination?.sendJpegFrame(
                                    jpegData = frameData.data,
                                    width = frameData.width,
                                    height = frameData.height,
                                    timestamp = frameData.timestamp,
                                    frameNumber = frameData.frameNumber
                                )?.onSuccess {
                                    val latency = System.currentTimeMillis() - sendStartTime
                                    computerFrameCounter.incrementAndGet()
                                    computerBytesTransferred.addAndGet(frameData.data.size.toLong())
                                    computerLatencySum.addAndGet(latency)
                                    computerLatencyCount.incrementAndGet()
                                    _statistics.update { it.copy(computerStatus = ConnectionStatus.CONNECTED) }
                                }?.onFailure { error ->
                                    StreamingLogger.error(TAG, "Computer 1 streaming error: ${error.message}")
                                    _statistics.update { it.copy(computerStatus = ConnectionStatus.ERROR) }
                                }
                            }
                        }
                    }
            } catch (e: Exception) {
                StreamingLogger.error(TAG, "Computer 1 streaming job error: ${e.message}")
                _statistics.update { it.copy(computerStatus = ConnectionStatus.ERROR) }
            }
        }
    }

    /**
     * Disable computer streaming
     */
    fun disableComputerStreaming() {
        StreamingLogger.info(TAG, "Disabling computer streaming")
        computerDestination?.disconnect()
        computerJob?.cancel()
        computerJob = null
        computerDestination = null
        computerUploadSemaphore = null

        computerFrameCounter.set(0)
        lastComputerFrameCount.set(0)
        computerBytesTransferred.set(0)
        lastComputerBytes.set(0)
        computerLatencySum.set(0)
        computerLatencyCount.set(0)

        _statistics.update { it.copy(
            computerStatus = ConnectionStatus.DISCONNECTED,
            computerFps = 0f,
            computerLatency = 0
        ) }
    }

    /**
     * Enable computer 2 streaming to specified endpoint
     */
    fun enableComputer2Streaming(endpoint: String, port: Int, targetFps: Int = 7, jpegQuality: Int = 70) {
        StreamingLogger.info(TAG, "Enabling computer 2 streaming: $endpoint:$port @ ${targetFps}fps, quality=${jpegQuality}%")
        android.util.Log.w(TAG, "⚠️ COMPUTER 2 STREAMING CONFIG: targetFps=${targetFps}, jpegQuality=${jpegQuality}, endpoint=$endpoint:$port")

        cachedJpegQuality = jpegQuality

        computer2Job?.cancel()

        computer2Destination = ComputerStreamDestination(
            endpoint = endpoint,
            port = port,
            targetFps = targetFps,
            jpegQuality = jpegQuality
        )
        android.util.Log.i(TAG, "✅ ComputerStreamDestination 2 created with targetFps=$targetFps")

        computer2Destination?.connect(coroutineScope)?.onSuccess {
            android.util.Log.i(TAG, "✅ Computer 2 connection initiated")
        }?.onFailure { error ->
            android.util.Log.e(TAG, "❌ Computer 2 connection failed: ${error.message}")
        }

        _statistics.update { it.copy(computer2Status = ConnectionStatus.CONNECTING) }

        val frameIntervalMs = 1000L / targetFps
        val maxConcurrentUploads = 5
        computer2UploadSemaphore = Semaphore(maxConcurrentUploads)

        computer2Job = coroutineScope.launch(Dispatchers.IO) {
            try {
                @OptIn(kotlinx.coroutines.FlowPreview::class)
                computer2FrameFlow
                    .sample(frameIntervalMs)
                    .collect { frameData ->
                        launch(Dispatchers.IO) {
                            computer2UploadSemaphore?.withPermit {
                                val sendStartTime = System.currentTimeMillis()

                                computer2Destination?.sendJpegFrame(
                                    jpegData = frameData.data,
                                    width = frameData.width,
                                    height = frameData.height,
                                    timestamp = frameData.timestamp,
                                    frameNumber = frameData.frameNumber
                                )?.onSuccess {
                                    val latency = System.currentTimeMillis() - sendStartTime
                                    computer2FrameCounter.incrementAndGet()
                                    computer2BytesTransferred.addAndGet(frameData.data.size.toLong())
                                    computer2LatencySum.addAndGet(latency)
                                    computer2LatencyCount.incrementAndGet()
                                    _statistics.update { it.copy(computer2Status = ConnectionStatus.CONNECTED) }
                                }?.onFailure { error ->
                                    StreamingLogger.error(TAG, "Computer 2 streaming error: ${error.message}")
                                    _statistics.update { it.copy(computer2Status = ConnectionStatus.ERROR) }
                                }
                            }
                        }
                    }
            } catch (e: Exception) {
                StreamingLogger.error(TAG, "Computer 2 streaming job error: ${e.message}")
                _statistics.update { it.copy(computer2Status = ConnectionStatus.ERROR) }
            }
        }
    }

    /**
     * Disable computer 2 streaming
     */
    fun disableComputer2Streaming() {
        StreamingLogger.info(TAG, "Disabling computer 2 streaming")
        computer2Destination?.disconnect()
        computer2Job?.cancel()
        computer2Job = null
        computer2Destination = null
        computer2UploadSemaphore = null

        computer2FrameCounter.set(0)
        lastComputer2FrameCount.set(0)
        computer2BytesTransferred.set(0)
        lastComputer2Bytes.set(0)
        computer2LatencySum.set(0)
        computer2LatencyCount.set(0)

        _statistics.update { it.copy(
            computer2Status = ConnectionStatus.DISCONNECTED,
            computer2Fps = 0f,
            computer2Latency = 0
        ) }
    }

    /**
     * Enable cloud streaming for specified user
     */
    fun enableCloudStreaming(userId: String, baseUrl: String = "https://memory-backend-328251955578.us-east1.run.app") {
        StreamingLogger.info(TAG, "Enabling cloud streaming: userId=$userId")

        // Cancel existing job if any
        cloudJob?.cancel()

        // Create new destination
        cloudDestination = CloudStreamDestination(
            baseUrl = baseUrl,
            userId = userId,
            captureIntervalMs = 5000L
        )

        // Update status
        _statistics.update { it.copy(cloudStatus = ConnectionStatus.CONNECTING) }

        // Start collection with sampling
        cloudJob = coroutineScope.launch(Dispatchers.IO) {
            try {
                @OptIn(kotlinx.coroutines.FlowPreview::class)
                cloudFrameFlow
                    .sample(5000L) // Sample every 5 seconds
                    .collect { frameData ->
                        cloudDestination?.sendFrame(
                            frameData = frameData.data,
                            width = frameData.width,
                            height = frameData.height,
                            timestamp = frameData.timestamp
                        )?.onSuccess {
                            cloudFrameCounter.incrementAndGet()
                            _statistics.update { it.copy(cloudStatus = ConnectionStatus.CONNECTED) }
                        }?.onFailure { error ->
                            StreamingLogger.error(TAG, "Cloud streaming error: ${error.message}")
                            _statistics.update { it.copy(cloudStatus = ConnectionStatus.ERROR) }
                        }
                    }
            } catch (e: Exception) {
                StreamingLogger.error(TAG, "Cloud streaming job error: ${e.message}")
                _statistics.update { it.copy(cloudStatus = ConnectionStatus.ERROR) }
            }
        }
    }

    /**
     * Disable cloud streaming
     */
    fun disableCloudStreaming() {
        StreamingLogger.info(TAG, "Disabling cloud streaming")
        cloudJob?.cancel()
        cloudJob = null
        cloudDestination?.close()
        cloudDestination = null
        _statistics.update { it.copy(cloudStatus = ConnectionStatus.DISCONNECTED, cloudFps = 0f) }
    }

    /**
     * Distribute a video frame to all enabled destinations.
     * JPEG encoding happens once and is shared by both computer destinations.
     */
    suspend fun distributeFrame(
        frameData: ByteArray,
        width: Int,
        height: Int,
        timestamp: Long
    ) {
        val frameNumber = frameCounter.incrementAndGet()

        StreamingLogger.debug(TAG, "Distributing frame #$frameNumber: ${width}x${height}, ${frameData.size} bytes")

        // Encode JPEG once for both computer destinations
        val needsJpeg = computerDestination != null || computer2Destination != null
        val jpegData = if (needsJpeg) encodeJpeg(frameData, width, height, cachedJpegQuality) else null

        // Emit pre-encoded JPEG to computer 1
        if (computerDestination != null && jpegData != null) {
            val jpegFrame = FrameData(
                data = jpegData,
                width = width,
                height = height,
                timestamp = timestamp,
                frameNumber = frameNumber
            )
            val emitted = computerFrameFlow.tryEmit(jpegFrame)
            if (!emitted) {
                droppedFrameCounter.incrementAndGet()
                StreamingLogger.warning(TAG, "Dropped frame #$frameNumber for computer 1 (buffer full)")
            }
        }

        // Emit same pre-encoded JPEG to computer 2
        if (computer2Destination != null && jpegData != null) {
            val jpegFrame = FrameData(
                data = jpegData,
                width = width,
                height = height,
                timestamp = timestamp,
                frameNumber = frameNumber
            )
            val emitted = computer2FrameFlow.tryEmit(jpegFrame)
            if (!emitted) {
                droppedFrameCounter.incrementAndGet()
                StreamingLogger.warning(TAG, "Dropped frame #$frameNumber for computer 2 (buffer full)")
            }
        }

        // Emit raw I420 to cloud destination (it does its own JPEG encoding at higher quality)
        if (cloudDestination != null) {
            val rawFrame = FrameData(
                data = frameData,
                width = width,
                height = height,
                timestamp = timestamp,
                frameNumber = frameNumber
            )
            val emitted = cloudFrameFlow.tryEmit(rawFrame)
            if (!emitted) {
                droppedFrameCounter.incrementAndGet()
                StreamingLogger.warning(TAG, "Dropped frame #$frameNumber for cloud (buffer full)")
            }
        }
    }

    /**
     * Convert I420 raw data to JPEG (encode once, share across destinations)
     */
    private fun encodeJpeg(i420Data: ByteArray, width: Int, height: Int, quality: Int): ByteArray {
        val nv21 = convertI420toNV21(i420Data, width, height)
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
        input.copyInto(output, 0, 0, size)
        for (n in 0 until quarter) {
            output[size + n * 2] = input[size + quarter + n]
            output[size + n * 2 + 1] = input[size + n]
        }
        return output
    }

    /**
     * Start statistics collection job
     */
    private fun startStatisticsCollection() {
        startTime.set(System.currentTimeMillis())

        statsJob = coroutineScope.launch {
            while (isActive) {
                delay(1000) // Update stats every second

                val uptime = (System.currentTimeMillis() - startTime.get()) / 1000

                // Computer 1 FPS
                val currentComputerFrames = computerFrameCounter.get()
                val computerFps = (currentComputerFrames - lastComputerFrameCount.getAndSet(currentComputerFrames)).toFloat()

                // Computer 2 FPS
                val currentComputer2Frames = computer2FrameCounter.get()
                val computer2Fps = (currentComputer2Frames - lastComputer2FrameCount.getAndSet(currentComputer2Frames)).toFloat()

                // Cloud FPS
                val currentCloudFrames = cloudFrameCounter.get()
                val cloudFps = (currentCloudFrames - lastCloudFrameCount.getAndSet(currentCloudFrames)).toFloat()

                // Bandwidth (KB/s) — sum of both computers
                val currentBytes = computerBytesTransferred.get()
                val bytesPerSecond1 = currentBytes - lastComputerBytes.getAndSet(currentBytes)
                val currentBytes2 = computer2BytesTransferred.get()
                val bytesPerSecond2 = currentBytes2 - lastComputer2Bytes.getAndSet(currentBytes2)
                val bandwidthKBps = (bytesPerSecond1 + bytesPerSecond2) / 1024f

                // Computer 1 average latency
                val latencyCount = computerLatencyCount.getAndSet(0)
                val latencySum = computerLatencySum.getAndSet(0)
                val avgLatency = if (latencyCount > 0) latencySum / latencyCount else 0L

                // Computer 2 average latency
                val latency2Count = computer2LatencyCount.getAndSet(0)
                val latency2Sum = computer2LatencySum.getAndSet(0)
                val avgLatency2 = if (latency2Count > 0) latency2Sum / latency2Count else 0L

                _statistics.update { current ->
                    current.copy(
                        computerFps = computerFps,
                        computer2Fps = computer2Fps,
                        cloudFps = cloudFps,
                        bandwidthKBps = bandwidthKBps,
                        droppedFrames = droppedFrameCounter.get(),
                        computerLatency = avgLatency,
                        computer2Latency = avgLatency2,
                        uptimeSeconds = uptime
                    )
                }
            }
        }
    }

    /**
     * Clean up resources
     */
    fun cleanup() {
        StreamingLogger.info(TAG, "Cleaning up VideoStreamingManager")
        disableComputerStreaming()
        disableComputer2Streaming()
        disableCloudStreaming()
        statsJob?.cancel()
    }
}
