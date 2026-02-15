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
    private var cloudDestination: CloudStreamDestination? = null

    // Frame flow channels with overflow handling
    private val computerFrameFlow = MutableSharedFlow<FrameData>(
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
    private val cloudFrameCounter = AtomicInteger(0)
    private val lastComputerFrameCount = AtomicInteger(0)
    private val lastCloudFrameCount = AtomicInteger(0)

    // Bandwidth tracking
    private val computerBytesTransferred = AtomicLong(0)
    private val lastComputerBytes = AtomicLong(0)

    // Latency tracking (moving average)
    private val computerLatencySum = AtomicLong(0)
    private val computerLatencyCount = AtomicInteger(0)

    private val _statistics = MutableStateFlow(StreamingStats())
    val statistics: StateFlow<StreamingStats> = _statistics.asStateFlow()

    // Collection jobs
    private var computerJob: Job? = null
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

        // Update status
        _statistics.update { it.copy(computerStatus = ConnectionStatus.CONNECTING) }

        // Start collection with parallel/async uploads
        computerJob = coroutineScope.launch(Dispatchers.IO) {
            try {
                computerFrameFlow.collect { frameData ->
                    // Launch upload in parallel (don't block for response)
                    launch(Dispatchers.IO) {
                        val sendStartTime = System.currentTimeMillis()

                        computerDestination?.sendFrame(
                            frameData = frameData.data,
                            width = frameData.width,
                            height = frameData.height,
                            timestamp = frameData.timestamp,
                            frameNumber = frameData.frameNumber
                        )?.onSuccess {
                            val latency = System.currentTimeMillis() - sendStartTime

                            // Update counters
                            computerFrameCounter.incrementAndGet()
                            computerBytesTransferred.addAndGet(frameData.data.size.toLong())

                            // Track latency (simple moving average)
                            computerLatencySum.addAndGet(latency)
                            computerLatencyCount.incrementAndGet()

                            _statistics.update { it.copy(computerStatus = ConnectionStatus.CONNECTED) }
                        }?.onFailure { error ->
                            StreamingLogger.error(TAG, "Computer streaming error: ${error.message}")
                            _statistics.update { it.copy(computerStatus = ConnectionStatus.ERROR) }
                        }
                    }
                }
            } catch (e: Exception) {
                StreamingLogger.error(TAG, "Computer streaming job error: ${e.message}")
                _statistics.update { it.copy(computerStatus = ConnectionStatus.ERROR) }
            }
        }
    }

    /**
     * Disable computer streaming
     */
    fun disableComputerStreaming() {
        StreamingLogger.info(TAG, "Disabling computer streaming")
        computerJob?.cancel()
        computerJob = null
        computerDestination = null

        // Reset counters
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
     * Distribute a video frame to all enabled destinations
     */
    suspend fun distributeFrame(
        frameData: ByteArray,
        width: Int,
        height: Int,
        timestamp: Long
    ) {
        val frameNumber = frameCounter.incrementAndGet()

        StreamingLogger.debug(TAG, "Distributing frame #$frameNumber: ${width}x${height}, ${frameData.size} bytes")

        val frame = FrameData(
            data = frameData,
            width = width,
            height = height,
            timestamp = timestamp,
            frameNumber = frameNumber
        )

        // Emit to computer destination
        if (computerDestination != null) {
            val emitted = computerFrameFlow.tryEmit(frame)
            if (!emitted) {
                droppedFrameCounter.incrementAndGet()
                StreamingLogger.warning(TAG, "Dropped frame #$frameNumber for computer (buffer full)")
            } else {
                StreamingLogger.debug(TAG, "Frame #$frameNumber queued for computer")
            }
        }

        // Emit to cloud destination
        if (cloudDestination != null) {
            val emitted = cloudFrameFlow.tryEmit(frame)
            if (!emitted) {
                droppedFrameCounter.incrementAndGet()
                StreamingLogger.warning(TAG, "Dropped frame #$frameNumber for cloud (buffer full)")
            } else {
                StreamingLogger.debug(TAG, "Frame #$frameNumber queued for cloud")
            }
        }
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

                // Calculate FPS (frames sent in the last second)
                val currentComputerFrames = computerFrameCounter.get()
                val computerFps = (currentComputerFrames - lastComputerFrameCount.getAndSet(currentComputerFrames)).toFloat()

                val currentCloudFrames = cloudFrameCounter.get()
                val cloudFps = (currentCloudFrames - lastCloudFrameCount.getAndSet(currentCloudFrames)).toFloat()

                // Calculate bandwidth (KB/s)
                val currentBytes = computerBytesTransferred.get()
                val bytesPerSecond = currentBytes - lastComputerBytes.getAndSet(currentBytes)
                val bandwidthKBps = bytesPerSecond / 1024f

                // Calculate average latency
                val latencyCount = computerLatencyCount.getAndSet(0)
                val latencySum = computerLatencySum.getAndSet(0)
                val avgLatency = if (latencyCount > 0) latencySum / latencyCount else 0L

                _statistics.update { current ->
                    current.copy(
                        computerFps = computerFps,
                        cloudFps = cloudFps,
                        bandwidthKBps = bandwidthKBps,
                        droppedFrames = droppedFrameCounter.get(),
                        computerLatency = avgLatency,
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
        disableCloudStreaming()
        statsJob?.cancel()
    }
}
