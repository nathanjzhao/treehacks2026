/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamingConfiguration - Multi-Destination Streaming Configuration
//
// This file defines the data classes for configuring multi-destination video streaming
// from Meta Ray-Ban glasses to both computer (for VGGT 3D reconstruction) and cloud backend.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming

import kotlinx.serialization.Serializable

/**
 * Computer endpoint configuration for direct IP streaming via Tailscale VPN
 */
@Serializable
data class ComputerEndpoint(
    val ip: String = "100.93.34.56", // Default: Tailscale computer IP
    val port: Int = 8080,
    val enabled: Boolean = false
)

/**
 * Cloud endpoint configuration for backend API streaming
 */
@Serializable
data class CloudEndpoint(
    val baseUrl: String = "https://memory-backend-328251955578.us-east1.run.app",
    val userId: String = "",
    val enabled: Boolean = false
)

/**
 * Streaming quality and performance settings
 */
@Serializable
data class StreamingSettings(
    val computerTargetFps: Int = 30, // 1-30 FPS range for streaming
    val sdkFrameRate: Int = 30, // SDK capture frame rate (15/24/30)
    val jpegQuality: Int = 70, // 25-100% JPEG compression quality
    val cloudCaptureIntervalMs: Long = 5000 // Capture every 5 seconds for cloud
)

/**
 * Complete streaming configuration
 */
@Serializable
data class StreamingConfiguration(
    val computer: ComputerEndpoint = ComputerEndpoint(),
    val computer2: ComputerEndpoint = ComputerEndpoint(),
    val cloud: CloudEndpoint = CloudEndpoint(),
    val settings: StreamingSettings = StreamingSettings()
)

/**
 * Real-time streaming statistics
 */
data class StreamingStats(
    val computerFps: Float = 0f,
    val computer2Fps: Float = 0f,
    val cloudFps: Float = 0f,
    val bandwidthKBps: Float = 0f,
    val droppedFrames: Int = 0,
    val computerLatency: Long = 0, // Average upload time in ms
    val computer2Latency: Long = 0, // Average upload time in ms
    val cloudLatency: Long = 0, // Average upload time in ms
    val computerStatus: ConnectionStatus = ConnectionStatus.DISCONNECTED,
    val computer2Status: ConnectionStatus = ConnectionStatus.DISCONNECTED,
    val cloudStatus: ConnectionStatus = ConnectionStatus.DISCONNECTED,
    val uptimeSeconds: Long = 0
)

/**
 * Connection status for each streaming destination
 */
enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    ERROR
}
