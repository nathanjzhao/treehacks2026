/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamingSettingsScreen - Multi-Destination Streaming Configuration UI
//
// Allows users to configure computer and cloud streaming endpoints,
// adjust quality settings, and test connections.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StreamingSettingsScreen(
    streamViewModel: StreamViewModel,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    val uiState by streamViewModel.uiState.collectAsStateWithLifecycle()
    val config = uiState.streamingConfiguration
    val scope = rememberCoroutineScope()

    // Local state for editing
    var computerIp by remember { mutableStateOf(config.computer.ip) }
    var computerPort by remember { mutableStateOf(config.computer.port.toString()) }
    var computerEnabled by remember { mutableStateOf(config.computer.enabled) }

    var computer2Ip by remember { mutableStateOf(config.computer2.ip) }
    var computer2Port by remember { mutableStateOf(config.computer2.port.toString()) }
    var computer2Enabled by remember { mutableStateOf(config.computer2.enabled) }

    var cloudUserId by remember { mutableStateOf(config.cloud.userId) }
    var cloudEnabled by remember { mutableStateOf(config.cloud.enabled) }

    var targetFps by remember { mutableStateOf(config.settings.computerTargetFps.toFloat()) }
    var sdkFrameRate by remember { mutableStateOf(config.settings.sdkFrameRate) }
    var jpegQuality by remember { mutableStateOf(config.settings.jpegQuality.toFloat()) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Streaming Settings") },
                navigationIcon = {
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Computer Streaming Section
            Card {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(
                                "Computer Streaming",
                                style = MaterialTheme.typography.titleMedium
                            )
                            // Real-time status indicator
                            uiState.streamingStats?.let { stats ->
                                Row(
                                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    // Status dot
                                    Box(
                                        modifier = Modifier
                                            .size(8.dp)
                                            .background(
                                                color = when (stats.computerStatus) {
                                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ->
                                                        MaterialTheme.colorScheme.primary
                                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTING ->
                                                        MaterialTheme.colorScheme.tertiary
                                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.ERROR ->
                                                        MaterialTheme.colorScheme.error
                                                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                                                },
                                                shape = MaterialTheme.shapes.small
                                            )
                                    )
                                    Text(
                                        when (stats.computerStatus) {
                                            com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ->
                                                "Connected · ${String.format("%.1f", stats.computerFps)} FPS · ${stats.computerLatency}ms"
                                            com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTING ->
                                                "Connecting..."
                                            com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.ERROR ->
                                                "Error - Check connection"
                                            else -> "Disconnected"
                                        },
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                        Switch(
                            checked = computerEnabled,
                            onCheckedChange = { enabled ->
                                computerEnabled = enabled
                                scope.launch {
                                    if (enabled && computerIp.isNotEmpty()) {
                                        streamViewModel.enableComputerStreaming(
                                            computerIp,
                                            computerPort.toIntOrNull() ?: 3000
                                        )
                                    } else {
                                        streamViewModel.disableComputerStreaming()
                                    }
                                }
                            }
                        )
                    }

                    Text(
                        "Stream video to web application via Tailscale network",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    OutlinedTextField(
                        value = computerIp,
                        onValueChange = { newIp ->
                            computerIp = newIp
                            // Auto-apply if streaming is enabled
                            if (computerEnabled && newIp.isNotEmpty()) {
                                scope.launch {
                                    streamViewModel.enableComputerStreaming(
                                        newIp,
                                        computerPort.toIntOrNull() ?: 3000
                                    )
                                }
                            }
                        },
                        label = { Text("Tailscale IP Address") },
                        placeholder = { Text("100.x.x.x") },
                        supportingText = { Text("Your computer's Tailscale IP") },
                        modifier = Modifier.fillMaxWidth()
                    )

                    OutlinedTextField(
                        value = computerPort,
                        onValueChange = { newPort ->
                            computerPort = newPort
                            // Auto-apply if streaming is enabled
                            if (computerEnabled && computerIp.isNotEmpty()) {
                                scope.launch {
                                    streamViewModel.enableComputerStreaming(
                                        computerIp,
                                        newPort.toIntOrNull() ?: 3000
                                    )
                                }
                            }
                        },
                        label = { Text("Port") },
                        placeholder = { Text("3000") },
                        supportingText = { Text("Next.js dev server port") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }

            // Computer 2 Streaming Section
            Card {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(
                                "Computer 2 Streaming",
                                style = MaterialTheme.typography.titleMedium
                            )
                            // Real-time status indicator
                            uiState.streamingStats?.let { stats ->
                                Row(
                                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .size(8.dp)
                                            .background(
                                                color = when (stats.computer2Status) {
                                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ->
                                                        MaterialTheme.colorScheme.primary
                                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTING ->
                                                        MaterialTheme.colorScheme.tertiary
                                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.ERROR ->
                                                        MaterialTheme.colorScheme.error
                                                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                                                },
                                                shape = MaterialTheme.shapes.small
                                            )
                                    )
                                    Text(
                                        when (stats.computer2Status) {
                                            com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ->
                                                "Connected · ${String.format("%.1f", stats.computer2Fps)} FPS · ${stats.computer2Latency}ms"
                                            com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTING ->
                                                "Connecting..."
                                            com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.ERROR ->
                                                "Error - Check connection"
                                            else -> "Disconnected"
                                        },
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                        Switch(
                            checked = computer2Enabled,
                            onCheckedChange = { enabled ->
                                computer2Enabled = enabled
                                scope.launch {
                                    if (enabled && computer2Ip.isNotEmpty()) {
                                        streamViewModel.enableComputer2Streaming(
                                            computer2Ip,
                                            computer2Port.toIntOrNull() ?: 3000
                                        )
                                    } else {
                                        streamViewModel.disableComputer2Streaming()
                                    }
                                }
                            }
                        )
                    }

                    Text(
                        "Stream video to second computer via Tailscale network",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    OutlinedTextField(
                        value = computer2Ip,
                        onValueChange = { newIp ->
                            computer2Ip = newIp
                            if (computer2Enabled && newIp.isNotEmpty()) {
                                scope.launch {
                                    streamViewModel.enableComputer2Streaming(
                                        newIp,
                                        computer2Port.toIntOrNull() ?: 3000
                                    )
                                }
                            }
                        },
                        label = { Text("Tailscale IP Address") },
                        placeholder = { Text("100.x.x.x") },
                        supportingText = { Text("Second computer's Tailscale IP") },
                        modifier = Modifier.fillMaxWidth()
                    )

                    OutlinedTextField(
                        value = computer2Port,
                        onValueChange = { newPort ->
                            computer2Port = newPort
                            if (computer2Enabled && computer2Ip.isNotEmpty()) {
                                scope.launch {
                                    streamViewModel.enableComputer2Streaming(
                                        computer2Ip,
                                        newPort.toIntOrNull() ?: 3000
                                    )
                                }
                            }
                        },
                        label = { Text("Port") },
                        placeholder = { Text("3000") },
                        supportingText = { Text("Next.js dev server port") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }

            // Cloud Streaming Section
            Card {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "Cloud Streaming",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Switch(
                            checked = cloudEnabled,
                            onCheckedChange = { enabled ->
                                cloudEnabled = enabled
                                scope.launch {
                                    if (enabled) {
                                        streamViewModel.enableCloudStreaming(cloudUserId)
                                    } else {
                                        streamViewModel.disableCloudStreaming()
                                    }
                                }
                            }
                        )
                    }

                    Text(
                        "Stream to cloud backend for storage and analysis (every 5 seconds)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    OutlinedTextField(
                        value = cloudUserId,
                        onValueChange = { cloudUserId = it },
                        label = { Text("User ID") },
                        placeholder = { Text("test_user_1") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = cloudEnabled
                    )

                    Button(
                        onClick = {
                            scope.launch {
                                streamViewModel.enableCloudStreaming(cloudUserId)
                            }
                        },
                        enabled = cloudEnabled && cloudUserId.isNotEmpty(),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Save Cloud Settings")
                    }
                }
            }

            // Quality Settings Section
            Card {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        "Quality Settings",
                        style = MaterialTheme.typography.titleMedium
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "Target FPS (Computer Stream)",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            "${targetFps.toInt()} FPS",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = androidx.compose.ui.text.font.FontWeight.Bold,
                            color = if (targetFps < 15) MaterialTheme.colorScheme.error
                            else if (targetFps < 25) MaterialTheme.colorScheme.tertiary
                            else MaterialTheme.colorScheme.primary
                        )
                    }
                    Slider(
                        value = targetFps,
                        onValueChange = { targetFps = it },
                        valueRange = 1f..30f,
                        steps = 28,
                        modifier = Modifier.fillMaxWidth()
                    )
                    if (targetFps < 15) {
                        Text(
                            "⚠️ Low FPS setting detected! Increase to 25-30 for smooth streaming.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    } else {
                        Text(
                            "Higher FPS = smoother video but more bandwidth. Recommended: 25-30 FPS.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        "SDK Frame Rate",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf(15, 24, 30).forEach { fps ->
                            FilterChip(
                                selected = sdkFrameRate == fps,
                                onClick = { sdkFrameRate = fps },
                                label = { Text("${fps} FPS") }
                            )
                        }
                    }
                    Text(
                        "Higher FPS = smoother video, more battery usage",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        "JPEG Quality: ${jpegQuality.toInt()}%",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Slider(
                        value = jpegQuality,
                        onValueChange = { jpegQuality = it },
                        valueRange = 25f..100f,
                        steps = 74,
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        "Higher quality = better image but larger file size",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    Button(
                        onClick = {
                            scope.launch {
                                streamViewModel.updateQualitySettings(
                                    targetFps.toInt(),
                                    sdkFrameRate,
                                    jpegQuality.toInt()
                                )
                            }
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Save Quality Settings")
                    }
                }
            }

            // Live Statistics Section
            Card {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        "Live Statistics",
                        style = MaterialTheme.typography.titleMedium
                    )

                    uiState.streamingStats?.let { stats ->
                        // Computer Metrics (only show if enabled)
                        if (computerEnabled) {
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text(
                                    "Computer Stream",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = MaterialTheme.colorScheme.primary
                                )

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        "Frame Rate:",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        String.format("%.1f FPS", stats.computerFps),
                                        style = MaterialTheme.typography.bodyMedium,
                                        fontWeight = androidx.compose.ui.text.font.FontWeight.SemiBold,
                                        color = if (stats.computerFps > 0) MaterialTheme.colorScheme.primary
                                        else MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        "Latency:",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        "${stats.computerLatency}ms",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = when {
                                            stats.computerLatency == 0L -> MaterialTheme.colorScheme.onSurfaceVariant
                                            stats.computerLatency < 150 -> MaterialTheme.colorScheme.primary
                                            stats.computerLatency < 300 -> MaterialTheme.colorScheme.tertiary
                                            else -> MaterialTheme.colorScheme.error
                                        }
                                    )
                                }

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        "Bandwidth:",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        String.format("%.1f KB/s", stats.bandwidthKBps),
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                }
                            }

                            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                        }

                        // Computer 2 Metrics (only show if enabled)
                        if (computer2Enabled) {
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text(
                                    "Computer 2 Stream",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = MaterialTheme.colorScheme.secondary
                                )

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        "Frame Rate:",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        String.format("%.1f FPS", stats.computer2Fps),
                                        style = MaterialTheme.typography.bodyMedium,
                                        fontWeight = androidx.compose.ui.text.font.FontWeight.SemiBold,
                                        color = if (stats.computer2Fps > 0) MaterialTheme.colorScheme.primary
                                        else MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        "Latency:",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        "${stats.computer2Latency}ms",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = when {
                                            stats.computer2Latency == 0L -> MaterialTheme.colorScheme.onSurfaceVariant
                                            stats.computer2Latency < 150 -> MaterialTheme.colorScheme.primary
                                            stats.computer2Latency < 300 -> MaterialTheme.colorScheme.tertiary
                                            else -> MaterialTheme.colorScheme.error
                                        }
                                    )
                                }
                            }

                            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                        }

                        // Cloud Metrics (only show if enabled)
                        if (cloudEnabled) {
                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Text(
                                    "Cloud Stream",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = MaterialTheme.colorScheme.tertiary
                                )

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        "Status:",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        "${stats.cloudStatus}",
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = when (stats.cloudStatus) {
                                            com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ->
                                                MaterialTheme.colorScheme.primary
                                            com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.ERROR ->
                                                MaterialTheme.colorScheme.error
                                            else -> MaterialTheme.colorScheme.onSurfaceVariant
                                        }
                                    )
                                }
                            }

                            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                        }

                        // General Stats
                        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    "Dropped Frames:",
                                    style = MaterialTheme.typography.bodyMedium
                                )
                                Text(
                                    "${stats.droppedFrames}",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = if (stats.droppedFrames > 0) MaterialTheme.colorScheme.error
                                    else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    "Session Uptime:",
                                    style = MaterialTheme.typography.bodyMedium
                                )
                                Text(
                                    "${stats.uptimeSeconds / 60}m ${stats.uptimeSeconds % 60}s",
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }
                    } ?: Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            "No active streams",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            "Enable computer or cloud streaming to see live statistics",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center
                        )
                    }
                }
            }
        }
    }
}
