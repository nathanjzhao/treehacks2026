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

    var cloudUserId by remember { mutableStateOf(config.cloud.userId) }
    var cloudEnabled by remember { mutableStateOf(config.cloud.enabled) }

    var targetFps by remember { mutableStateOf(config.settings.computerTargetFps.toFloat()) }
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
                        Text(
                            "Computer Streaming",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Switch(
                            checked = computerEnabled,
                            onCheckedChange = { enabled ->
                                computerEnabled = enabled
                                scope.launch {
                                    if (enabled) {
                                        streamViewModel.enableComputerStreaming(
                                            computerIp,
                                            computerPort.toIntOrNull() ?: 8080
                                        )
                                    } else {
                                        streamViewModel.disableComputerStreaming()
                                    }
                                }
                            }
                        )
                    }

                    Text(
                        "Stream to computer via phone hotspot for VGGT 3D reconstruction",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    OutlinedTextField(
                        value = computerIp,
                        onValueChange = { computerIp = it },
                        label = { Text("IP Address") },
                        placeholder = { Text("172.20.10.1") },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = computerEnabled
                    )

                    OutlinedTextField(
                        value = computerPort,
                        onValueChange = { computerPort = it },
                        label = { Text("Port") },
                        placeholder = { Text("8080") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth(),
                        enabled = computerEnabled
                    )

                    Button(
                        onClick = {
                            scope.launch {
                                streamViewModel.enableComputerStreaming(
                                    computerIp,
                                    computerPort.toIntOrNull() ?: 8080
                                )
                            }
                        },
                        enabled = computerEnabled,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Save Computer Settings")
                    }
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

                    Text(
                        "Target FPS: ${targetFps.toInt()}",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Slider(
                        value = targetFps,
                        onValueChange = { targetFps = it },
                        valueRange = 2f..10f,
                        steps = 7,
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        "Higher FPS = better 3D reconstruction but more battery usage",
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
                        valueRange = 50f..90f,
                        steps = 7,
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        "Higher quality = better image but larger file size",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // Connection Status Section
            Card {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        "Connection Status",
                        style = MaterialTheme.typography.titleMedium
                    )

                    uiState.streamingStats?.let { stats ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Computer:")
                            Text(
                                "${stats.computerStatus} (${stats.computerFps} FPS)",
                                color = when (stats.computerStatus) {
                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ->
                                        MaterialTheme.colorScheme.primary
                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.ERROR ->
                                        MaterialTheme.colorScheme.error
                                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                                }
                            )
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Cloud:")
                            Text(
                                "${stats.cloudStatus} (${stats.cloudFps} FPS)",
                                color = when (stats.cloudStatus) {
                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ->
                                        MaterialTheme.colorScheme.primary
                                    com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.ERROR ->
                                        MaterialTheme.colorScheme.error
                                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                                }
                            )
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Bandwidth:")
                            Text("${stats.bandwidthKBps} KB/s")
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Dropped Frames:")
                            Text("${stats.droppedFrames}")
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Uptime:")
                            Text("${stats.uptimeSeconds}s")
                        }
                    } ?: Text(
                        "No statistics available",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}
