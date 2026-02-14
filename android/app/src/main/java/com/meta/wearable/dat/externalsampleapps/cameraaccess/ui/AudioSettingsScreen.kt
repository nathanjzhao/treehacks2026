/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.meta.wearable.dat.externalsampleapps.cameraaccess.audio.*

/**
 * Audio settings screen for configuring dual-mode audio recording.
 */
@Composable
fun AudioSettingsScreen(
    viewModel: AudioRecordingViewModel = viewModel(),
    onNavigateBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            AudioSettingsTopBar(onNavigateBack = onNavigateBack)
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Error message
            uiState.errorMessage?.let { error ->
                ErrorCard(
                    message = error,
                    onDismiss = { viewModel.clearError() }
                )
            }

            // Audio recording toggle
            AudioEnabledCard(
                enabled = uiState.audioConfiguration.enabled,
                onToggle = { viewModel.setAudioEnabled(it) }
            )

            // Status section
            if (uiState.audioEnabled) {
                StatusCard(
                    captureState = uiState.captureState,
                    stats = uiState.audioStats
                )
            }

            // API keys section
            ApiKeysCard(
                porcupineKey = uiState.audioConfiguration.porcupineAccessKey,
                openaiKey = uiState.audioConfiguration.openaiApiKey,
                onPorcupineKeyChange = { viewModel.setPorcupineKey(it) },
                onOpenAIKeyChange = { viewModel.setOpenAIKey(it) }
            )

            // Mira settings
            MiraSettingsCard(
                baseUrl = uiState.audioConfiguration.miraBaseUrl,
                patientId = uiState.audioConfiguration.patientId,
                onBaseUrlChange = { viewModel.setMiraBaseUrl(it) },
                onPatientIdChange = { viewModel.setPatientId(it) }
            )

            // Passive context toggle
            PassiveContextCard(
                enabled = uiState.audioConfiguration.passiveContextEnabled,
                onToggle = { viewModel.setPassiveContextEnabled(it) }
            )

            // Wake word settings
            WakeWordSettingsCard(
                settings = uiState.audioConfiguration.settings,
                onSettingsChange = { viewModel.updateSettings(it) }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AudioSettingsTopBar(onNavigateBack: () -> Unit) {
    TopAppBar(
        title = { Text("Audio Recording") },
        navigationIcon = {
            IconButton(onClick = onNavigateBack) {
                Icon(Icons.Default.ArrowBack, contentDescription = "Back")
            }
        }
    )
}

@Composable
private fun ErrorCard(
    message: String,
    onDismiss: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.weight(1f)
            ) {
                Icon(
                    Icons.Default.Warning,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.error
                )
                Text(
                    message,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }
            IconButton(onClick = onDismiss) {
                Icon(Icons.Default.Close, contentDescription = "Dismiss")
            }
        }
    }
}

@Composable
private fun AudioEnabledCard(
    enabled: Boolean,
    onToggle: (Boolean) -> Unit
) {
    Card {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "Audio Recording",
                    style = MaterialTheme.typography.titleMedium
                )
                Text(
                    "Enable dual-mode audio recording",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Switch(
                checked = enabled,
                onCheckedChange = onToggle
            )
        }
    }
}

@Composable
private fun StatusCard(
    captureState: AudioCaptureState,
    stats: AudioStats?
) {
    Card {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                "Status",
                style = MaterialTheme.typography.titleMedium
            )

            // Current state
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    when (captureState) {
                        is AudioCaptureState.Listening -> Icons.Default.Hearing
                        is AudioCaptureState.Recording -> Icons.Default.Mic
                        is AudioCaptureState.Error -> Icons.Default.Warning
                        else -> Icons.Default.Info
                    },
                    contentDescription = null,
                    tint = when (captureState) {
                        is AudioCaptureState.Error -> MaterialTheme.colorScheme.error
                        else -> MaterialTheme.colorScheme.primary
                    }
                )
                Text(
                    getStateDescription(captureState),
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            // Statistics
            if (stats != null) {
                Divider()
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    StatRow("Wake word detections", stats.wakeWordDetections.toString())
                    StatRow("Passive contexts", stats.passiveContexts.toString())
                    StatRow("Transcriptions completed", stats.transcriptionsCompleted.toString())
                    StatRow("Queries sent", stats.queriesSent.toString())
                    if (stats.transcriptionsFailed > 0) {
                        StatRow("Failed", stats.transcriptionsFailed.toString())
                    }
                    StatRow("Uptime", "${stats.uptimeSeconds}s")
                }
            }
        }
    }
}

@Composable
private fun StatRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            value,
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
private fun ApiKeysCard(
    porcupineKey: String,
    openaiKey: String,
    onPorcupineKeyChange: (String) -> Unit,
    onOpenAIKeyChange: (String) -> Unit
) {
    var showPorcupineKey by remember { mutableStateOf(false) }
    var showOpenAIKey by remember { mutableStateOf(false) }

    Card {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                "API Keys",
                style = MaterialTheme.typography.titleMedium
            )

            // Porcupine access key
            OutlinedTextField(
                value = porcupineKey,
                onValueChange = onPorcupineKeyChange,
                label = { Text("Porcupine Access Key") },
                placeholder = { Text("Get from console.picovoice.ai") },
                modifier = Modifier.fillMaxWidth(),
                visualTransformation = if (showPorcupineKey) {
                    VisualTransformation.None
                } else {
                    PasswordVisualTransformation()
                },
                trailingIcon = {
                    IconButton(onClick = { showPorcupineKey = !showPorcupineKey }) {
                        Icon(
                            if (showPorcupineKey) Icons.Default.Visibility else Icons.Default.VisibilityOff,
                            contentDescription = if (showPorcupineKey) "Hide" else "Show"
                        )
                    }
                },
                singleLine = true
            )

            // OpenAI API key
            OutlinedTextField(
                value = openaiKey,
                onValueChange = onOpenAIKeyChange,
                label = { Text("OpenAI API Key") },
                placeholder = { Text("sk-...") },
                modifier = Modifier.fillMaxWidth(),
                visualTransformation = if (showOpenAIKey) {
                    VisualTransformation.None
                } else {
                    PasswordVisualTransformation()
                },
                trailingIcon = {
                    IconButton(onClick = { showOpenAIKey = !showOpenAIKey }) {
                        Icon(
                            if (showOpenAIKey) Icons.Default.Visibility else Icons.Default.VisibilityOff,
                            contentDescription = if (showOpenAIKey) "Hide" else "Show"
                        )
                    }
                },
                singleLine = true
            )
        }
    }
}

@Composable
private fun MiraSettingsCard(
    baseUrl: String,
    patientId: String,
    onBaseUrlChange: (String) -> Unit,
    onPatientIdChange: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    val selectedPatient = remember(patientId) {
        DemoPatients.ALL.find { it.id == patientId } ?: DemoPatients.ALL[0]
    }

    Card {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                "Mira Settings",
                style = MaterialTheme.typography.titleMedium
            )

            // Base URL
            OutlinedTextField(
                value = baseUrl,
                onValueChange = onBaseUrlChange,
                label = { Text("Mira Base URL") },
                placeholder = { Text("http://192.168.1.100:3000") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            // Patient ID selector
            ExposedDropdownMenuBox(
                expanded = expanded,
                onExpandedChange = { expanded = it }
            ) {
                OutlinedTextField(
                    value = "${selectedPatient.name} (${selectedPatient.room})",
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Patient") },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )

                ExposedDropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    DemoPatients.ALL.forEach { patient ->
                        DropdownMenuItem(
                            text = {
                                Column {
                                    Text(patient.name)
                                    Text(
                                        patient.room,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            },
                            onClick = {
                                onPatientIdChange(patient.id)
                                expanded = false
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PassiveContextCard(
    enabled: Boolean,
    onToggle: (Boolean) -> Unit
) {
    Card {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "Passive Context Collection",
                    style = MaterialTheme.typography.titleMedium
                )
                Text(
                    "Log all speech (silent, no TTS response)",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Switch(
                checked = enabled,
                onCheckedChange = onToggle
            )
        }
    }
}

@Composable
private fun WakeWordSettingsCard(
    settings: AudioSettings,
    onSettingsChange: (AudioSettings) -> Unit
) {
    Card {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                "Wake Word Settings",
                style = MaterialTheme.typography.titleMedium
            )

            // Wake word sensitivity
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Wake Word Sensitivity")
                    Text("${(settings.wakeWordSensitivity * 100).toInt()}%")
                }
                Slider(
                    value = settings.wakeWordSensitivity,
                    onValueChange = { onSettingsChange(settings.copy(wakeWordSensitivity = it)) },
                    valueRange = 0f..1f,
                    steps = 9 // 10% increments
                )
                Text(
                    "Higher = more sensitive, more false positives",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Divider()

            // Silence timeout
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Silence Timeout")
                    Text("${settings.silenceTimeoutMs / 1000.0}s")
                }
                Slider(
                    value = settings.silenceTimeoutMs.toFloat(),
                    onValueChange = { onSettingsChange(settings.copy(silenceTimeoutMs = it.toLong())) },
                    valueRange = 1000f..5000f,
                    steps = 7 // 0.5s increments
                )
                Text(
                    "Stop recording after this much silence",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Divider()

            // Silence threshold
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Silence Threshold")
                    Text("${settings.silenceThresholdRMS.toInt()}")
                }
                Slider(
                    value = settings.silenceThresholdRMS.toFloat(),
                    onValueChange = { onSettingsChange(settings.copy(silenceThresholdRMS = it.toDouble())) },
                    valueRange = 100f..1000f,
                    steps = 8 // 100 unit increments
                )
                Text(
                    "Lower = more sensitive to quiet speech",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

private fun getStateDescription(state: AudioCaptureState): String {
    return when (state) {
        is AudioCaptureState.Stopped -> "Stopped"
        is AudioCaptureState.Listening -> "Listening for wake words..."
        is AudioCaptureState.WakeWordDetected -> "Wake word detected: ${state.keyword}"
        is AudioCaptureState.SpeechDetected -> "Passive speech detected"
        is AudioCaptureState.Recording -> {
            val mode = if (state.mode == RecordingMode.QUERY) "Query" else "Passive"
            "Recording ($mode): ${state.durationMs / 1000.0}s"
        }
        is AudioCaptureState.Transcribing -> {
            val mode = if (state.mode == RecordingMode.QUERY) "Query" else "Passive"
            "Transcribing ($mode)..."
        }
        is AudioCaptureState.SendingQuery -> "Sending query: ${state.transcription}"
        is AudioCaptureState.SendingContext -> "Logging context: ${state.transcription}"
        is AudioCaptureState.PlayingTTS -> "Playing response..."
        is AudioCaptureState.Error -> "Error: ${state.message}"
    }
}
