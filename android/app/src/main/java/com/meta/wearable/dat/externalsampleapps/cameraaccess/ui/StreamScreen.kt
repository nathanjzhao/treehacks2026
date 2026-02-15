/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamScreen - DAT Camera Streaming UI
//
// This composable demonstrates the main streaming UI for DAT camera functionality. It shows how to
// display live video from wearable devices and handle photo capture.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.activity.ComponentActivity
import androidx.activity.compose.LocalActivity
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.meta.wearable.dat.camera.types.StreamSessionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.R
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamViewModel
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesViewModel

@Composable
fun StreamScreen(
    wearablesViewModel: WearablesViewModel,
    modifier: Modifier = Modifier,
    streamViewModel: StreamViewModel =
        viewModel(
            factory =
                StreamViewModel.Factory(
                    application = (LocalActivity.current as ComponentActivity).application,
                    wearablesViewModel = wearablesViewModel,
                ),
        ),
) {
  val streamUiState by streamViewModel.uiState.collectAsStateWithLifecycle()
  var showDebugConsole by remember { mutableStateOf(false) }
  var showSettings by remember { mutableStateOf(false) }
  var showAudioSettings by remember { mutableStateOf(false) }

  LaunchedEffect(Unit) { streamViewModel.startStream() }

  Box(modifier = modifier.fillMaxSize()) {
    streamUiState.videoFrame?.let { videoFrame ->
      Image(
          bitmap = videoFrame.asImageBitmap(),
          contentDescription = stringResource(R.string.live_stream),
          modifier = Modifier.fillMaxSize(),
          contentScale = ContentScale.Crop,
      )
    }
    if (streamUiState.streamSessionState == StreamSessionState.STARTING) {
      CircularProgressIndicator(
          modifier = Modifier.align(Alignment.Center),
      )
    }

    // Streaming status badge at top
    if (streamUiState.multiDestinationStreamingEnabled) {
      StreamingStatusBadge(
          stats = streamUiState.streamingStats,
          modifier = Modifier
              .align(Alignment.TopCenter)
              .padding(top = 16.dp)
      )
    }

    Box(modifier = Modifier.fillMaxSize().padding(all = 24.dp)) {
      Row(
          modifier =
              Modifier.align(Alignment.BottomCenter)
                  .navigationBarsPadding()
                  .fillMaxWidth()
                  .height(56.dp),
          horizontalArrangement = Arrangement.spacedBy(8.dp),
          verticalAlignment = Alignment.CenterVertically,
      ) {
        SwitchButton(
            label = stringResource(R.string.stop_stream_button_title),
            onClick = {
              streamViewModel.stopStream()
              wearablesViewModel.navigateToDeviceSelection()
            },
            isDestructive = true,
            modifier = Modifier.weight(1f),
        )

        // Photo capture button
        CaptureButton(
            onClick = { streamViewModel.capturePhoto() },
        )
      }

      // Settings button (top-left)
      androidx.compose.material3.FloatingActionButton(
          onClick = { showSettings = true },
          modifier = Modifier
              .align(Alignment.TopStart)
              .padding(top = 60.dp),
          containerColor = androidx.compose.material3.MaterialTheme.colorScheme.surface.copy(alpha = 0.8f)
      ) {
        androidx.compose.material3.Icon(
            androidx.compose.material.icons.Icons.Default.Settings,
            contentDescription = "Streaming Settings"
        )
      }

      // Audio settings button (top-left, below settings)
      androidx.compose.material3.FloatingActionButton(
          onClick = { showAudioSettings = true },
          modifier = Modifier
              .align(Alignment.TopStart)
              .padding(top = 128.dp),
          containerColor = androidx.compose.material3.MaterialTheme.colorScheme.surface.copy(alpha = 0.8f)
      ) {
        androidx.compose.material3.Icon(
            androidx.compose.material.icons.Icons.Default.Mic,
            contentDescription = "Audio Settings"
        )
      }

      // Debug console button (top-right)
      androidx.compose.material3.FloatingActionButton(
          onClick = { showDebugConsole = true },
          modifier = Modifier
              .align(Alignment.TopEnd)
              .padding(top = 60.dp),
          containerColor = androidx.compose.material3.MaterialTheme.colorScheme.surface.copy(alpha = 0.8f)
      ) {
        androidx.compose.material3.Icon(
            androidx.compose.material.icons.Icons.Default.BugReport,
            contentDescription = "Debug Console"
        )
      }
    }
  }

  streamUiState.capturedPhoto?.let { photo ->
    if (streamUiState.isShareDialogVisible) {
      SharePhotoDialog(
          photo = photo,
          onDismiss = { streamViewModel.hideShareDialog() },
          onShare = { bitmap ->
            streamViewModel.sharePhoto(bitmap)
            streamViewModel.hideShareDialog()
          },
      )
    }
  }

  // Debug console dialog
  if (showDebugConsole) {
    DebugConsoleScreen(
        onDismiss = { showDebugConsole = false }
    )
  }

  // Settings dialog
  if (showSettings) {
    StreamingSettingsScreen(
        streamViewModel = streamViewModel,
        onDismiss = { showSettings = false }
    )
  }

  // Audio settings dialog
  if (showAudioSettings) {
    AudioSettingsScreen(
        onNavigateBack = { showAudioSettings = false }
    )
  }
}

@Composable
fun StreamingStatusBadge(
    stats: com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingStats?,
    modifier: Modifier = Modifier
) {
    if (stats == null) return

    androidx.compose.material3.Surface(
        modifier = modifier,
        color = androidx.compose.material3.MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
        shape = androidx.compose.foundation.shape.RoundedCornerShape(20.dp),
        shadowElevation = 4.dp
    ) {
        androidx.compose.foundation.layout.Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Status indicator
            val isStreaming = stats.computerStatus == com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ||
                    stats.computer2Status == com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED ||
                    stats.cloudStatus == com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED

            androidx.compose.foundation.Canvas(
                modifier = Modifier.size(8.dp)
            ) {
                drawCircle(
                    color = if (isStreaming) androidx.compose.ui.graphics.Color.Green else androidx.compose.ui.graphics.Color.Gray
                )
            }

            // Streaming info
            androidx.compose.material3.Text(
                text = buildString {
                    append("Streaming: ")
                    val destinations = mutableListOf<String>()
                    if (stats.computerStatus == com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED) {
                        destinations.add("PC1")
                    }
                    if (stats.computer2Status == com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED) {
                        destinations.add("PC2")
                    }
                    if (stats.cloudStatus == com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.ConnectionStatus.CONNECTED) {
                        destinations.add("Cloud")
                    }
                    append(destinations.joinToString("+"))
                    if (stats.computerFps > 0) {
                        append(" | ${String.format("%.1f", stats.computerFps)} FPS")
                    }
                    if (stats.bandwidthKBps > 0) {
                        append(" | ${String.format("%.0f", stats.bandwidthKBps)} KB/s")
                    }
                },
                style = androidx.compose.material3.MaterialTheme.typography.bodySmall
            )
        }
    }
}
