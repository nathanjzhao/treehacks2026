/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// DebugConsoleScreen - Real-time Debug Logging UI
//
// Provides real-time logging console for debugging streaming issues,
// monitoring frame flow, and troubleshooting network problems.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.LogEntry
import com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.LogLevel
import com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingLogger

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DebugConsoleScreen(
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    val logs by StreamingLogger.logs.collectAsStateWithLifecycle()
    val enabled by StreamingLogger.enabled.collectAsStateWithLifecycle()
    val listState = rememberLazyListState()

    // Auto-scroll to bottom when new logs arrive
    LaunchedEffect(logs.size) {
        if (logs.isNotEmpty()) {
            listState.animateScrollToItem(logs.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Debug Console") },
                navigationIcon = {
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Default.Close, contentDescription = "Close")
                    }
                },
                actions = {
                    // Toggle logging
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(end = 8.dp)
                    ) {
                        Text("Logging", style = MaterialTheme.typography.bodySmall)
                        Spacer(modifier = Modifier.width(4.dp))
                        Switch(
                            checked = enabled,
                            onCheckedChange = {
                                if (it) StreamingLogger.enable() else StreamingLogger.disable()
                            }
                        )
                    }
                    TextButton(onClick = { StreamingLogger.clear() }) {
                        Text("Clear")
                    }
                }
            )
        }
    ) { padding ->
        Box(
            modifier = modifier
                .fillMaxSize()
                .padding(padding)
                .background(Color(0xFF1E1E1E)) // Dark background like IDE console
        ) {
            if (logs.isEmpty()) {
                Text(
                    "No logs yet. Enable logging with the switch above.",
                    color = Color.Gray,
                    modifier = Modifier
                        .align(Alignment.Center)
                        .padding(16.dp)
                )
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(8.dp)
                ) {
                    items(logs) { entry ->
                        LogEntryRow(entry)
                    }
                }
            }
        }
    }
}

@Composable
fun LogEntryRow(entry: LogEntry) {
    val color = when (entry.level) {
        LogLevel.DEBUG -> Color(0xFF808080)    // Gray
        LogLevel.INFO -> Color(0xFF00FF00)     // Green
        LogLevel.WARNING -> Color(0xFFFFA500)  // Orange
        LogLevel.ERROR -> Color(0xFFFF0000)    // Red
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 2.dp)
    ) {
        Text(
            text = entry.timestamp,
            color = Color(0xFF888888),
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(80.dp)
        )
        Text(
            text = "[${entry.level.name}]",
            color = color,
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(70.dp)
        )
        Text(
            text = entry.tag,
            color = Color(0xFFBBBBBB),
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(120.dp)
        )
        Text(
            text = entry.message,
            color = Color.White,
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.weight(1f)
        )
    }
}
