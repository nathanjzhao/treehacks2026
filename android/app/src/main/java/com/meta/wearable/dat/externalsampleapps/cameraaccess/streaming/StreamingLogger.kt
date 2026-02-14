/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamingLogger - Debug Console Logging System
//
// Provides real-time logging for debugging streaming issues, monitoring frame flow,
// and troubleshooting network problems.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming

import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.text.SimpleDateFormat
import java.util.*

enum class LogLevel {
    DEBUG, INFO, WARNING, ERROR
}

data class LogEntry(
    val timestamp: String,
    val level: LogLevel,
    val tag: String,
    val message: String
)

object StreamingLogger {
    private val _logs = MutableStateFlow<List<LogEntry>>(emptyList())
    val logs: StateFlow<List<LogEntry>> = _logs.asStateFlow()

    private val _enabled = MutableStateFlow(false)
    val enabled: StateFlow<Boolean> = _enabled.asStateFlow()

    private val maxLogs = 500 // Keep last 500 entries
    private val dateFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)

    fun enable() {
        _enabled.value = true
        log(LogLevel.INFO, "StreamingLogger", "Debug console enabled")
    }

    fun disable() {
        log(LogLevel.INFO, "StreamingLogger", "Debug console disabled")
        _enabled.value = false
    }

    fun clear() {
        _logs.value = emptyList()
        log(LogLevel.INFO, "StreamingLogger", "Logs cleared")
    }

    fun log(level: LogLevel, tag: String, message: String) {
        if (!_enabled.value && level != LogLevel.ERROR) return

        val entry = LogEntry(
            timestamp = dateFormat.format(Date()),
            level = level,
            tag = tag,
            message = message
        )

        _logs.value = (_logs.value + entry).takeLast(maxLogs)

        // Also log to Android logcat
        when (level) {
            LogLevel.DEBUG -> Log.d(tag, message)
            LogLevel.INFO -> Log.i(tag, message)
            LogLevel.WARNING -> Log.w(tag, message)
            LogLevel.ERROR -> Log.e(tag, message)
        }
    }

    fun debug(tag: String, message: String) = log(LogLevel.DEBUG, tag, message)
    fun info(tag: String, message: String) = log(LogLevel.INFO, tag, message)
    fun warning(tag: String, message: String) = log(LogLevel.WARNING, tag, message)
    fun error(tag: String, message: String) = log(LogLevel.ERROR, tag, message)
}
