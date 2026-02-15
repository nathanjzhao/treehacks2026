/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamingPreferencesDataStore - Configuration Persistence
//
// Persists streaming configuration using Jetpack DataStore.
// Configuration survives app restarts and phone reboots.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import java.io.IOException

// Extension property to create DataStore
private val Context.streamingDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "streaming_settings"
)

class StreamingPreferencesDataStore(private val context: Context) {

    // Preference keys
    private object PreferencesKeys {
        // Computer endpoint
        val COMPUTER_IP = stringPreferencesKey("computer_ip")
        val COMPUTER_PORT = intPreferencesKey("computer_port")
        val COMPUTER_ENABLED = booleanPreferencesKey("computer_enabled")

        // Computer 2 endpoint
        val COMPUTER2_IP = stringPreferencesKey("computer2_ip")
        val COMPUTER2_PORT = intPreferencesKey("computer2_port")
        val COMPUTER2_ENABLED = booleanPreferencesKey("computer2_enabled")

        // Cloud endpoint
        val CLOUD_BASE_URL = stringPreferencesKey("cloud_base_url")
        val CLOUD_USER_ID = stringPreferencesKey("cloud_user_id")
        val CLOUD_ENABLED = booleanPreferencesKey("cloud_enabled")

        // Settings
        val COMPUTER_TARGET_FPS = intPreferencesKey("computer_target_fps")
        val JPEG_QUALITY = intPreferencesKey("jpeg_quality")
        val CLOUD_CAPTURE_INTERVAL_MS = longPreferencesKey("cloud_capture_interval_ms")
    }

    /**
     * Flow of streaming configuration - automatically emits updates when preferences change
     */
    val streamingConfiguration: Flow<StreamingConfiguration> = context.streamingDataStore.data
        .catch { exception ->
            if (exception is IOException) {
                StreamingLogger.error("DataStore", "Error reading preferences: ${exception.message}")
                emit(emptyPreferences())
            } else {
                throw exception
            }
        }
        .map { preferences ->
            StreamingConfiguration(
                computer = ComputerEndpoint(
                    ip = preferences[PreferencesKeys.COMPUTER_IP] ?: "172.20.10.1",
                    port = preferences[PreferencesKeys.COMPUTER_PORT] ?: 8080,
                    enabled = preferences[PreferencesKeys.COMPUTER_ENABLED] ?: false
                ),
                computer2 = ComputerEndpoint(
                    ip = preferences[PreferencesKeys.COMPUTER2_IP] ?: "172.20.10.1",
                    port = preferences[PreferencesKeys.COMPUTER2_PORT] ?: 8080,
                    enabled = preferences[PreferencesKeys.COMPUTER2_ENABLED] ?: false
                ),
                cloud = CloudEndpoint(
                    baseUrl = preferences[PreferencesKeys.CLOUD_BASE_URL]
                        ?: "https://memory-backend-328251955578.us-east1.run.app",
                    userId = preferences[PreferencesKeys.CLOUD_USER_ID] ?: "",
                    enabled = preferences[PreferencesKeys.CLOUD_ENABLED] ?: false
                ),
                settings = StreamingSettings(
                    computerTargetFps = preferences[PreferencesKeys.COMPUTER_TARGET_FPS] ?: 7,
                    jpegQuality = preferences[PreferencesKeys.JPEG_QUALITY] ?: 70,
                    cloudCaptureIntervalMs = preferences[PreferencesKeys.CLOUD_CAPTURE_INTERVAL_MS] ?: 5000L
                )
            )
        }

    /**
     * Update computer endpoint configuration
     */
    suspend fun updateComputerEndpoint(endpoint: ComputerEndpoint) {
        context.streamingDataStore.edit { preferences ->
            preferences[PreferencesKeys.COMPUTER_IP] = endpoint.ip
            preferences[PreferencesKeys.COMPUTER_PORT] = endpoint.port
            preferences[PreferencesKeys.COMPUTER_ENABLED] = endpoint.enabled
        }
        StreamingLogger.info("DataStore", "Computer endpoint updated: ${endpoint.ip}:${endpoint.port}, enabled=${endpoint.enabled}")
    }

    /**
     * Update computer 2 endpoint configuration
     */
    suspend fun updateComputer2Endpoint(endpoint: ComputerEndpoint) {
        context.streamingDataStore.edit { preferences ->
            preferences[PreferencesKeys.COMPUTER2_IP] = endpoint.ip
            preferences[PreferencesKeys.COMPUTER2_PORT] = endpoint.port
            preferences[PreferencesKeys.COMPUTER2_ENABLED] = endpoint.enabled
        }
        StreamingLogger.info("DataStore", "Computer 2 endpoint updated: ${endpoint.ip}:${endpoint.port}, enabled=${endpoint.enabled}")
    }

    /**
     * Update cloud endpoint configuration
     */
    suspend fun updateCloudEndpoint(endpoint: CloudEndpoint) {
        context.streamingDataStore.edit { preferences ->
            preferences[PreferencesKeys.CLOUD_BASE_URL] = endpoint.baseUrl
            preferences[PreferencesKeys.CLOUD_USER_ID] = endpoint.userId
            preferences[PreferencesKeys.CLOUD_ENABLED] = endpoint.enabled
        }
        StreamingLogger.info("DataStore", "Cloud endpoint updated: userId=${endpoint.userId}, enabled=${endpoint.enabled}")
    }

    /**
     * Update streaming quality settings
     */
    suspend fun updateSettings(settings: StreamingSettings) {
        context.streamingDataStore.edit { preferences ->
            preferences[PreferencesKeys.COMPUTER_TARGET_FPS] = settings.computerTargetFps
            preferences[PreferencesKeys.JPEG_QUALITY] = settings.jpegQuality
            preferences[PreferencesKeys.CLOUD_CAPTURE_INTERVAL_MS] = settings.cloudCaptureIntervalMs
        }
        StreamingLogger.info("DataStore", "Settings updated: FPS=${settings.computerTargetFps}, Quality=${settings.jpegQuality}%")
    }

    /**
     * Clear all preferences (reset to defaults)
     */
    suspend fun clearAll() {
        context.streamingDataStore.edit { preferences ->
            preferences.clear()
        }
        StreamingLogger.info("DataStore", "All preferences cleared")
    }
}
