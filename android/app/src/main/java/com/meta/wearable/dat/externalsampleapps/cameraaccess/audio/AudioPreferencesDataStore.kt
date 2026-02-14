/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import kotlinx.serialization.Serializable
import java.io.IOException

/**
 * DataStore extension for audio preferences.
 */
private val Context.audioDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "audio_settings"
)

/**
 * Preference keys for audio configuration.
 */
private object PreferencesKeys {
    val AUDIO_ENABLED = booleanPreferencesKey("audio_enabled")
    val PORCUPINE_ACCESS_KEY = stringPreferencesKey("porcupine_access_key")
    val OPENAI_API_KEY = stringPreferencesKey("openai_api_key")
    val MIRA_BASE_URL = stringPreferencesKey("mira_base_url")
    val PATIENT_ID = stringPreferencesKey("patient_id")
    val PASSIVE_CONTEXT_ENABLED = booleanPreferencesKey("passive_context_enabled")
    val WAKE_WORD_SENSITIVITY = floatPreferencesKey("wake_word_sensitivity")
    val SILENCE_TIMEOUT_MS = longPreferencesKey("silence_timeout_ms")
    val MIN_RECORDING_DURATION_MS = longPreferencesKey("min_recording_duration_ms")
    val MAX_RECORDING_DURATION_MS = longPreferencesKey("max_recording_duration_ms")
    val SILENCE_THRESHOLD_RMS = stringPreferencesKey("silence_threshold_rms")
}

/**
 * Audio configuration settings.
 */
@Serializable
data class AudioConfiguration(
    val enabled: Boolean = false,
    val porcupineAccessKey: String = "",
    val openaiApiKey: String = "",
    val miraBaseUrl: String = "http://192.168.1.100:3000",
    val patientId: String = "a1b2c3d4-0001-4000-8000-000000000001", // Demo: Margaret Chen
    val passiveContextEnabled: Boolean = true,
    val settings: AudioSettings = AudioSettings()
)

/**
 * Audio recording settings (tunable parameters).
 */
@Serializable
data class AudioSettings(
    val wakeWordSensitivity: Float = 0.5f,
    val silenceTimeoutMs: Long = 2000L,
    val minRecordingDurationMs: Long = 500L,
    val maxRecordingDurationMs: Long = 30000L,
    val silenceThresholdRMS: Double = 500.0
)

/**
 * Demo patient IDs from Mira backend.
 */
object DemoPatients {
    const val MARGARET_CHEN = "a1b2c3d4-0001-4000-8000-000000000001"
    const val ROBERT_WILLIAMS = "a1b2c3d4-0002-4000-8000-000000000002"
    const val HELEN_GARCIA = "a1b2c3d4-0003-4000-8000-000000000003"

    val ALL = listOf(
        PatientInfo(MARGARET_CHEN, "Margaret Chen", "Room 204"),
        PatientInfo(ROBERT_WILLIAMS, "Robert Williams", "Room 112"),
        PatientInfo(HELEN_GARCIA, "Helen Garcia", "Room 318")
    )
}

/**
 * Patient information for UI display.
 */
data class PatientInfo(
    val id: String,
    val name: String,
    val room: String
)

/**
 * DataStore for persisting audio configuration.
 */
class AudioPreferencesDataStore(private val context: Context) {

    /**
     * Flow of current audio configuration.
     */
    val audioConfiguration: Flow<AudioConfiguration> = context.audioDataStore.data
        .catch { exception ->
            if (exception is IOException) {
                emit(emptyPreferences())
            } else {
                throw exception
            }
        }
        .map { preferences ->
            AudioConfiguration(
                enabled = preferences[PreferencesKeys.AUDIO_ENABLED] ?: false,
                porcupineAccessKey = preferences[PreferencesKeys.PORCUPINE_ACCESS_KEY] ?: "",
                openaiApiKey = preferences[PreferencesKeys.OPENAI_API_KEY] ?: "",
                miraBaseUrl = preferences[PreferencesKeys.MIRA_BASE_URL] ?: "http://192.168.1.100:3000",
                patientId = preferences[PreferencesKeys.PATIENT_ID] ?: DemoPatients.MARGARET_CHEN,
                passiveContextEnabled = preferences[PreferencesKeys.PASSIVE_CONTEXT_ENABLED] ?: true,
                settings = AudioSettings(
                    wakeWordSensitivity = preferences[PreferencesKeys.WAKE_WORD_SENSITIVITY] ?: 0.5f,
                    silenceTimeoutMs = preferences[PreferencesKeys.SILENCE_TIMEOUT_MS] ?: 2000L,
                    minRecordingDurationMs = preferences[PreferencesKeys.MIN_RECORDING_DURATION_MS] ?: 500L,
                    maxRecordingDurationMs = preferences[PreferencesKeys.MAX_RECORDING_DURATION_MS] ?: 30000L,
                    silenceThresholdRMS = preferences[PreferencesKeys.SILENCE_THRESHOLD_RMS]?.toDoubleOrNull() ?: 500.0
                )
            )
        }

    /**
     * Update audio enabled state.
     */
    suspend fun updateAudioEnabled(enabled: Boolean) {
        context.audioDataStore.edit { preferences ->
            preferences[PreferencesKeys.AUDIO_ENABLED] = enabled
        }
    }

    /**
     * Update Porcupine access key.
     */
    suspend fun updatePorcupineKey(key: String) {
        context.audioDataStore.edit { preferences ->
            preferences[PreferencesKeys.PORCUPINE_ACCESS_KEY] = key
        }
    }

    /**
     * Update OpenAI API key.
     */
    suspend fun updateOpenAIKey(key: String) {
        context.audioDataStore.edit { preferences ->
            preferences[PreferencesKeys.OPENAI_API_KEY] = key
        }
    }

    /**
     * Update Mira base URL.
     */
    suspend fun updateMiraBaseUrl(url: String) {
        context.audioDataStore.edit { preferences ->
            preferences[PreferencesKeys.MIRA_BASE_URL] = url
        }
    }

    /**
     * Update patient ID.
     */
    suspend fun updatePatientId(patientId: String) {
        context.audioDataStore.edit { preferences ->
            preferences[PreferencesKeys.PATIENT_ID] = patientId
        }
    }

    /**
     * Update passive context enabled state.
     */
    suspend fun updatePassiveContextEnabled(enabled: Boolean) {
        context.audioDataStore.edit { preferences ->
            preferences[PreferencesKeys.PASSIVE_CONTEXT_ENABLED] = enabled
        }
    }

    /**
     * Update audio settings.
     */
    suspend fun updateSettings(settings: AudioSettings) {
        context.audioDataStore.edit { preferences ->
            preferences[PreferencesKeys.WAKE_WORD_SENSITIVITY] = settings.wakeWordSensitivity
            preferences[PreferencesKeys.SILENCE_TIMEOUT_MS] = settings.silenceTimeoutMs
            preferences[PreferencesKeys.MIN_RECORDING_DURATION_MS] = settings.minRecordingDurationMs
            preferences[PreferencesKeys.MAX_RECORDING_DURATION_MS] = settings.maxRecordingDurationMs
            preferences[PreferencesKeys.SILENCE_THRESHOLD_RMS] = settings.silenceThresholdRMS.toString()
        }
    }

    /**
     * Clear all audio preferences.
     */
    suspend fun clearAll() {
        context.audioDataStore.edit { preferences ->
            preferences.clear()
        }
    }
}
