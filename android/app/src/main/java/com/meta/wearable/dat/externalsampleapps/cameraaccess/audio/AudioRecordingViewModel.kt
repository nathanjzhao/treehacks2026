/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingLogger
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

/**
 * ViewModel for audio recording state management.
 *
 * Manages AudioRecordingManager lifecycle and exposes UI state.
 */
class AudioRecordingViewModel(application: Application) : AndroidViewModel(application) {

    private val audioPreferences = AudioPreferencesDataStore(application)
    private var audioRecordingManager: AudioRecordingManager? = null

    // UI State
    private val _uiState = MutableStateFlow(AudioUiState())
    val uiState: StateFlow<AudioUiState> = _uiState.asStateFlow()

    companion object {
        private const val TAG = "AudioRecordingViewModel"
    }

    init {
        // Observe configuration changes
        viewModelScope.launch {
            audioPreferences.audioConfiguration.collect { config ->
                _uiState.update { it.copy(audioConfiguration = config) }

                // Auto-start/stop based on enabled flag
                if (config.enabled && audioRecordingManager == null) {
                    enableAudioRecording()
                } else if (!config.enabled && audioRecordingManager != null) {
                    disableAudioRecording()
                }
            }
        }
    }

    /**
     * Enable audio recording system.
     */
    fun enableAudioRecording() {
        if (audioRecordingManager != null) {
            StreamingLogger.info(TAG, "Audio recording already enabled")
            return
        }

        viewModelScope.launch {
            val config = _uiState.value.audioConfiguration

            // Validate configuration
            if (config.porcupineAccessKey.isBlank()) {
                _uiState.update {
                    it.copy(
                        errorMessage = "Porcupine access key is required"
                    )
                }
                return@launch
            }
            if (config.openaiApiKey.isBlank()) {
                _uiState.update {
                    it.copy(
                        errorMessage = "OpenAI API key is required"
                    )
                }
                return@launch
            }

            // Initialize manager
            val manager = AudioRecordingManager(getApplication(), config)
            if (manager.initialize()) {
                audioRecordingManager = manager
                manager.start()

                // Observe state and stats
                viewModelScope.launch {
                    manager.state.collect { state ->
                        _uiState.update { it.copy(captureState = state) }
                    }
                }
                viewModelScope.launch {
                    manager.stats.collect { stats ->
                        _uiState.update { it.copy(audioStats = stats) }
                    }
                }

                _uiState.update {
                    it.copy(
                        audioEnabled = true,
                        errorMessage = null
                    )
                }

                StreamingLogger.info(TAG, "Audio recording enabled")
            } else {
                _uiState.update {
                    it.copy(
                        errorMessage = "Failed to initialize audio recording"
                    )
                }
                StreamingLogger.error(TAG, "Failed to initialize audio recording")
            }
        }
    }

    /**
     * Disable audio recording system.
     */
    fun disableAudioRecording() {
        audioRecordingManager?.cleanup()
        audioRecordingManager = null

        _uiState.update {
            it.copy(
                audioEnabled = false,
                captureState = AudioCaptureState.Stopped,
                audioStats = null
            )
        }

        StreamingLogger.info(TAG, "Audio recording disabled")
    }

    /**
     * Update audio enabled state.
     */
    fun setAudioEnabled(enabled: Boolean) {
        viewModelScope.launch {
            audioPreferences.updateAudioEnabled(enabled)
        }
    }

    /**
     * Update Porcupine access key.
     */
    fun setPorcupineKey(key: String) {
        viewModelScope.launch {
            audioPreferences.updatePorcupineKey(key)
        }
    }

    /**
     * Update OpenAI API key.
     */
    fun setOpenAIKey(key: String) {
        viewModelScope.launch {
            audioPreferences.updateOpenAIKey(key)
        }
    }

    /**
     * Update Mira base URL.
     */
    fun setMiraBaseUrl(url: String) {
        viewModelScope.launch {
            audioPreferences.updateMiraBaseUrl(url)
        }
    }

    /**
     * Update patient ID.
     */
    fun setPatientId(patientId: String) {
        viewModelScope.launch {
            audioPreferences.updatePatientId(patientId)
        }
    }

    /**
     * Update passive context enabled state.
     */
    fun setPassiveContextEnabled(enabled: Boolean) {
        viewModelScope.launch {
            audioPreferences.updatePassiveContextEnabled(enabled)
        }
    }

    /**
     * Update audio settings.
     */
    fun updateSettings(settings: AudioSettings) {
        viewModelScope.launch {
            audioPreferences.updateSettings(settings)
        }
    }

    /**
     * Clear error message.
     */
    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    override fun onCleared() {
        super.onCleared()
        audioRecordingManager?.cleanup()
    }
}

/**
 * UI state for audio recording.
 */
data class AudioUiState(
    val captureState: AudioCaptureState = AudioCaptureState.Stopped,
    val audioEnabled: Boolean = false,
    val audioStats: AudioStats? = null,
    val audioConfiguration: AudioConfiguration = AudioConfiguration(),
    val errorMessage: String? = null
)
