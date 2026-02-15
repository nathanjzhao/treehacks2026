/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioManager
import com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingLogger

/**
 * Helper class for managing Bluetooth SCO audio routing for Ray-Ban glasses microphone.
 *
 * Bluetooth SCO (Synchronous Connection Oriented) is required to route audio from the
 * Ray-Ban glasses microphone to the Android device.
 */
class BluetoothScoManager(private val context: Context) {
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private var scoStateReceiver: BroadcastReceiver? = null
    private var isStarted = false

    companion object {
        private const val TAG = "BluetoothScoManager"
    }

    /**
     * Start Bluetooth SCO audio routing.
     *
     * @return true if SCO start was initiated, false if already started or failed
     */
    fun startBluetoothSco(): Boolean {
        if (isStarted) {
            StreamingLogger.info(TAG, "Bluetooth SCO already started")
            return false
        }

        try {
            // Register broadcast receiver for SCO state changes
            registerScoStateReceiver()

            // Configure audio manager for Bluetooth SCO
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            audioManager.isBluetoothScoOn = true
            audioManager.startBluetoothSco()

            isStarted = true
            StreamingLogger.info(TAG, "Bluetooth SCO started")
            return true

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Failed to start Bluetooth SCO: ${e.message}")
            return false
        }
    }

    /**
     * Stop Bluetooth SCO audio routing.
     */
    fun stopBluetoothSco() {
        if (!isStarted) {
            return
        }

        try {
            audioManager.stopBluetoothSco()
            audioManager.isBluetoothScoOn = false
            audioManager.mode = AudioManager.MODE_NORMAL

            unregisterScoStateReceiver()

            isStarted = false
            StreamingLogger.info(TAG, "Bluetooth SCO stopped")

        } catch (e: Exception) {
            StreamingLogger.error(TAG, "Failed to stop Bluetooth SCO: ${e.message}")
        }
    }

    /**
     * Check if Bluetooth SCO is currently connected.
     *
     * @return true if SCO is connected, false otherwise
     */
    fun isScoConnected(): Boolean {
        return audioManager.isBluetoothScoOn
    }

    /**
     * Register broadcast receiver to monitor SCO state changes.
     */
    private fun registerScoStateReceiver() {
        if (scoStateReceiver != null) {
            return
        }

        scoStateReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                val state = intent?.getIntExtra(
                    AudioManager.EXTRA_SCO_AUDIO_STATE,
                    AudioManager.SCO_AUDIO_STATE_DISCONNECTED
                )

                when (state) {
                    AudioManager.SCO_AUDIO_STATE_CONNECTED -> {
                        StreamingLogger.info(TAG, "SCO audio connected")
                    }
                    AudioManager.SCO_AUDIO_STATE_DISCONNECTED -> {
                        StreamingLogger.info(TAG, "SCO audio disconnected")
                    }
                    AudioManager.SCO_AUDIO_STATE_CONNECTING -> {
                        StreamingLogger.info(TAG, "SCO audio connecting...")
                    }
                    AudioManager.SCO_AUDIO_STATE_ERROR -> {
                        StreamingLogger.error(TAG, "SCO audio error")
                    }
                }
            }
        }

        val filter = IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED)
        context.registerReceiver(scoStateReceiver, filter)
        StreamingLogger.debug(TAG, "SCO state receiver registered")
    }

    /**
     * Unregister SCO state broadcast receiver.
     */
    private fun unregisterScoStateReceiver() {
        scoStateReceiver?.let {
            try {
                context.unregisterReceiver(it)
                scoStateReceiver = null
                StreamingLogger.debug(TAG, "SCO state receiver unregistered")
            } catch (e: IllegalArgumentException) {
                // Receiver was already unregistered
                StreamingLogger.debug(TAG, "SCO state receiver was already unregistered")
            }
        }
    }

    /**
     * Clean up resources.
     */
    fun cleanup() {
        stopBluetoothSco()
    }
}
