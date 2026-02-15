/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamViewModel - DAT Camera Streaming API Demo
//
// This ViewModel demonstrates the DAT Camera Streaming APIs for:
// - Creating and managing stream sessions with wearable devices
// - Receiving video frames from device cameras
// - Capturing photos during streaming sessions
// - Handling different video qualities and formats
// - Processing raw video data (I420 -> NV21 conversion)

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.camera.StreamSession
import com.meta.wearable.dat.camera.startStreamSession
import com.meta.wearable.dat.camera.types.PhotoData
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.StreamSessionState
import com.meta.wearable.dat.camera.types.VideoFrame
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.DeviceSelector
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesViewModel
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class StreamViewModel(
    application: Application,
    private val wearablesViewModel: WearablesViewModel,
) : AndroidViewModel(application) {

  companion object {
    private const val TAG = "StreamViewModel"
    private val INITIAL_STATE = StreamUiState()
  }

  private val deviceSelector: DeviceSelector = wearablesViewModel.deviceSelector
  private var streamSession: StreamSession? = null

  private val _uiState = MutableStateFlow(INITIAL_STATE)
  val uiState: StateFlow<StreamUiState> = _uiState.asStateFlow()

  private var videoJob: Job? = null
  private var stateJob: Job? = null

  // Multi-destination streaming components
  private val videoStreamingManager = com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.VideoStreamingManager(
      context = application,
      coroutineScope = viewModelScope
  )
  private val streamingPreferences = com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingPreferencesDataStore(application)
  private var configJob: Job? = null
  private var statsJob: Job? = null

  fun startStream() {
    videoJob?.cancel()
    stateJob?.cancel()
    val settings = _uiState.value.streamingConfiguration.settings
    val streamSession =
        Wearables.startStreamSession(
                getApplication(),
                deviceSelector,
                StreamConfiguration(videoQuality = VideoQuality.HIGH, settings.sdkFrameRate),
            )
            .also { streamSession = it }
    videoJob = viewModelScope.launch { streamSession.videoStream.collect { handleVideoFrame(it) } }
    stateJob =
        viewModelScope.launch {
          streamSession.state.collect { currentState ->
            val prevState = _uiState.value.streamSessionState
            _uiState.update { it.copy(streamSessionState = currentState) }

            // navigate back when state transitioned to STOPPED
            if (currentState != prevState && currentState == StreamSessionState.STOPPED) {
              stopStream()
              wearablesViewModel.navigateToDeviceSelection()
            }
          }
        }

    // Initialize streaming configuration and statistics
    startStreamingMonitoring()
  }

  private fun startStreamingMonitoring() {
    // Load and monitor configuration
    configJob?.cancel()
    configJob = viewModelScope.launch {
      streamingPreferences.streamingConfiguration.collect { config ->
        _uiState.update { it.copy(streamingConfiguration = config) }

        // Auto-enable/disable streaming based on configuration
        if (config.computer.enabled) {
          videoStreamingManager.enableComputerStreaming(
              endpoint = config.computer.ip,
              port = config.computer.port,
              targetFps = config.settings.computerTargetFps,
              jpegQuality = config.settings.jpegQuality
          )
        } else {
          videoStreamingManager.disableComputerStreaming()
        }

        if (config.cloud.enabled && config.cloud.userId.isNotEmpty()) {
          videoStreamingManager.enableCloudStreaming(
              userId = config.cloud.userId,
              baseUrl = config.cloud.baseUrl
          )
        } else {
          videoStreamingManager.disableCloudStreaming()
        }

        // Update UI enabled state
        _uiState.update {
          it.copy(multiDestinationStreamingEnabled = config.computer.enabled || config.cloud.enabled)
        }
      }
    }

    // Monitor streaming statistics
    statsJob?.cancel()
    statsJob = viewModelScope.launch {
      videoStreamingManager.statistics.collect { stats ->
        _uiState.update { it.copy(streamingStats = stats) }
      }
    }
  }

  fun stopStream() {
    videoJob?.cancel()
    videoJob = null
    stateJob?.cancel()
    stateJob = null
    streamSession?.close()
    streamSession = null
    _uiState.update { INITIAL_STATE }
  }

  fun capturePhoto() {
    if (uiState.value.isCapturing) {
      Log.d(TAG, "Photo capture already in progress, ignoring request")
      return
    }

    if (uiState.value.streamSessionState == StreamSessionState.STREAMING) {
      Log.d(TAG, "Starting photo capture")
      _uiState.update { it.copy(isCapturing = true) }

      viewModelScope.launch {
        streamSession
            ?.capturePhoto()
            ?.onSuccess { photoData ->
              Log.d(TAG, "Photo capture successful")
              handlePhotoData(photoData)
              _uiState.update { it.copy(isCapturing = false) }
            }
            ?.onFailure {
              Log.e(TAG, "Photo capture failed")
              _uiState.update { it.copy(isCapturing = false) }
            }
      }
    } else {
      Log.w(
          TAG,
          "Cannot capture photo: stream not active (state=${uiState.value.streamSessionState})",
      )
    }
  }

  fun showShareDialog() {
    _uiState.update { it.copy(isShareDialogVisible = true) }
  }

  fun hideShareDialog() {
    _uiState.update { it.copy(isShareDialogVisible = false) }
  }

  fun sharePhoto(bitmap: Bitmap) {
    val context = getApplication<Application>()
    val imagesFolder = File(context.cacheDir, "images")
    try {
      imagesFolder.mkdirs()
      val file = File(imagesFolder, "shared_image.png")
      FileOutputStream(file).use { stream ->
        bitmap.compress(Bitmap.CompressFormat.PNG, 90, stream)
      }

      val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
      val intent = Intent(Intent.ACTION_SEND)
      intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
      intent.putExtra(Intent.EXTRA_STREAM, uri)
      intent.type = "image/png"
      intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

      val chooser = Intent.createChooser(intent, "Share Image")
      chooser.flags = Intent.FLAG_ACTIVITY_NEW_TASK
      context.startActivity(chooser)
    } catch (e: IOException) {
      Log.e("StreamViewModel", "Failed to share photo", e)
    }
  }

  private fun handleVideoFrame(videoFrame: VideoFrame) {
    // VideoFrame contains raw I420 video data in a ByteBuffer
    val buffer = videoFrame.buffer
    val dataSize = buffer.remaining()
    val byteArray = ByteArray(dataSize)

    // Save current position
    val originalPosition = buffer.position()
    buffer.get(byteArray)
    // Restore position
    buffer.position(originalPosition)

    // Convert I420 to NV21 format which is supported by Android's YuvImage
    val nv21 = convertI420toNV21(byteArray, videoFrame.width, videoFrame.height)
    val image = YuvImage(nv21, ImageFormat.NV21, videoFrame.width, videoFrame.height, null)
    val out =
        ByteArrayOutputStream().use { stream ->
          image.compressToJpeg(Rect(0, 0, videoFrame.width, videoFrame.height), 50, stream)
          stream.toByteArray()
        }

    val bitmap = BitmapFactory.decodeByteArray(out, 0, out.size)
    _uiState.update { it.copy(videoFrame = bitmap) }

    // Multi-destination streaming: distribute frame to enabled destinations
    if (_uiState.value.multiDestinationStreamingEnabled) {
      viewModelScope.launch {
        videoStreamingManager.distributeFrame(
            frameData = byteArray,
            width = videoFrame.width,
            height = videoFrame.height,
            timestamp = System.currentTimeMillis()
        )
      }
    }
  }

  // Convert I420 (YYYYYYYY:UUVV) to NV21 (YYYYYYYY:VUVU)
  private fun convertI420toNV21(input: ByteArray, width: Int, height: Int): ByteArray {
    val output = ByteArray(input.size)
    val size = width * height
    val quarter = size / 4

    input.copyInto(output, 0, 0, size) // Y is the same

    for (n in 0 until quarter) {
      output[size + n * 2] = input[size + quarter + n] // V first
      output[size + n * 2 + 1] = input[size + n] // U second
    }
    return output
  }

  private fun handlePhotoData(photo: PhotoData) {
    val capturedPhoto =
        when (photo) {
          is PhotoData.Bitmap -> photo.bitmap
          is PhotoData.HEIC -> {
            val byteArray = ByteArray(photo.data.remaining())
            photo.data.get(byteArray)

            // Extract EXIF transformation matrix and apply to bitmap
            val exifInfo = getExifInfo(byteArray)
            val transform = getTransform(exifInfo)
            decodeHeic(byteArray, transform)
          }
        }
    _uiState.update { it.copy(capturedPhoto = capturedPhoto, isShareDialogVisible = true) }
  }

  // HEIC Decoding with EXIF transformation
  private fun decodeHeic(heicBytes: ByteArray, transform: Matrix): Bitmap {
    val bitmap = BitmapFactory.decodeByteArray(heicBytes, 0, heicBytes.size)
    return applyTransform(bitmap, transform)
  }

  private fun getExifInfo(heicBytes: ByteArray): ExifInterface? {
    return try {
      ByteArrayInputStream(heicBytes).use { inputStream -> ExifInterface(inputStream) }
    } catch (e: IOException) {
      Log.w(TAG, "Failed to read EXIF from HEIC", e)
      null
    }
  }

  private fun getTransform(exifInfo: ExifInterface?): Matrix {
    val matrix = Matrix()

    if (exifInfo == null) {
      return matrix // Identity matrix (no transformation)
    }

    when (
        exifInfo.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
        )
    ) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> {
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_180 -> {
        matrix.postRotate(180f)
      }
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
        matrix.postScale(1f, -1f)
      }
      ExifInterface.ORIENTATION_TRANSPOSE -> {
        matrix.postRotate(90f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_90 -> {
        matrix.postRotate(90f)
      }
      ExifInterface.ORIENTATION_TRANSVERSE -> {
        matrix.postRotate(270f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_270 -> {
        matrix.postRotate(270f)
      }
      ExifInterface.ORIENTATION_NORMAL,
      ExifInterface.ORIENTATION_UNDEFINED -> {
        // No transformation needed
      }
    }

    return matrix
  }

  private fun applyTransform(bitmap: Bitmap, matrix: Matrix): Bitmap {
    if (matrix.isIdentity) {
      return bitmap
    }

    return try {
      val transformed = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
      if (transformed != bitmap) {
        bitmap.recycle()
      }
      transformed
    } catch (e: OutOfMemoryError) {
      Log.e(TAG, "Failed to apply transformation due to memory", e)
      bitmap
    }
  }

  // Multi-destination streaming control methods

  suspend fun enableComputerStreaming(ip: String, port: Int) {
    streamingPreferences.updateComputerEndpoint(
        _uiState.value.streamingConfiguration.computer.copy(
            ip = ip,
            port = port,
            enabled = true
        )
    )
  }

  suspend fun disableComputerStreaming() {
    streamingPreferences.updateComputerEndpoint(
        _uiState.value.streamingConfiguration.computer.copy(enabled = false)
    )
  }

  suspend fun enableCloudStreaming(userId: String) {
    streamingPreferences.updateCloudEndpoint(
        _uiState.value.streamingConfiguration.cloud.copy(
            userId = userId,
            enabled = true
        )
    )
  }

  suspend fun disableCloudStreaming() {
    streamingPreferences.updateCloudEndpoint(
        _uiState.value.streamingConfiguration.cloud.copy(enabled = false)
    )
  }

  suspend fun updateQualitySettings(targetFps: Int, sdkFrameRate: Int, jpegQuality: Int) {
    val currentConfig = _uiState.value.streamingConfiguration
    val wasComputerEnabled = currentConfig.computer.enabled
    val wasCloudEnabled = currentConfig.cloud.enabled

    // Update settings
    streamingPreferences.updateSettings(
        currentConfig.settings.copy(
            computerTargetFps = targetFps,
            sdkFrameRate = sdkFrameRate,
            jpegQuality = jpegQuality
        )
    )

    // Restart computer streaming if it was enabled (to apply new FPS/quality)
    if (wasComputerEnabled) {
      android.util.Log.i("StreamViewModel", "Restarting computer streaming with new settings: ${targetFps}fps, ${jpegQuality}% quality")
      videoStreamingManager.disableComputerStreaming()
      kotlinx.coroutines.delay(100) // Brief delay to ensure cleanup
      videoStreamingManager.enableComputerStreaming(
          endpoint = currentConfig.computer.ip,
          port = currentConfig.computer.port,
          targetFps = targetFps,
          jpegQuality = jpegQuality
      )
    }

    // Restart SDK stream if frame rate changed
    if (sdkFrameRate != currentConfig.settings.sdkFrameRate) {
      android.util.Log.i("StreamViewModel", "Restarting SDK stream with new frame rate: ${sdkFrameRate}fps")
      // Stream will be restarted on next startStream() call with updated settings
    }
  }

  fun getStreamingStats(): StateFlow<com.meta.wearable.dat.externalsampleapps.cameraaccess.streaming.StreamingStats> {
    return videoStreamingManager.statistics
  }

  override fun onCleared() {
    super.onCleared()
    stopStream()
    stateJob?.cancel()
    configJob?.cancel()
    statsJob?.cancel()
    videoStreamingManager.cleanup()
  }

  class Factory(
      private val application: Application,
      private val wearablesViewModel: WearablesViewModel,
  ) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
      if (modelClass.isAssignableFrom(StreamViewModel::class.java)) {
        @Suppress("UNCHECKED_CAST", "KotlinGenericsCast")
        return StreamViewModel(
            application = application,
            wearablesViewModel = wearablesViewModel,
        )
            as T
      }
      throw IllegalArgumentException("Unknown ViewModel class")
    }
  }
}
