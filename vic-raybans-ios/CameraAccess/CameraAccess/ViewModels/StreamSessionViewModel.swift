/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// StreamSessionViewModel.swift
//
// Core view model demonstrating video streaming from Meta wearable devices using the DAT SDK.
// This class showcases the key streaming patterns: device selection, session management,
// video frame handling, photo capture, and error handling.
//

import MWDATCamera
import MWDATCore
import SwiftUI
import CoreLocation

enum StreamingStatus {
  case streaming
  case waiting
  case stopped
}

@MainActor
class StreamSessionViewModel: ObservableObject {
  @Published var currentVideoFrame: UIImage?
  @Published var hasReceivedFirstFrame: Bool = false
  @Published var streamingStatus: StreamingStatus = .stopped
  @Published var showError: Bool = false
  @Published var errorMessage: String = ""
  @Published var hasActiveDevice: Bool = false

  var isStreaming: Bool {
    streamingStatus != .stopped
  }

  // Timer properties
  @Published var activeTimeLimit: StreamTimeLimit = .noLimit
  @Published var remainingTime: TimeInterval = 0

  // Photo capture properties
  @Published var capturedPhoto: UIImage?
  @Published var showPhotoPreview: Bool = false
  
  // Backend integration
  private var memoryCaptureManager: MemoryCaptureManager?
  @Published var isUploading: Bool = false
  @Published var uploadStatus: String = ""

  private var timerTask: Task<Void, Never>?
  // The core DAT SDK StreamSession - handles all streaming operations
  private var streamSession: StreamSession
  // Listener tokens are used to manage DAT SDK event subscriptions
  private var stateListenerToken: AnyListenerToken?
  private var videoFrameListenerToken: AnyListenerToken?
  private var errorListenerToken: AnyListenerToken?
  private var photoDataListenerToken: AnyListenerToken?
  private let wearables: WearablesInterface
  private let deviceSelector: AutoDeviceSelector
  private var deviceMonitorTask: Task<Void, Never>?

  // Speech Recognition
  public let speechRecognizer = SpeechRecognizer()
  
  // TTS Manager
  public let ttsManager = TTSManager()
  
  // Query Mode
  public let queryClient = QueryWebSocketClient()
  @Published public var isRecordingQuery: Bool = false
  @Published public var isProcessingQuery: Bool = false
  
  // Computed property for legacy compatibility if needed, or update views to use new props
  public var isQuerying: Bool { isRecordingQuery || isProcessingQuery }
  
  @Published public var queryStatus: String = ""
  @Published public var lastAIResponse: String = ""  // Caption of what AI said
  
  // Current Location (cached from view)
  public var currentLocation: CLLocationCoordinate2D?
  
  private var capturedTranscription: String?
  private var periodicCaptureTask: Task<Void, Never>?

  init(wearables: WearablesInterface) {
    self.wearables = wearables
    // Let the SDK auto-select from available devices
    self.deviceSelector = AutoDeviceSelector(wearables: wearables)
    let config = StreamSessionConfig(
      videoCodec: VideoCodec.raw,
      resolution: StreamingResolution.low,
      frameRate: 24)
    streamSession = StreamSession(streamSessionConfig: config, deviceSelector: deviceSelector)

    // Monitor device availability
    deviceMonitorTask = Task { @MainActor in
      for await device in deviceSelector.activeDeviceStream() {
        self.hasActiveDevice = device != nil
      }
    }

    // Subscribe to session state changes using the DAT SDK listener pattern
    stateListenerToken = streamSession.statePublisher.listen { [weak self] state in
      Task { @MainActor [weak self] in
        self?.updateStatusFromState(state)
      }
    }

    // Subscribe to video frames from the device camera
    videoFrameListenerToken = streamSession.videoFramePublisher.listen { [weak self] videoFrame in
      Task { @MainActor [weak self] in
        guard let self else { return }

        if let image = videoFrame.makeUIImage() {
          self.currentVideoFrame = image
          if !self.hasReceivedFirstFrame {
            self.hasReceivedFirstFrame = true
          }
        }
      }
    }

    // Subscribe to streaming errors
    errorListenerToken = streamSession.errorPublisher.listen { [weak self] error in
      Task { @MainActor [weak self] in
        guard let self else { return }
        let newErrorMessage = formatStreamingError(error)
        if newErrorMessage != self.errorMessage {
          showError(newErrorMessage)
        }
      }
    }

    updateStatusFromState(streamSession.state)

    // Subscribe to photo capture events
    photoDataListenerToken = streamSession.photoDataPublisher.listen { [weak self] photoData in
      Task { @MainActor [weak self] in
        guard let self else { return }
        if let uiImage = UIImage(data: photoData.data) {
          self.capturedPhoto = uiImage
          
          // BRANCH 0: Query Recording Mode - Store photo for query (button is being held)
          if self.isRecordingQuery {
            self.queryImage = uiImage
            NSLog("[StreamSessionViewModel] 📸 Query photo captured and stored")
            return
          }
          
          // BRANCH 1: Query Processing Mode (button released, sending query)
          if self.isProcessingQuery {
             self.uploadStatus = "Sending query..."
             let textToSend = self.capturedTranscription ?? self.speechRecognizer.transcribedText
             self.capturedTranscription = nil
             
             // Check if we actually have text or force a fallback
             let finalQueryText = textToSend.isEmpty ? "What am I looking at?" : textToSend
             
             do {
               try await self.queryClient.sendQuery(image: uiImage, text: finalQueryText)
               self.uploadStatus = "Query sent!"
             } catch {
               self.uploadStatus = "Query failed: \(error.localizedDescription)"
               self.ttsManager.speak("Sorry, I couldn't send your question.")
               self.isProcessingQuery = false
               self.startPeriodicCaptureTask()
             }
             return
          }

          // BRANCH 2: Regular Memory Capture
          // Ensure we don't upload background memories while the user is actively recording a query
          if let manager = self.memoryCaptureManager {
            self.isUploading = true
            self.uploadStatus = "Uploading to backend..."
            
            var textToSend = self.capturedTranscription ?? self.speechRecognizer.transcribedText
            if textToSend.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
              textToSend = "(silent)"
            }
            self.capturedTranscription = nil
            
            await manager.processCapture(photo: uiImage, transcription: textToSend)
            self.isUploading = false
            self.uploadStatus = "Upload complete!"
          }
        }
      }
    }
  }

  func handleStartStreaming() async {
    let permission = Permission.camera
    do {
      let status = try await wearables.checkPermissionStatus(permission)
      if status == .granted {
        await startSession()
        return
      }
      let requestStatus = try await wearables.requestPermission(permission)
      if requestStatus == .granted {
        await startSession()
        return
      }
      showError("Permission denied")
    } catch {
      showError("Permission error: \(error.description)")
    }
  }

  func startSession() async {
    // Reset to unlimited time when starting a new stream
    activeTimeLimit = .noLimit
    remainingTime = 0
    stopTimer()
    
    // Initialize memory capture manager and connect WebSocket
    if memoryCaptureManager == nil {
      // Pass the speech recognizer instance so manager can use it if needed
      memoryCaptureManager = MemoryCaptureManager(userId: "cass", speechRecognizer: speechRecognizer) // TODO: Get from settings
      memoryCaptureManager?.onCaptureComplete = { [weak self] captureId in
        Task { @MainActor [weak self] in
          self?.uploadStatus = "Backend processing: \(captureId)"
        }
      }
      memoryCaptureManager?.onError = { [weak self] error in
        Task { @MainActor [weak self] in
          self?.uploadStatus = "Error: \(error.localizedDescription)"
          self?.isUploading = false
        }
      }
    }
    
    // Start Speech Recognition
    do {
       let authorized = await speechRecognizer.requestAuthorization()
       if authorized {
         try speechRecognizer.startRecognition()
       } else {
         showError("Speech recognition permission denied")
       }
    } catch {
       NSLog("Speech recognition error: \(error)")
    }
    
    do {
      try memoryCaptureManager?.connect()
      setupQueryClient()
      try queryClient.connect()
    } catch {
      NSLog("[StreamSessionViewModel] Failed to connect WebSocket: \(error)")
    }

    await streamSession.start()
    
    // Start the 20-second capture loop
    startPeriodicCaptureTask()
  }

  private func showError(_ message: String) {
    errorMessage = message
    showError = true
  }

  func stopSession() async {
    stopTimer()
    stopPeriodicCaptureTask()
    speechRecognizer.stopRecognition()
    memoryCaptureManager?.disconnect()
    await streamSession.stop()
  }

  func dismissError() {
    showError = false
    errorMessage = ""
  }

  func setTimeLimit(_ limit: StreamTimeLimit) {
    activeTimeLimit = limit
    remainingTime = limit.durationInSeconds ?? 0

    if limit.isTimeLimited {
      startTimer()
    } else {
      stopTimer()
    }
  }

  func capturePhoto() {
    streamSession.capturePhoto(format: .jpeg)
  }
  
  // Periodic Capture Logic (Every 20 seconds)
  private func startPeriodicCaptureTask() {
    stopPeriodicCaptureTask()
    periodicCaptureTask = Task { @MainActor [weak self] in
        while !Task.isCancelled {
            // Wait 20 seconds
            try? await Task.sleep(nanoseconds: 20 * NSEC_PER_SEC)
            guard !Task.isCancelled, let self else { break }
            
            // 1. Capture current text
            let text = self.speechRecognizer.transcribedText
            self.capturedTranscription = text
            
            // 2. Reset text for next turn
            self.speechRecognizer.resetText()
            
            // 3. Trigger Photo Capture (this will invoke the listener above)
            // Note: processCapture inside the listener will use self.capturedTranscription
            self.capturePhoto()
            
            NSLog("[StreamSessionViewModel] 📸 Auto-capture triggered. Text: \(text.prefix(20))...")
        }
    }
  }
  
  private func stopPeriodicCaptureTask() {
    periodicCaptureTask?.cancel()
    periodicCaptureTask = nil
  }

  func dismissPhotoPreview() {
    showPhotoPreview = false
    capturedPhoto = nil
  }

  private func startTimer() {
    stopTimer()
    timerTask = Task { @MainActor [weak self] in
      while let self, remainingTime > 0 {
        try? await Task.sleep(nanoseconds: NSEC_PER_SEC)
        guard !Task.isCancelled else { break }
        remainingTime -= 1
      }
      if let self, !Task.isCancelled {
        await stopSession()
      }
    }
  }

  private func stopTimer() {
    timerTask?.cancel()
    timerTask = nil
  }

  private func updateStatusFromState(_ state: StreamSessionState) {
    switch state {
    case .stopped:
      currentVideoFrame = nil
      streamingStatus = .stopped
    case .waitingForDevice, .starting, .stopping, .paused:
      streamingStatus = .waiting
    case .streaming:
      streamingStatus = .streaming
    }
  }

  private func formatStreamingError(_ error: StreamSessionError) -> String {
    switch error {
    case .internalError:
      return "An internal error occurred. Please try again."
    case .deviceNotFound:
      return "Device not found. Please ensure your device is connected."
    case .deviceNotConnected:
      return "Device not connected. Please check your connection and try again."
    case .timeout:
      return "The operation timed out. Please try again."
    case .videoStreamingError:
      return "Video streaming failed. Please try again."
    case .audioStreamingError:
      return "Audio streaming failed. Please try again."
    case .permissionDenied:
      return "Camera permission denied. Please grant permission in Settings."
    @unknown default:
      return "An unknown streaming error occurred."
    }
  }

  // MARK: - Query Handling
  // Timeout task for query responses
  private var queryTimeoutTask: Task<Void, Never>?
  
  private func setupQueryClient() {
    queryClient.onLog = { message in
         NSLog(message)
    }
    
    // When playback finishes (either TTS or Audio File), resume services
    ttsManager.onSpeechEnded = { [weak self] in
        Task { @MainActor [weak self] in
            NSLog("[StreamSessionViewModel] 🗣️ Speech ended. Resuming background tasks...")
            self?.startPeriodicCaptureTask()
            try? self?.speechRecognizer.startRecognition()
        }
    }
    
    queryClient.onAnswerReceived = { [weak self] answer in
      Task { @MainActor [weak self] in
        self?.cancelQueryTimeout()
        self?.queryStatus = "Response received"
        self?.lastAIResponse = answer  // Save for caption display
        // Update UI state to remove spinner
        self?.isProcessingQuery = false
        
        // Speak fallback only (this will trigger onSpeechEnded when done)
        self?.ttsManager.speak(answer)
      }
    }
    
    queryClient.onAudioReceived = { [weak self] base64Audio in
      Task { @MainActor [weak self] in
        self?.cancelQueryTimeout()
        // Note: Audio response doesn't have text, so we mark it as audio
        self?.lastAIResponse = "[Audio response playing...]"
        // Update UI state to remove spinner
        self?.isProcessingQuery = false
        
        self?.ttsManager.playAudioData(base64Audio)
      }
    }
    
    // Handle clarification requests from backend
    queryClient.onClarificationNeeded = { [weak self] message, options in
      Task { @MainActor [weak self] in
        self?.cancelQueryTimeout()
        NSLog("[StreamSessionViewModel] 🤔 Clarification needed: \(message)")
        self?.queryStatus = "Clarification needed"
        self?.isProcessingQuery = false
        
        // Speak the clarification message
        var speechText = message
        if !options.isEmpty {
          speechText += " Options are: " + options.joined(separator: ", ")
        }
        self?.lastAIResponse = speechText  // Save for caption
        self?.ttsManager.speak(speechText)
      }
    }
    
    queryClient.onError = { [weak self] error in
      Task { @MainActor [weak self] in
        self?.cancelQueryTimeout()
        self?.queryStatus = "Error: \(error.localizedDescription)"
        self?.ttsManager.speak("Sorry, something went wrong.")
        self?.isProcessingQuery = false
        
        // Resume immediately on error
        self?.startPeriodicCaptureTask()
        try? self?.speechRecognizer.startRecognition()
      }
    }
  }
  
  private func startQueryTimeout() {
    cancelQueryTimeout()
    queryTimeoutTask = Task { @MainActor [weak self] in
      // Wait 10 seconds for a response
      try? await Task.sleep(nanoseconds: 10 * NSEC_PER_SEC)
      guard !Task.isCancelled, let self, self.isProcessingQuery else { return }
      
      NSLog("[StreamSessionViewModel] ⏰ Query timeout - no response received")
      self.queryStatus = "No response"
      self.isProcessingQuery = false
      self.ttsManager.speak("Sorry, I didn't get a response. Please try again.")
      
      // Resume background tasks
      self.startPeriodicCaptureTask()
      try? self.speechRecognizer.startRecognition()
    }
  }
  
  private func cancelQueryTimeout() {
    queryTimeoutTask?.cancel()
    queryTimeoutTask = nil
  }

  // Store query image captured at button press
  private var queryImage: UIImage?
  
  public func startQueryInput() {
    NSLog("[StreamSessionViewModel] 🎤 startQueryInput() called - isRecording:\(isRecordingQuery), isProcessing:\(isProcessingQuery)")
    
    // If already recording or processing, ignore (prevents double-tap issues)
    if isRecordingQuery {
      NSLog("[StreamSessionViewModel] ⚠️ Already recording, ignoring startQueryInput")
      return
    }
    
    // Stop any playing audio immediately
    ttsManager.stop()
    
    // Stop background context collection
    stopPeriodicCaptureTask()
    
    // Set recording state FIRST so photo listener knows to store the photo
    isRecordingQuery = true
    isProcessingQuery = false 
    queryStatus = "Listening..."
    queryImage = nil  // Clear any previous query image
    lastAIResponse = ""  // Clear previous AI response caption
    
    // Reset text for the new query
    speechRecognizer.resetText()
    
    // IMMEDIATELY trigger photo capture - this captures what the user is looking at
    // The photo listener (BRANCH 0) will store it in queryImage
    capturePhoto()
    NSLog("[StreamSessionViewModel] 📸 Photo capture triggered")
    
    // Start listening for speech at the same time
    try? speechRecognizer.startRecognition()
    
    NSLog("[StreamSessionViewModel] ✅ Query input started - capturing photo + listening")
  }
  
  public func finishQueryInput() {
    NSLog("[StreamSessionViewModel] 🎤 finishQueryInput() called - isRecording:\(isRecordingQuery), isProcessing:\(isProcessingQuery)")
    
    guard isRecordingQuery else {
      NSLog("[StreamSessionViewModel] ⚠️ Not recording, ignoring finishQueryInput")
      return
    }
    
    // 1. Capture the query text FIRST
    let transcription = speechRecognizer.transcribedText.trimmingCharacters(in: .whitespacesAndNewlines)
    NSLog("[StreamSessionViewModel] 📝 Captured transcription: '\(transcription)'")
    
    // Update recording state immediately
    isRecordingQuery = false
    
    // 2. Check if we actually got any speech
    if transcription.isEmpty {
      NSLog("[StreamSessionViewModel] ⚠️ No transcription captured - resetting")
      queryStatus = "No recording detected"
      isProcessingQuery = false
      queryImage = nil
      
      // Resume background tasks
      startPeriodicCaptureTask()
      try? speechRecognizer.startRecognition()
      
      // Clear status after a short delay
      Task { @MainActor in
        try? await Task.sleep(nanoseconds: 2 * NSEC_PER_SEC)
        if self.queryStatus == "No recording detected" {
          self.queryStatus = ""
        }
      }
      return
    }
    
    // 3. We have transcription - proceed with query
    isProcessingQuery = true
    queryStatus = "Processing query..."
    
    // 4. PAUSE Transcription Service
    speechRecognizer.stopRecognition()
    
    // 5. Send query directly using the image captured at button press
    let imageToSend = queryImage
    queryImage = nil  // Clear for next use
    
    Task { @MainActor in
      do {
        // Include location if available
        let lat = self.currentLocation?.latitude
        let lon = self.currentLocation?.longitude
        
        try await self.queryClient.sendQuery(
            image: imageToSend, 
            text: transcription,
            latitude: lat,
            longitude: lon
        )
        self.uploadStatus = "Query sent!"
        NSLog("[StreamSessionViewModel] ✅ Query sent successfully")
        
        // Start timeout - if no response in 10 seconds, show fallback
        self.startQueryTimeout()
      } catch {
        self.uploadStatus = "Query failed: \(error.localizedDescription)"
        self.queryStatus = "Query failed"
        self.ttsManager.speak("Sorry, I couldn't send your question.")
        self.isProcessingQuery = false
        self.startPeriodicCaptureTask()
        try? self.speechRecognizer.startRecognition()
      }
    }
  }
}
