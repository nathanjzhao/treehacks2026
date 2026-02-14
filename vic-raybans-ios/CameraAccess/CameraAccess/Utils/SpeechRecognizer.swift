/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// SpeechRecognizer.swift
//
// Service for real-time speech recognition using iOS Speech Framework.
// Processes audio buffers and emits transcribed text updates.
//

import AVFoundation
import Foundation
import Speech

@MainActor
class SpeechRecognizer: ObservableObject {
  @Published var transcribedText: String = ""
  @Published var isRecognizing: Bool = false
  
  private var speechRecognizer: SFSpeechRecognizer?
  private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
  private var recognitionTask: SFSpeechRecognitionTask?
  private let audioEngine = AVAudioEngine()
  private var lastResetText: String = "" // Track text at last reset to show only new content
  
  init() {
    // Initialize with device locale
    speechRecognizer = SFSpeechRecognizer(locale: Locale.current)
  }
  
  func requestAuthorization() async -> Bool {
    await withCheckedContinuation { continuation in
      SFSpeechRecognizer.requestAuthorization { status in
        continuation.resume(returning: status == .authorized)
      }
    }
  }
  
  func startRecognition() throws {
    NSLog("[SpeechRecognizer] startRecognition() called")
    
    // Cancel previous task if exists
    stopRecognition()
    
    guard let recognizer = speechRecognizer, recognizer.isAvailable else {
      NSLog("[SpeechRecognizer] ❌ Recognizer unavailable")
      throw SpeechRecognizerError.recognizerUnavailable
    }
    
    // Audio session is already configured and activated in StreamSessionViewModel.startSession()
    // It uses .playAndRecord category which supports both input and output
    // Reference: Audio session is managed centrally, no need to reconfigure here
    NSLog("[SpeechRecognizer] Using shared audio session (playAndRecord) from StreamSessionViewModel")
    
    // Create recognition request
    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = true
    request.requiresOnDeviceRecognition = false
    
    // Get audio format from input node
    let inputNode = audioEngine.inputNode
    let recordingFormat = inputNode.inputFormat(forBus: 0)
    NSLog("[SpeechRecognizer] Audio format: \(recordingFormat)")
    
    // Install tap to feed audio to recognition request
    inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
      request.append(buffer)
    }
    
    // Start audio engine
    audioEngine.prepare()
    try audioEngine.start()
    NSLog("[SpeechRecognizer] ✅ Audio engine started")
    
    // Start recognition task
    recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
      Task { @MainActor [weak self] in
        guard let self else { return }
        
        if let error = error {
          // Handle errors silently for partial results
          // Error code 203 indicates cancellation, which is expected when stopping recognition
          let errorCode = (error as NSError).code
          NSLog("[SpeechRecognizer] Recognition error: \(error.localizedDescription) (code: \(errorCode))")
          if errorCode != 203 { // 203 is SFSpeechRecognitionError.cancelled
            self.isRecognizing = false
          }
          return
        }
        
        if let result = result {
          let fullText = result.bestTranscription.formattedString
          // Only show text that was added since last reset
          if fullText.hasPrefix(self.lastResetText) {
            let newText = String(fullText.dropFirst(self.lastResetText.count))
            self.transcribedText = newText.trimmingCharacters(in: .whitespacesAndNewlines)
          } else {
            // If text doesn't start with last reset text, show full text (fallback)
            self.transcribedText = fullText
          }
          self.isRecognizing = true
          
          NSLog("[SpeechRecognizer] Transcribed: \(self.transcribedText)")
          
          // Stop if final result
          if result.isFinal {
            self.isRecognizing = false
          }
        }
      }
    }
    
    recognitionRequest = request
    isRecognizing = true
    NSLog("[SpeechRecognizer] ✅ Recognition started successfully")
  }
  
  func stopRecognition() {
    NSLog("[SpeechRecognizer] stopRecognition() called - isRecognizing: \(isRecognizing)")
    
    recognitionTask?.cancel()
    recognitionTask = nil
    recognitionRequest?.endAudio()
    recognitionRequest = nil
    
    if audioEngine.isRunning {
      audioEngine.stop()
      NSLog("[SpeechRecognizer] Audio engine stopped")
    }
    audioEngine.inputNode.removeTap(onBus: 0)
    
    // Don't deactivate audio session - DAT SDK manages it
    // try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    
    isRecognizing = false
  }
  
  func resetText() {
    NSLog("[SpeechRecognizer] resetText() called - clearing transcription")
    // Clear everything for a fresh start
    lastResetText = ""
    transcribedText = ""
  }
}

enum SpeechRecognizerError: Error {
  case recognizerUnavailable
  case authorizationDenied
  case engineError
  case audioSessionError
}
