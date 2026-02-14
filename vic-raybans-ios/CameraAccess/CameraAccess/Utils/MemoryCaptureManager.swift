/**
 * MemoryCaptureManager.swift
 * 
 * Integrates photo capture from Ray-Ban glasses with backend upload and WebSocket.
 * Handles the complete flow: Capture → Upload → Send to Backend → Process
 */

import Foundation
import UIKit

@MainActor
class MemoryCaptureManager {
  private let uploader: GCPUploader
  private let webSocketClient: MemoryCaptureWebSocketClient
  private let speechRecognizer: SpeechRecognizer?
  
  var userId: String {
    get { uploader.userId }
    set {
      uploader.userId = newValue
      webSocketClient.userId = newValue
    }
  }
  
  // Callbacks
  var onCaptureComplete: ((String) -> Void)? // captureId
  var onError: ((Error) -> Void)?
  
  init(userId: String = "cass", speechRecognizer: SpeechRecognizer? = nil) {
    self.uploader = GCPUploader(userId: userId)
    self.webSocketClient = MemoryCaptureWebSocketClient(userId: userId)
    self.speechRecognizer = speechRecognizer
    
    // Set up WebSocket callbacks
    webSocketClient.onAckReceived = { [weak self] captureId in
      NSLog("[MemoryCaptureManager] ✅ Received ack for capture: \(captureId)")
      self?.onCaptureComplete?(captureId)
    }
    
    webSocketClient.onError = { [weak self] error in
      NSLog("[MemoryCaptureManager] ❌ WebSocket error: \(error.localizedDescription)")
      self?.onError?(error)
    }
  }
  
  /// Process a captured photo: Upload → Send to Backend
  /// - Parameters:
  ///   - photo: The captured UIImage
  ///   - transcription: Optional transcription text
  func processCapture(photo: UIImage, transcription: String? = nil) async {
    let captureId = uploader.generateCaptureId()
    let timestamp = ISO8601DateFormatter().string(from: Date())
    
    NSLog("[MemoryCaptureManager] 📸 Processing capture: \(captureId)")
    
    do {
      // Step 1: Upload photo to backend
      NSLog("[MemoryCaptureManager] 📤 Uploading photo...")
      let uploadResult = try await uploader.upload(
        image: photo,
        text: transcription ?? "",
        captureId: captureId
      )
      
      guard uploadResult.success else {
        throw NSError(domain: "MemoryCaptureManager", code: 1, userInfo: [
          NSLocalizedDescriptionKey: "Photo upload failed"
        ])
      }
      
      NSLog("[MemoryCaptureManager] ✅ Photo uploaded: \(uploadResult.photoURL ?? "no URL")")
      
      // Step 2: Get transcription if not provided
      var finalTranscription = transcription
      if finalTranscription == nil && speechRecognizer != nil {
        // Use speech recognizer's current transcription
        finalTranscription = speechRecognizer?.transcribedText
      }
      
      // Step 3: Send memory capture via WebSocket
      NSLog("[MemoryCaptureManager] 📡 Sending memory capture to backend...")
      try webSocketClient.sendMemoryCapture(
        captureId: captureId,
        timestamp: timestamp,
        photoURL: uploadResult.photoURL,
        audioURL: uploadResult.audioURL,
        transcription: finalTranscription
      )
      
      NSLog("[MemoryCaptureManager] ✅ Memory capture sent successfully!")
      
    } catch {
      NSLog("[MemoryCaptureManager] ❌ Error processing capture: \(error.localizedDescription)")
      onError?(error)
    }
  }
  
  /// Connect WebSocket (call when app starts or streaming begins)
  func connect() throws {
    try webSocketClient.connect()
  }
  
  /// Disconnect WebSocket (call when app closes or streaming stops)
  func disconnect() {
    webSocketClient.disconnect()
  }
}
