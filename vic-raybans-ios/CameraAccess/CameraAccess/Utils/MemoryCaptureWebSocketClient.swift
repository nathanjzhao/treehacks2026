/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// MemoryCaptureWebSocketClient.swift
//
// WebSocket client for sending memory capture messages to GCP backend.
// Connects to: wss://memory-backend-328251955578.us-east1.run.app/ws/ios/{userId}
// Sends memory_capture messages with photoURL, transcription, etc.
//

import Foundation

@MainActor
class MemoryCaptureWebSocketClient: NSObject, URLSessionWebSocketDelegate {
  var webSocketTask: URLSessionWebSocketTask?
  private var urlSession: URLSession?
  var baseURL: String = "https://memory-backend-328251955578.us-east1.run.app"
  var userId: String = ""
  private var isConnected: Bool = false
  
  // Callbacks
  var onAckReceived: ((String) -> Void)? // captureId from ack
  var onError: ((Error) -> Void)?
  
  // Custom captureId generator - if nil, uses datetime format
  // Example: { "capture-\(Date().timeIntervalSince1970)" } for timestamp-based IDs
  var captureIdGenerator: (() -> String)? = nil
  
  init(baseURL: String = "https://memory-backend-328251955578.us-east1.run.app", userId: String = "") {
    self.baseURL = baseURL
    self.userId = userId
    super.init()
    // Set default datetime-based captureId generator
    self.captureIdGenerator = { [weak self] in
      self?.generateDatetimeCaptureId() ?? UUID().uuidString
    }
  }
  
  /// Generate datetime-based captureId (default format: yyyyMMddHHmmss)
  private func generateDatetimeCaptureId() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMddHHmmss"
    formatter.timeZone = TimeZone.current
    return formatter.string(from: Date())
  }
  
  /// Generate captureId using custom generator or default datetime
  func generateCaptureId() -> String {
    if let generator = captureIdGenerator {
      return generator()
    }
    return generateDatetimeCaptureId()
  }
  
  /// Connect to WebSocket endpoint
  func connect() throws {
    guard !userId.isEmpty else {
      throw MemoryCaptureWebSocketError.missingUserId
    }
    
    // Convert HTTP URL to WebSocket URL
    let wsURLString = baseURL
      .replacingOccurrences(of: "https://", with: "wss://")
      .replacingOccurrences(of: "http://", with: "ws://")
    let endpoint = "\(wsURLString)/ws/ios/\(userId)"
    
    guard let url = URL(string: endpoint) else {
      throw MemoryCaptureWebSocketError.invalidURL
    }
    
    let session = URLSession(configuration: .default, delegate: self, delegateQueue: OperationQueue())
    webSocketTask = session.webSocketTask(with: url)
    webSocketTask?.resume()
    urlSession = session
    receiveMessage() // Start receiving messages
    NSLog("[MemoryCaptureWebSocketClient] Connecting to: \(endpoint)")
  }
  
  /// Disconnect from WebSocket
  func disconnect() {
    webSocketTask?.cancel(with: .goingAway, reason: nil)
    webSocketTask = nil
    urlSession = nil
    isConnected = false
    NSLog("[MemoryCaptureWebSocketClient] Disconnected")
  }
  
  /// Send memory capture message
  /// - Parameters:
  ///   - captureId: Unique identifier for this capture. If nil, uses generateCaptureId()
  ///   - timestamp: ISO 8601 timestamp
  ///   - photoURL: Full GCS URL to photo (optional)
  ///   - audioURL: Full GCS URL to audio (optional)
  ///   - transcription: Transcribed text (optional)
  func sendMemoryCapture(
    captureId: String? = nil,
    timestamp: String,
    photoURL: String? = nil,
    audioURL: String? = nil,
    transcription: String? = nil
  ) throws {
    let finalCaptureId = captureId ?? generateCaptureId()
    
    // Auto-connect if not connected
    if webSocketTask == nil || !isConnected {
      NSLog("[MemoryCaptureWebSocketClient] WebSocket not connected, attempting to connect...")
      try connect()
    }
    
    guard let webSocketTask = webSocketTask else {
      NSLog("[MemoryCaptureWebSocketClient] ❌ Cannot send: WebSocket task is nil after connect attempt")
      throw MemoryCaptureWebSocketError.notConnected
    }
    
    // Check WebSocket task state - URLSessionWebSocketTask.state is a URLSessionTask.State enum
    // Valid states: .running, .suspended, .canceling, .completed
    // We can only send when state is .running
    let taskState = webSocketTask.state
    if taskState != .running {
      NSLog("[MemoryCaptureWebSocketClient] ⚠️ WebSocket task state is \(taskState), not .running. Attempting to reconnect...")
      // Disconnect and reconnect
      disconnect()
      try connect()
      guard let newTask = self.webSocketTask, newTask.state == .running else {
        NSLog("[MemoryCaptureWebSocketClient] ❌ Failed to establish connection after reconnect attempt")
        throw MemoryCaptureWebSocketError.notConnected
      }
    }
    
    var capture: [String: Any] = [
      "type": "memory_capture",
      "id": finalCaptureId,
      "timestamp": timestamp
    ]
    
    if let photoURL = photoURL {
      capture["photoURL"] = photoURL
    }
    
    if let audioURL = audioURL {
      capture["audioURL"] = audioURL
    }
    
    // Always include transcription field, even if empty
    // Backend expects this field to be present
    if let transcription = transcription {
      capture["transcription"] = transcription
      NSLog("[MemoryCaptureWebSocketClient] Including transcription (length: \(transcription.count))")
    } else {
      capture["transcription"] = ""
      NSLog("[MemoryCaptureWebSocketClient] ⚠️ No transcription provided, sending empty string")
    }
    
    guard let jsonData = try? JSONSerialization.data(withJSONObject: capture, options: []) else {
      throw MemoryCaptureWebSocketError.jsonEncodingFailed
    }
    
    guard let jsonString = String(data: jsonData, encoding: .utf8) else {
      throw MemoryCaptureWebSocketError.jsonEncodingFailed
    }
    
    // Log message details for debugging
    let transcriptionValue = capture["transcription"] as? String ?? ""
    let transcriptionLen = transcriptionValue.count
    let hasPhoto = capture["photoURL"] != nil
    let hasAudio = capture["audioURL"] != nil
    NSLog("[MemoryCaptureWebSocketClient] 📤 Sending memory capture - id: \(finalCaptureId), transcription_len: \(transcriptionLen), photo: \(hasPhoto ? "yes" : "no"), audio: \(hasAudio ? "yes" : "no")")
    
    let message = URLSessionWebSocketTask.Message.string(jsonString)
    
    webSocketTask.send(message) { error in
      if let error = error {
        NSLog("[MemoryCaptureWebSocketClient] Failed to send message: \(error.localizedDescription)")
        Task { @MainActor [weak self] in
          self?.onError?(error)
        }
      } else {
        NSLog("[MemoryCaptureWebSocketClient] ✅ Sent memory capture (id: \(finalCaptureId), transcription_len: \(transcriptionLen))")
      }
    }
  }
  
  /// Receive messages from WebSocket
  private func receiveMessage() {
    webSocketTask?.receive { [weak self] result in
      guard let self = self else { return }
      
      switch result {
      case .success(let message):
        switch message {
        case .string(let text):
          self.handleTextMessage(text)
        case .data(let data):
          if let text = String(data: data, encoding: .utf8) {
            self.handleTextMessage(text)
          }
        @unknown default:
          NSLog("[MemoryCaptureWebSocketClient] Unknown message type")
        }
        self.receiveMessage() // Continue receiving messages
      case .failure(let error):
        NSLog("[MemoryCaptureWebSocketClient] WebSocket receive error: \(error.localizedDescription)")
        Task { @MainActor [weak self] in
          self?.onError?(error)
        }
      }
    }
  }
  
  /// Handle incoming text message
  private func handleTextMessage(_ text: String) {
    guard let data = text.data(using: .utf8),
          let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
      NSLog("[MemoryCaptureWebSocketClient] Failed to parse JSON response: \(text)")
      Task { @MainActor [weak self] in
        self?.onError?(MemoryCaptureWebSocketError.jsonParsingFailed)
      }
      return
    }
    
    // Check for ack response
    if let type = json["type"] as? String, type == "ack",
       let captureId = json["captureId"] as? String {
      NSLog("[MemoryCaptureWebSocketClient] Received ack for captureId: \(captureId)")
      Task { @MainActor [weak self] in
        self?.onAckReceived?(captureId)
      }
    } else if let ok = json["ok"] as? Bool, !ok, let errorDetail = json["error"] as? String {
      NSLog("[MemoryCaptureWebSocketClient] Backend error: \(errorDetail)")
      Task { @MainActor [weak self] in
        self?.onError?(MemoryCaptureWebSocketError.backendError(errorDetail))
      }
    } else {
      NSLog("[MemoryCaptureWebSocketClient] Unexpected JSON format: \(text)")
      Task { @MainActor [weak self] in
        self?.onError?(MemoryCaptureWebSocketError.unexpectedResponseFormat)
      }
    }
  }
  
  // MARK: - URLSessionWebSocketDelegate
  
  func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
    isConnected = true
    NSLog("[MemoryCaptureWebSocketClient] WebSocket connected")
  }
  
  func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
    isConnected = false
    let reasonString = reason.flatMap { String(data: $0, encoding: .utf8) } ?? "No reason"
    NSLog("[MemoryCaptureWebSocketClient] WebSocket disconnected with code: \(closeCode.rawValue), reason: \(reasonString)")
    Task { @MainActor [weak self] in
      self?.onError?(MemoryCaptureWebSocketError.disconnected(reasonString))
    }
  }
}

enum MemoryCaptureWebSocketError: Error, LocalizedError {
  case missingUserId
  case invalidURL
  case notConnected
  case jsonEncodingFailed
  case jsonParsingFailed
  case unexpectedResponseFormat
  case backendError(String)
  case disconnected(String)
  
  var errorDescription: String? {
    switch self {
    case .missingUserId:
      return "User ID is required for WebSocket connection."
    case .invalidURL:
      return "Invalid WebSocket URL."
    case .notConnected:
      return "WebSocket is not connected."
    case .jsonEncodingFailed:
      return "Failed to encode JSON message."
    case .jsonParsingFailed:
      return "Failed to parse JSON response from backend."
    case .unexpectedResponseFormat:
      return "Received unexpected response format from backend."
    case .backendError(let detail):
      return "Backend error: \(detail)"
    case .disconnected(let reason):
      return "WebSocket disconnected: \(reason)"
    }
  }
}

