/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// QueryWebSocketClient.swift
//
// WebSocket client for querying the memory system via /ws/query/{userId} endpoint.
// Sends image and text query, receives text answer or audio response.
//
//erfef

import Foundation
import UIKit

@MainActor
class QueryWebSocketClient: NSObject {
  var webSocketTask: URLSessionWebSocketTask?
  private var urlSession: URLSession?
  var baseURL: String = "https://memory-backend-328251955578.us-east1.run.app"
  var userId: String = "cass"
  
  // Callbacks
  var onAnswerReceived: ((String) -> Void)? // Text answer
  var onAudioReceived: ((String) -> Void)? // Base64 audio string
  var onClarificationNeeded: ((String, [String]) -> Void)? // Message and options
  var onError: ((Error) -> Void)?
  var onLog: ((String) -> Void)? // Debug log callback for UI display
  
  init(baseURL: String = "https://memory-backend-328251955578.us-east1.run.app", userId: String = "cass") {
    self.baseURL = baseURL
    self.userId = userId
    super.init()
  }
  
  /// Log message to both NSLog and UI callback
  private func log(_ message: String) {
    NSLog(message)
    onLog?(message)
  }
  
  /// Connect to WebSocket endpoint
  func connect() throws {
    log("[QueryWebSocketClient] 🔌 connect() called")
    
    guard !userId.isEmpty else {
      log("[QueryWebSocketClient] ❌ connect() failed: userId is empty")
      throw QueryWebSocketError.missingUserId
    }
    
    log("[QueryWebSocketClient] ✅ userId: '\(userId)'")
    
    // Convert HTTP URL to WebSocket URL
    let wsURLString = baseURL
      .replacingOccurrences(of: "https://", with: "wss://")
      .replacingOccurrences(of: "http://", with: "ws://")
    let endpoint = "\(wsURLString)/ws/query/\(userId)"
    log("[QueryWebSocketClient] 🔗 Endpoint: \(endpoint)")
    
    guard let url = URL(string: endpoint) else {
      log("[QueryWebSocketClient] ❌ Invalid URL: \(endpoint)")
      throw QueryWebSocketError.invalidURL
    }
    
    log("[QueryWebSocketClient] ✅ URL created: \(url.absoluteString)")
    log("[QueryWebSocketClient] 📡 Creating URLSession...")
    
    let session = URLSession(configuration: .default, delegate: self, delegateQueue: OperationQueue())
    webSocketTask = session.webSocketTask(with: url)
    log("[QueryWebSocketClient] ✅ WebSocket task created")
    
    log("[QueryWebSocketClient] ▶️ Resuming WebSocket task...")
    webSocketTask?.resume()
    urlSession = session
    
    log("[QueryWebSocketClient] 👂 Starting message receiver...")
    receiveMessage() // Start receiving messages
    log("[QueryWebSocketClient] ✅ Connection initiated to: \(endpoint)")
  }
  
  /// Upload image to backend to get a URL
  /// - Parameters:
  ///   - image: Image to upload
  ///   - queryId: Unique ID for the query
  /// - Returns: The URL of the uploaded image
  func uploadImage(image: UIImage, queryId: String) async throws -> String {
    log("[QueryWebSocketClient] 📤 Uploading image for query: \(queryId)")
    
    let urlString = "\(baseURL)/query-upload/\(queryId)"
    guard let url = URL(string: urlString) else {
      throw QueryWebSocketError.invalidURL
    }
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
      throw QueryWebSocketError.imageConversionFailed
    }
    
    var body = Data()
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"query_image.jpg\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    body.append(imageData)
    body.append("\r\n".data(using: .utf8)!)
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)
    
    request.httpBody = body
    
    let sessionToUse = urlSession ?? URLSession.shared
    let (data, response) = try await sessionToUse.data(for: request)
    
    guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
        log("[QueryWebSocketClient] ❌ Upload failed with status: \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        throw QueryWebSocketError.uploadFailed
    }
    
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let imageUrl = json["url"] as? String else {
        log("[QueryWebSocketClient] ❌ Failed to parse response or missing 'url'")
        throw QueryWebSocketError.invalidResponse
    }
    
    log("[QueryWebSocketClient] ✅ Image uploaded successfully: \(imageUrl)")
    return imageUrl
  }
  
  /// Send query with image (via upload) and text
  func sendQuery(image: UIImage?, text: String, includeFaces: Bool = true, maxImages: Int = 8, latitude: Double? = nil, longitude: Double? = nil) async throws {
    log("[QueryWebSocketClient] 📤 sendQuery() called - text: '\(text)'")
    
    // Auto-reconnect if needed
    if webSocketTask == nil || webSocketTask?.state == .completed || webSocketTask?.state == .canceling {
        log("[QueryWebSocketClient] ⚠️ WebSocket not connected (state: \(webSocketTask?.state.rawValue ?? -1)), reconnecting...")
        try connect()
    }
    
    guard let webSocketTask = webSocketTask else {
      throw QueryWebSocketError.notConnected
    }
    
    var imageURL: String? = nil
    
    // 1. Upload image if present
    if let image = image {
        let queryId = UUID().uuidString
        imageURL = try await uploadImage(image: image, queryId: queryId)
    }
    
    // 2. Prepare JSON payload
    var request: [String: Any] = [
      "text": text,
      "includeFaces": includeFaces,
      "maxImages": maxImages
    ]
    
    if let url = imageURL {
        request["imageURL"] = url
    }
    
    // Add location if available
    if let lat = latitude, let lon = longitude {
        request["latitude"] = lat
        request["longitude"] = lon
        log("[QueryWebSocketClient] 📍 Adding location to query: \(lat), \(lon)")
    }
    
    guard let jsonData = try? JSONSerialization.data(withJSONObject: request, options: []),
          let jsonString = String(data: jsonData, encoding: .utf8) else {
      throw QueryWebSocketError.jsonEncodingFailed
    }
    
    let message = URLSessionWebSocketTask.Message.string(jsonString)
    try await webSocketTask.send(message)
    log("[QueryWebSocketClient] ✅ Query sent (WS)")
  }
  
  /// Receive messages from WebSocket
  private func receiveMessage() {
    log("[QueryWebSocketClient] 👂 Setting up message receiver...")
    webSocketTask?.receive { [weak self] result in
      guard let self = self else {
        NSLog("[QueryWebSocketClient] ⚠️ receiveMessage: self is nil")
        return
      }
      
      switch result {
      case .success(let message):
        self.log("[QueryWebSocketClient] ✅ Message received!")
        switch message {
        case .string(let text):
          self.log("[QueryWebSocketClient] 📝 Received string message: \(text.prefix(200))...")
          Task { @MainActor [weak self] in
            self?.handleTextMessage(text)
          }
        case .data(let data):
          self.log("[QueryWebSocketClient] 📦 Received data message: \(data.count) bytes")
          if let text = String(data: data, encoding: .utf8) {
            self.log("[QueryWebSocketClient] 📝 Converted data to string: \(text.prefix(200))...")
            Task { @MainActor [weak self] in
              self?.handleTextMessage(text)
            }
          } else {
            self.log("[QueryWebSocketClient] ❌ Failed to convert data to string")
          }
        @unknown default:
          self.log("[QueryWebSocketClient] ⚠️ Unknown message type")
        }
        
        // Continue receiving messages
        self.log("[QueryWebSocketClient] 🔄 Continuing to receive messages...")
        self.receiveMessage()
        
      case .failure(let error):
        self.log("[QueryWebSocketClient] ❌ Receive error: \(error.localizedDescription)")
        self.log("[QueryWebSocketClient] ❌ Error domain: \((error as NSError).domain), code: \((error as NSError).code)")
        Task { @MainActor [weak self] in
          self?.onError?(error)
        }
      }
    }
  }
  
  /// Handle text message from WebSocket
  private func handleTextMessage(_ text: String) {
    log("[QueryWebSocketClient] Received message: \(text.prefix(200))")
    
    log("[QueryWebSocketClient] 📨 handleTextMessage() - processing message")
    log("[QueryWebSocketClient] 📨 Message length: \(text.count) chars, preview: \(text.prefix(500))")
    
    guard let data = text.data(using: .utf8) else {
      log("[QueryWebSocketClient] ❌ Failed to convert text to data")
      return
    }
    
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      log("[QueryWebSocketClient] ❌ Failed to parse JSON response")
      log("[QueryWebSocketClient] ❌ Raw text: \(text)")
      return
    }
    
    log("[QueryWebSocketClient] ✅ JSON parsed successfully")
    log("[QueryWebSocketClient] 📋 JSON keys: \(json.keys.joined(separator: ", "))")
    
    // Handle different response types
    if let responseType = json["type"] as? String {
      log("[QueryWebSocketClient] 📋 Response type: '\(responseType)'")
      switch responseType {
      case "response":
        log("[QueryWebSocketClient] 📋 Processing 'response' type...")
        if let ok = json["ok"] as? Bool {
          log("[QueryWebSocketClient] 📋 Response ok: \(ok)")
          if ok {
            // PRIORITY: Check for audio FIRST - if present, use audio and skip TTS
            if let audio = json["audio"] as? String, !audio.isEmpty {
              log("[QueryWebSocketClient] ✅ Found audio field! Length: \(audio.count) chars")
              log("[QueryWebSocketClient] 📞 Calling onAudioReceived callback (PRIORITY)...")
              onAudioReceived?(audio)
              log("[QueryWebSocketClient] ✅ onAudioReceived callback called - skipping TTS")
            } else if let answer = json["answer"] as? String {
              // FALLBACK: Only use TTS if there's no audio
              log("[QueryWebSocketClient] ✅ Found answer field (no audio)! Length: \(answer.count) chars")
              log("[QueryWebSocketClient] ✅ Answer preview: \(answer.prefix(100))...")
              log("[QueryWebSocketClient] 📞 Calling onAnswerReceived callback (fallback)...")
              onAnswerReceived?(answer)
              log("[QueryWebSocketClient] ✅ onAnswerReceived callback called")
            } else {
              log("[QueryWebSocketClient] ⚠️ No 'answer' or 'audio' field in response")
            }
          } else {
            // Error response
            let errorMsg = json["error"] as? String ?? json["detail"] as? String ?? "Unknown error"
            log("[QueryWebSocketClient] ❌ Response ok=false, error: \(errorMsg)")
            let error = NSError(domain: "QueryWebSocket", code: 1, userInfo: [NSLocalizedDescriptionKey: errorMsg])
            log("[QueryWebSocketClient] 📞 Calling onError callback...")
            onError?(error)
            log("[QueryWebSocketClient] ✅ onError callback called")
          }
        } else {
          log("[QueryWebSocketClient] ⚠️ No 'ok' field in response")
        }
        
      case "clarification_needed":
        log("[QueryWebSocketClient] 📋 Processing 'clarification_needed' type...")
        let message = json["message"] as? String ?? "Clarification needed"
        let options = json["options"] as? [String] ?? []
        log("[QueryWebSocketClient] 📋 Clarification message: \(message)")
        log("[QueryWebSocketClient] 📋 Options count: \(options.count)")
        log("[QueryWebSocketClient] 📞 Calling onClarificationNeeded callback...")
        onClarificationNeeded?(message, options)
        log("[QueryWebSocketClient] ✅ onClarificationNeeded callback called")
        
      default:
        log("[QueryWebSocketClient] ⚠️ Unknown response type: '\(responseType)'")
        log("[QueryWebSocketClient] 📋 Full JSON: \(json)")
      }
    } else {
      log("[QueryWebSocketClient] ⚠️ No 'type' field in JSON, trying fallback extraction...")
      // Fallback: try to extract answer directly
      if let answer = json["answer"] as? String {
        log("[QueryWebSocketClient] ✅ Found answer in fallback: \(answer.prefix(100))...")
        log("[QueryWebSocketClient] 📞 Calling onAnswerReceived callback (fallback)...")
        onAnswerReceived?(answer)
      } else if let audio = json["audio"] as? String {
        log("[QueryWebSocketClient] ✅ Found audio in fallback: length \(audio.count)")
        log("[QueryWebSocketClient] 📞 Calling onAudioReceived callback (fallback)...")
        onAudioReceived?(audio)
      } else {
        log("[QueryWebSocketClient] ❌ Response does not contain answer or audio")
        log("[QueryWebSocketClient] 📋 Available keys: \(json.keys.joined(separator: ", "))")
        log("[QueryWebSocketClient] 📋 Full JSON: \(json)")
      }
    }
  }
  
  /// Disconnect from WebSocket
  func disconnect() {
    webSocketTask?.cancel(with: .goingAway, reason: nil)
    webSocketTask = nil
    urlSession = nil
    log("[QueryWebSocketClient] Disconnected")
  }
}

// MARK: - URLSessionWebSocketDelegate
extension QueryWebSocketClient: URLSessionWebSocketDelegate {
  func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
    log("[QueryWebSocketClient] ✅ WebSocket connection opened!")
    let protocolValue = `protocol` ?? "none"
    log("[QueryWebSocketClient] ✅ Protocol: \(protocolValue)")
    log("[QueryWebSocketClient] ✅ Task state: \(webSocketTask.state.rawValue)")
  }
  
  func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
    let reasonString = reason.flatMap { String(data: $0, encoding: .utf8) } ?? "unknown"
    log("[QueryWebSocketClient] ❌ WebSocket closed!")
    log("[QueryWebSocketClient] ❌ Close code: \(closeCode.rawValue)")
    log("[QueryWebSocketClient] ❌ Reason: \(reasonString)")
  }
}

enum QueryWebSocketError: Error {
  case missingUserId
  case invalidURL
  case notConnected
  case imageConversionFailed
  case jsonEncodingFailed
  case uploadFailed
  case invalidResponse
}

