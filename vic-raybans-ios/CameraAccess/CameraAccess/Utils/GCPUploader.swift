/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// GCPUploader.swift
//
// Service for uploading images and transcribed text to GCP via HTTP.
// Supports multipart form data uploads with file and text fields.
//
// For Po-Lin (Ray Bans / iOS):
// - Upload Photos: POST https://memory-backend-328251955578.us-east1.run.app/upload/{captureId}
// - WebSocket: wss://memory-backend-328251955578.us-east1.run.app/ws/ios/{userId}
//

import Foundation
import UIKit

class GCPUploader {
  // GCP endpoint base URL - per backend documentation for Ray Bans / iOS
  // HTTP: POST https://memory-backend-328251955578.us-east1.run.app/upload/{captureId}
  // WebSocket: wss://memory-backend-328251955578.us-east1.run.app/ws/ios/{userId}
  var baseURL: String = "https://memory-backend-328251955578.us-east1.run.app"
  var userId: String = "cass" // User ID for uploads (default: "cass")
  
  // Custom captureId generator - if nil, uses datetime format
  // Example: { "capture-\(Date().timeIntervalSince1970)" } for timestamp-based IDs
  var captureIdGenerator: (() -> String)? = nil
  
  init(baseURL: String = "https://memory-backend-328251955578.us-east1.run.app", userId: String = "cass") {
    self.baseURL = baseURL
    self.userId = userId
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
  
  /// Upload response containing photo URL and other metadata
  struct UploadResponse {
    let success: Bool
    let photoURL: String?
    let audioURL: String?
    let responseData: Data?
  }
  
  /// Upload image and text to GCP endpoint
  /// - Parameters:
  ///   - image: The image to upload
  ///   - text: The transcribed text to upload
  ///   - captureId: Unique identifier for this capture (UUID or timestamp). If nil, uses generateCaptureId()
  /// - Returns: UploadResponse with success status and parsed URLs
  func upload(image: UIImage, text: String, captureId: String? = nil) async throws -> UploadResponse {
    let finalCaptureId = captureId ?? generateCaptureId()
    // Build full endpoint URL with captureId: /upload/{captureId}
    let endpointPath = "/upload/\(finalCaptureId)"
    let fullURL = baseURL.hasSuffix("/") ? 
      "\(baseURL.dropLast())\(endpointPath)" : 
      "\(baseURL)\(endpointPath)"
    
    guard let url = URL(string: fullURL) else {
      NSLog("[GCPUploader] Invalid endpoint URL: \(fullURL)")
      throw GCPUploaderError.invalidURL
    }
    
    NSLog("[GCPUploader] Uploading to: \(fullURL)")
    
    // Convert image to JPEG data
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
      NSLog("[GCPUploader] Failed to convert image to JPEG")
      throw GCPUploaderError.imageConversionFailed
    }
    
    let imageSizeKB = Double(imageData.count) / 1024.0
    let textSize = text.data(using: .utf8)?.count ?? 0
    NSLog("[GCPUploader] Data size - Image: \(String(format: "%.1f", imageSizeKB)) KB, Text: \(textSize) bytes")
    
    // Create multipart form data
    let boundary = UUID().uuidString
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    request.timeoutInterval = 30.0 // 30 second timeout
    
    var body = Data()
    
    // Add file field (backend expects "file" not "image")
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    body.append(imageData)
    body.append("\r\n".data(using: .utf8)!)
    
    // Add text field
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"text\"\r\n\r\n".data(using: .utf8)!)
    body.append(text.data(using: .utf8)!)
    body.append("\r\n".data(using: .utf8)!)
    
    // Add user_id field if provided
    if !userId.isEmpty {
      body.append("--\(boundary)\r\n".data(using: .utf8)!)
      body.append("Content-Disposition: form-data; name=\"user_id\"\r\n\r\n".data(using: .utf8)!)
      body.append(userId.data(using: .utf8)!)
      body.append("\r\n".data(using: .utf8)!)
    }
    
    // End boundary
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)
    
    request.httpBody = body
    request.setValue("\(body.count)", forHTTPHeaderField: "Content-Length")
    
    let totalSizeKB = Double(body.count) / 1024.0
    NSLog("[GCPUploader] Total payload size: \(String(format: "%.1f", totalSizeKB)) KB")
    
    // Perform request
    let startTime = Date()
    NSLog("[GCPUploader] Sending HTTP request...")
    do {
      let (data, response) = try await URLSession.shared.data(for: request)
      let duration = Date().timeIntervalSince(startTime)
      NSLog("[GCPUploader] Received response after \(String(format: "%.2f", duration))s")
      
      guard let httpResponse = response as? HTTPURLResponse else {
        NSLog("[GCPUploader] Invalid response type")
        throw GCPUploaderError.invalidResponse
      }
      
      let statusCode = httpResponse.statusCode
      let responseSize = data.count
      
      NSLog("[GCPUploader] Response - Status: \(statusCode), Size: \(responseSize) bytes, Duration: \(String(format: "%.2f", duration))s")
      
      if (200...299).contains(statusCode) {
        let responseString = String(data: data, encoding: .utf8) ?? ""
        NSLog("[GCPUploader] ✅ Upload successful! Status: \(statusCode)")
        if !responseString.isEmpty {
          NSLog("[GCPUploader] Response body: \(responseString.prefix(200))")
        }
        
        // Parse response JSON to extract photoURL and audioURL
        var photoURL: String? = nil
        var audioURL: String? = nil
        
        if let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
          // Backend returns "url" field, but we also check "photoURL" for compatibility
          photoURL = json["url"] as? String ?? json["photoURL"] as? String
          audioURL = json["audioURL"] as? String
          
          if let photoURL = photoURL {
            NSLog("[GCPUploader] Parsed photoURL: \(photoURL)")
          }
          if let audioURL = audioURL {
            NSLog("[GCPUploader] Parsed audioURL: \(audioURL)")
          }
        }
        
        // If no URL in response, construct from captureId
        if photoURL == nil {
          photoURL = "https://storage.googleapis.com/reality-hack-2026-raw-media/memories/\(finalCaptureId)/photo.jpg"
          NSLog("[GCPUploader] Constructed photoURL from captureId: \(photoURL ?? "nil")")
        }
        
        return UploadResponse(success: true, photoURL: photoURL, audioURL: audioURL, responseData: data)
      } else {
        let responseString = String(data: data, encoding: .utf8) ?? "Unable to decode response"
        let errorMessage = "Status: \(statusCode) | Response: \(responseString.prefix(500))"
        NSLog("[GCPUploader] ❌ Upload failed with status \(statusCode)")
        NSLog("[GCPUploader] Response body: \(responseString)")
        NSLog("[GCPUploader] Response headers: \(httpResponse.allHeaderFields)")
        throw GCPUploaderError.serverError(statusCode: statusCode, message: responseString)
      }
    } catch {
      let duration = Date().timeIntervalSince(startTime)
      NSLog("[GCPUploader] Network error after \(String(format: "%.2f", duration))s: \(error.localizedDescription)")
      throw error
    }
  }
}

enum GCPUploaderError: Error, LocalizedError {
  case invalidURL
  case imageConversionFailed
  case invalidResponse
  case uploadFailed
  case serverError(statusCode: Int, message: String)
  
  var errorDescription: String? {
    switch self {
    case .invalidURL:
      return "Invalid endpoint URL"
    case .imageConversionFailed:
      return "Failed to convert image to JPEG"
    case .invalidResponse:
      return "Invalid response from server"
    case .uploadFailed:
      return "Upload failed"
    case .serverError(let statusCode, let message):
      return "Server error \(statusCode): \(message.prefix(200))"
    }
  }
}
