#!/usr/bin/env swift

/*
 * Test script for testing backend WebSocket connections
 * Run with: swift test_backend.swift
 * 
 * This allows you to test the backend functionality without needing Xcode
 */

import Foundation

// Simple WebSocket test client
class BackendTester {
    let baseURL = "https://memory-backend-328251955578.us-east1.run.app"
    var userId: String
    
    init(userId: String = "test_user") {
        self.userId = userId
    }
    
    func testMemoryCaptureWebSocket() async {
        print("🧪 Testing Memory Capture WebSocket...")
        print("📍 Endpoint: \(baseURL)/ws/ios/\(userId)")
        
        let wsURL = baseURL
            .replacingOccurrences(of: "https://", with: "wss://")
            .replacingOccurrences(of: "http://", with: "ws://")
        let endpoint = "\(wsURL)/ws/ios/\(userId)"
        
        guard let url = URL(string: endpoint) else {
            print("❌ Invalid URL: \(endpoint)")
            return
        }
        
        print("🔌 Connecting to: \(endpoint)")
        
        let session = URLSession.shared
        let task = session.webSocketTask(with: url)
        task.resume()
        
        // Send test memory capture
        let captureId = "test-\(Int(Date().timeIntervalSince1970))"
        let timestamp = ISO8601DateFormatter().string(from: Date())
        
        let message: [String: Any] = [
            "type": "memory_capture",
            "id": captureId,
            "timestamp": timestamp,
            "transcription": "Test transcription from command line script"
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: message),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            print("❌ Failed to encode JSON")
            return
        }
        
        print("📤 Sending test capture: \(captureId)")
        print("📝 Message: \(jsonString)")
        
        task.send(URLSessionWebSocketTask.Message.string(jsonString)) { error in
            if let error = error {
                print("❌ Send error: \(error.localizedDescription)")
            } else {
                print("✅ Message sent successfully!")
            }
        }
        
        // Receive response
        task.receive { result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    print("✅ Received response: \(text)")
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        print("✅ Received response: \(text)")
                    }
                @unknown default:
                    print("⚠️ Unknown message type")
                }
            case .failure(let error):
                print("❌ Receive error: \(error.localizedDescription)")
            }
        }
        
        // Keep connection alive for a bit
        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
        task.cancel(with: .goingAway, reason: nil)
        print("🔌 Connection closed")
    }
    
    func testQueryWebSocket() async {
        print("\n🧪 Testing Query WebSocket...")
        print("📍 Endpoint: \(baseURL)/ws/query/\(userId)")
        
        let wsURL = baseURL
            .replacingOccurrences(of: "https://", with: "wss://")
            .replacingOccurrences(of: "http://", with: "ws://")
        let endpoint = "\(wsURL)/ws/query/\(userId)"
        
        guard let url = URL(string: endpoint) else {
            print("❌ Invalid URL: \(endpoint)")
            return
        }
        
        print("🔌 Connecting to: \(endpoint)")
        
        let session = URLSession.shared
        let task = session.webSocketTask(with: url)
        task.resume()
        
        // Send test query
        let query: [String: Any] = [
            "text": "What was I doing in the past 10 minutes?",
            "includeFaces": false,
            "maxImages": 0
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: query),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            print("❌ Failed to encode JSON")
            return
        }
        
        print("📤 Sending test query: What was I doing in the past 10 minutes?")
        
        task.send(URLSessionWebSocketTask.Message.string(jsonString)) { error in
            if let error = error {
                print("❌ Send error: \(error.localizedDescription)")
            } else {
                print("✅ Query sent successfully!")
            }
        }
        
        // Receive response
        task.receive { result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    print("✅ Received answer:")
                    print(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        print("✅ Received answer:")
                        print(text)
                    }
                @unknown default:
                    print("⚠️ Unknown message type")
                }
            case .failure(let error):
                print("❌ Receive error: \(error.localizedDescription)")
            }
        }
        
        // Keep connection alive for response
        try? await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds
        task.cancel(with: .goingAway, reason: nil)
        print("🔌 Connection closed")
    }
}

// Main execution
print("🚀 Backend Connection Tester")
print(String(repeating: "=", count: 50))

let tester = BackendTester(userId: "test_user")

Task {
    await tester.testMemoryCaptureWebSocket()
    await tester.testQueryWebSocket()
    print("\n✅ Testing complete!")
    exit(0)
}

// Keep script running
RunLoop.main.run()
