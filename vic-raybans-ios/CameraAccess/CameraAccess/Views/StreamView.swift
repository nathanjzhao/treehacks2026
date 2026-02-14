/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// StreamView.swift
//
// Main UI for video streaming from Meta wearable devices using the DAT SDK.
// This view demonstrates the complete streaming API: video streaming with real-time display, photo capture,
// and error handling.
//

import MWDATCore
import SwiftUI
import CoreLocation

struct StreamView: View {
  @ObservedObject var viewModel: StreamSessionViewModel
  @ObservedObject var wearablesVM: WearablesViewModel
  @EnvironmentObject var geofenceManager: GeofenceManager

  var body: some View {
    ZStack {
      // Black background for letterboxing/pillarboxing
      Color.black
        .edgesIgnoringSafeArea(.all)

      // Video backdrop
      if let videoFrame = viewModel.currentVideoFrame, viewModel.hasReceivedFirstFrame {
        GeometryReader { geometry in
          Image(uiImage: videoFrame)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: geometry.size.width, height: geometry.size.height)
            .clipped()
        }
        .edgesIgnoringSafeArea(.all)
      } else {
        ProgressView()
          .scaleEffect(1.5)
          .foregroundColor(.white)
      }

      // TTS Test Controls Overlay
      VStack {
        HStack(spacing: 12) {
          Spacer()
          Button(action: {
            viewModel.ttsManager.speak("Testing audio output one.")
          }) {
            Image(systemName: "speaker.wave.2.fill")
              .foregroundColor(.white)
              .padding(8)
              .background(Color.black.opacity(0.6))
              .clipShape(Circle())
          }
          
          Button(action: {
            viewModel.ttsManager.speak("Hey cass, don't forget to take your keys.")
          }) {
            Image(systemName: "exclamationmark.bubble.fill")
              .foregroundColor(.white)
              .padding(8)
              .background(Color.black.opacity(0.6))
              .clipShape(Circle())
          }
          
          // Geofence Button
          Button(action: {
            addTestGeofence()
          }) {
            Image(systemName: "mappin.circle.fill")
              .foregroundColor(geofenceManager.geofences.isEmpty ? .white : .green)
              .padding(8)
              .background(Color.black.opacity(0.6))
              .clipShape(Circle())
          }
          
          // Clear Geofences Button (only show if there are geofences)
          if !geofenceManager.geofences.isEmpty {
            Button(action: {
               geofenceManager.removeAll()
               viewModel.ttsManager.speak("All geofences cleared")
            }) {
              Image(systemName: "trash.circle.fill")
                .foregroundColor(.red)
                .padding(8)
                .background(Color.black.opacity(0.6))
                .clipShape(Circle())
            }
          }
        }
        .padding(.top, 50)
        .padding(.trailing, 20)
        
        // Geofence count indicator
        if !geofenceManager.geofences.isEmpty {
          Text("📍 \(geofenceManager.geofences.count) geofence(s)")
            .font(.caption)
            .foregroundColor(.white)
            .padding(6)
            .background(Color.black.opacity(0.6))
            .cornerRadius(8)
        }
        
        Spacer()
      }

      
      // Bottom controls layer
      VStack {
        Spacer()
        ControlsView(viewModel: viewModel)
      }
      .padding(.all, 24)
      
      // Query Status Overlay
      VStack {
        if !viewModel.queryStatus.isEmpty {
           Text(viewModel.queryStatus)
             .font(.headline)
             .padding()
             .background(Color.black.opacity(0.7))
             .foregroundColor(.white)
             .cornerRadius(12)
             .padding(.top, 100)
             .transition(.opacity)
        }
        Spacer()
      }

      // Top Middle Query Button
      VStack {
        HStack {
            Spacer()
            // Using DragGesture for reliable TouchDown/TouchUp behavior
            ZStack {
                Circle()
                   .fill(viewModel.isQuerying ? Color.red : Color.blue)
                   .frame(width: 80, height: 80)
                   .shadow(color: .black.opacity(0.3), radius: 8, x: 0, y: 4)
                   .scaleEffect(viewModel.isQuerying ? 1.2 : 1.0)
                   .animation(.spring(response: 0.3, dampingFraction: 0.6), value: viewModel.isQuerying)
                   
                Image(systemName: "questionmark")
                   .font(.system(size: 40, weight: .bold))
                   .foregroundColor(.white)
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if !viewModel.isQuerying {
                            let generator = UIImpactFeedbackGenerator(style: .medium)
                            generator.impactOccurred()
                            viewModel.startQueryInput()
                        }
                    }
                    .onEnded { _ in
                        let generator = UIImpactFeedbackGenerator(style: .medium)
                        generator.impactOccurred()
                        viewModel.finishQueryInput()
                    }
            )
            Spacer()
        }
        .padding(.top, 120) // Position below TTS controls
        Spacer()
      }
      
      // Transcription overlay
      VStack {
        Spacer()
        
        if viewModel.isProcessingQuery {
             ProgressView()
               .progressViewStyle(CircularProgressViewStyle(tint: .white))
               .scaleEffect(1.5)
               .padding(12)
               .background(Color.black.opacity(0.7))
               .cornerRadius(12)
               .padding(.bottom, 100)
        } else if !viewModel.lastAIResponse.isEmpty {
          // AI Response - Cyan color
          Text("🤖 \(viewModel.lastAIResponse)")
            .font(.system(size: 14))
            .foregroundColor(.black)
            .padding(12)
            .background(Color.cyan.opacity(0.9))
            .cornerRadius(8)
            .padding(.horizontal, 20)
            .padding(.bottom, 100)
            .multilineTextAlignment(.center)
            .lineLimit(5)
        } else if !viewModel.speechRecognizer.transcribedText.isEmpty {
          // User transcription - White on black
          Text(viewModel.speechRecognizer.transcribedText)
            .font(.system(size: 14))
            .foregroundColor(.white)
            .padding(12)
            .background(Color.black.opacity(0.7))
            .cornerRadius(8)
            .padding(.horizontal, 20)
            .padding(.bottom, 100)
        }
      }
      
      // Timer display area with fixed height
      VStack {
        Spacer()
        if viewModel.activeTimeLimit.isTimeLimited && viewModel.remainingTime > 0 {
          Text("Streaming ending in \(viewModel.remainingTime.formattedCountdown)")
            .font(.system(size: 15))
            .foregroundColor(.white)
        }
      }
    }
    .onAppear {
      // Setup geofencing
      setupGeofencing()
    }
    // Sync location to ViewModel for backend queries
    .onReceive(Timer.publish(every: 2.0, on: .main, in: .common).autoconnect()) { _ in
        if let loc = geofenceManager.currentLocation {
            viewModel.currentLocation = loc.coordinate
        }
    }
    .onDisappear {
      Task {
        if viewModel.streamingStatus != .stopped {
          await viewModel.stopSession()
        }
      }
    }
    // Show captured photos from DAT SDK in a preview sheet
    .sheet(isPresented: $viewModel.showPhotoPreview) {
      if let photo = viewModel.capturedPhoto {
        PhotoPreviewView(
          photo: photo,
          onDismiss: {
            viewModel.dismissPhotoPreview()
          }
        )
      }
    }
  }
  
  // MARK: - Geofencing Helpers
  
  private func setupGeofencing() {
    // Request permissions
    geofenceManager.requestPermissions()
    
    // Wire up enter callback to speak via TTS
    geofenceManager.onEnterGeofence = { geofence in
      if let message = geofence.enterMessage {
        viewModel.ttsManager.speak(message)
      }
    }
    
    // Start monitoring saved geofences
    geofenceManager.startMonitoringAll()
  }
  
  private func addTestGeofence() {
    // 1. Remove existing "My Location" geofences to avoid clutter
    let existing = geofenceManager.geofences.filter { $0.name == "My Location" || $0.name == "Test Location" }
    for geo in existing {
        geofenceManager.removeGeofence(geo)
    }
    
    // 2. Try to add at CURRENT location
    let success = geofenceManager.addGeofenceAtCurrentLocation(
        name: "My Location",
        enterMessage: "Welcome back! You have arrived at your saved location.",
        exitMessage: "You have left your saved location."
    )
    
    // 3. Handle Feedback
    if success, let location = geofenceManager.geofences.last {
         let lat = String(format: "%.4f", location.latitude)
         let lon = String(format: "%.4f", location.longitude)
         let message = "Updated Location: \(lat), \(lon)"
         
         viewModel.ttsManager.speak("Location updated")
         viewModel.lastAIResponse = message
    } else {
        // Fallback
        geofenceManager.addTestGeofence(
          name: "Test Location",
          latitude: 42.3601,
          longitude: -71.0942,
          enterMessage: "You have arrived at the test location.",
          exitMessage: "You have left the test location."
        )
        let message = "GPS not ready. Using Boston (Test Mode)"
        viewModel.ttsManager.speak("GPS not ready, using default location")
        viewModel.lastAIResponse = message
    }
    
    // 4. Auto-hide message after 3 seconds
    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
        if self.viewModel.lastAIResponse.contains("Location") || self.viewModel.lastAIResponse.contains("GPS") {
            self.viewModel.lastAIResponse = ""
        }
    }
  }
}

// Extracted controls for clarity
struct ControlsView: View {
  @ObservedObject var viewModel: StreamSessionViewModel
  var body: some View {
    // Controls row
    HStack(spacing: 8) {
      CustomButton(
        title: "Stop streaming",
        style: .destructive,
        isDisabled: false
      ) {
        Task {
          await viewModel.stopSession()
        }
      }

      // Timer button
      CircleButton(
        icon: "timer",
        text: viewModel.activeTimeLimit != .noLimit ? viewModel.activeTimeLimit.displayText : nil
      ) {
        let nextTimeLimit = viewModel.activeTimeLimit.next
        viewModel.setTimeLimit(nextTimeLimit)
      }

      // Photo button
      CircleButton(icon: "camera.fill", text: nil) {
        viewModel.capturePhoto()
      }
    }
  }
}
