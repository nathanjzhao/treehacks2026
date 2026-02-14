/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// TTSManager.swift
//
// Text-to-Speech manager for Ray-Ban glasses audio output.
// Handles voice prompts, reminders, and responses.
//

import AVFoundation
import Foundation

@MainActor
class TTSManager: NSObject, ObservableObject, AVSpeechSynthesizerDelegate, AVAudioPlayerDelegate {
  private let synthesizer = AVSpeechSynthesizer()
  private var audioPlayer: AVAudioPlayer?
  @Published var isSpeaking: Bool = false
  
  // Callback when speech/audio finishes playing naturally
  var onSpeechEnded: (() -> Void)?
  
  override init() {
    super.init()
    synthesizer.delegate = self
    
    // Configure Audio Session for playback mixed with other audio (like the camera stream)
    do {
      let audioSession = AVAudioSession.sharedInstance()
      try audioSession.setCategory(.playAndRecord, options: [.mixWithOthers, .allowBluetooth, .defaultToSpeaker])
      try audioSession.setActive(true)
      NSLog("[TTSManager] Audio session configured with mixWithOthers")
    } catch {
      NSLog("[TTSManager] Failed to configure audio session: \(error)")
    }
  }
  
  /// Speak text through Ray-Ban speakers
  /// - Parameters:
  ///   - text: The text to speak
  ///   - rate: Speech rate (0.0-1.0, default 0.5)
  ///   - pitch: Pitch (0.5-2.0, default 1.0)
  ///   - volume: Volume (0.0-1.0, default 1.0)
  func speak(_ text: String, rate: Float = 0.5, pitch: Float = 1.0, volume: Float = 1.0) {
    NSLog("[TTSManager] Speaking: \(text)")
    
    let utterance = AVSpeechUtterance(string: text)
    utterance.rate = rate
    utterance.pitchMultiplier = pitch
    utterance.volume = volume
    utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
    
    synthesizer.speak(utterance)
  }
  
  /// Play audio from Base64 string
  /// - Parameter base64String: The Base64 encoded audio data
  func playAudioData(_ base64String: String) {
    NSLog("[TTSManager] 🎵 playAudioData called with \(base64String.count) chars")
    
    guard let data = Data(base64Encoded: base64String) else {
      NSLog("[TTSManager] ❌ Failed to decode Base64 audio string")
      return
    }
    
    NSLog("[TTSManager] ✅ Decoded \(data.count) bytes of audio data")
    
    // Stop any current playback
    stop()
    
    // Reconfigure audio session to ensure clean state
    do {
      let audioSession = AVAudioSession.sharedInstance()
      try audioSession.setCategory(.playAndRecord, options: [.mixWithOthers, .allowBluetooth, .defaultToSpeaker])
      try audioSession.setActive(true)
      NSLog("[TTSManager] ✅ Audio session reconfigured")
    } catch {
      NSLog("[TTSManager] ⚠️ Audio session reconfigure warning: \(error)")
    }
    
    do {
      audioPlayer = try AVAudioPlayer(data: data)
      audioPlayer?.delegate = self
      audioPlayer?.volume = 1.0
      audioPlayer?.prepareToPlay()
      
      let success = audioPlayer?.play() ?? false
      isSpeaking = success
      NSLog("[TTSManager] ▶️ Playing audio data: \(data.count) bytes, success: \(success)")
      
      if !success {
        NSLog("[TTSManager] ❌ play() returned false!")
      }
    } catch {
      NSLog("[TTSManager] ❌ Failed to play audio data: \(error)")
      // Trigger callback so the app doesn't stay stuck
      onSpeechEnded?()
    }
  }
  
  /// Stop current speech or audio
  func stop() {
    NSLog("[TTSManager] 🛑 stop() called")
    if synthesizer.isSpeaking {
        NSLog("[TTSManager] 🛑 Stopping synthesizer...")
        synthesizer.stopSpeaking(at: .immediate)
    }
    if let player = audioPlayer, player.isPlaying {
        NSLog("[TTSManager] 🛑 Stopping audio player...")
        player.stop()
    }
    isSpeaking = false
  }
  
  // MARK: - AVSpeechSynthesizerDelegate
  
  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
    isSpeaking = true
    NSLog("[TTSManager] Started speaking")
  }
  
  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
    isSpeaking = false
    NSLog("[TTSManager] Finished speaking")
    onSpeechEnded?()
  }
  
  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
    isSpeaking = false
    NSLog("[TTSManager] Cancelled speaking")
  }
  
  // MARK: - AVAudioPlayerDelegate
  
  func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
    isSpeaking = false
    NSLog("[TTSManager] Checkpoint: Finished playing audio (success: \(flag))")
    onSpeechEnded?()
  }
}
