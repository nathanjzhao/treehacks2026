/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// Geofence.swift
//
// Data model for geofenced locations with enter/exit messages.
// Used for location-based reminders and notifications.
//

import Foundation
import CoreLocation

struct Geofence: Identifiable, Codable, Equatable {
  let id: UUID
  let name: String
  let latitude: Double
  let longitude: Double
  let radius: Double  // in meters (minimum ~100m for reliable detection)
  
  // Messages to trigger
  let enterMessage: String?  // Audio message when user ENTERS this location
  let exitMessage: String?   // Message when user LEAVES this location
  
  // Convenience computed property
  var coordinate: CLLocationCoordinate2D {
    CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
  }
  
  var region: CLCircularRegion {
    let region = CLCircularRegion(
      center: coordinate,
      radius: radius,
      identifier: id.uuidString
    )
    region.notifyOnEntry = enterMessage != nil
    region.notifyOnExit = exitMessage != nil
    return region
  }
  
  init(
    id: UUID = UUID(),
    name: String,
    latitude: Double,
    longitude: Double,
    radius: Double = 100,  // Default 100 meters
    enterMessage: String? = nil,
    exitMessage: String? = nil
  ) {
    self.id = id
    self.name = name
    self.latitude = latitude
    self.longitude = longitude
    self.radius = max(radius, 100)  // Enforce minimum for reliability
    self.enterMessage = enterMessage
    self.exitMessage = exitMessage
  }
  
  // Create from CLLocation (convenience)
  init(
    name: String,
    location: CLLocation,
    radius: Double = 100,
    enterMessage: String? = nil,
    exitMessage: String? = nil
  ) {
    self.init(
      name: name,
      latitude: location.coordinate.latitude,
      longitude: location.coordinate.longitude,
      radius: radius,
      enterMessage: enterMessage,
      exitMessage: exitMessage
    )
  }
}

// MARK: - Persistence Helper
extension Geofence {
  static let storageKey = "savedGeofences"
  
  static func loadAll() -> [Geofence] {
    guard let data = UserDefaults.standard.data(forKey: storageKey),
          let geofences = try? JSONDecoder().decode([Geofence].self, from: data) else {
      return []
    }
    return geofences
  }
  
  static func saveAll(_ geofences: [Geofence]) {
    if let data = try? JSONEncoder().encode(geofences) {
      UserDefaults.standard.set(data, forKey: storageKey)
    }
  }
}
