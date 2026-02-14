/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// GeofenceManager.swift
//
// Manages geofenced locations and triggers audio/message events.
//

import Foundation
import CoreLocation
import SwiftUI

@MainActor
class GeofenceManager: ObservableObject {
  private let locationManager: LocationManager
  
  @Published var geofences: [Geofence] = []
  @Published var lastEvent: GeofenceEvent?
  @Published var showExitAlert: Bool = false
  @Published var exitAlertMessage: String = ""
  
  // Callbacks
  var onEnterGeofence: ((Geofence) -> Void)?
  var onExitGeofence: ((Geofence) -> Void)?
  
  init(locationManager: LocationManager) {
    self.locationManager = locationManager
    self.geofences = Geofence.loadAll()
    
    locationManager.onEnterRegion = { [weak self] regionId in
      Task { @MainActor in
        self?.handleEnterRegion(regionId)
      }
    }
    
    locationManager.onExitRegion = { [weak self] regionId in
      Task { @MainActor in
        self?.handleExitRegion(regionId)
      }
    }
    
    NSLog("[GeofenceManager] Ready - \(geofences.count) geofences loaded")
  }
  
  // MARK: - Permission Handling
  
  func requestPermissions() {
    if !locationManager.hasLocationPermission {
      locationManager.requestWhenInUsePermission()
    } else if !locationManager.hasAlwaysPermission {
      locationManager.requestAlwaysPermission()
    }
    // Try to warm up GPS
    locationManager.requestSingleLocation()
  }
  
  var hasRequiredPermissions: Bool {
    locationManager.hasAlwaysPermission
  }
  
  var currentLocation: CLLocation? {
    locationManager.currentLocation
  }
  
  // MARK: - Geofence Management
  
  func addGeofence(_ geofence: Geofence) {
    guard geofences.count < 20 else {
      NSLog("[GeofenceManager] Cannot add - max 20 geofences")
      return
    }
    
    geofences.append(geofence)
    Geofence.saveAll(geofences)
    locationManager.startMonitoring(region: geofence.region)
    NSLog("[GeofenceManager] Added geofence: \(geofence.name)")
  }
  
  func removeGeofence(_ geofence: Geofence) {
    locationManager.stopMonitoring(region: geofence.region)
    geofences.removeAll { $0.id == geofence.id }
    Geofence.saveAll(geofences)
    NSLog("[GeofenceManager] Removed geofence: \(geofence.name)")
  }
  
  func removeAll() {
    locationManager.stopMonitoringAll()
    geofences.removeAll()
    Geofence.saveAll(geofences)
    NSLog("[GeofenceManager] Removed all geofences")
  }
  
  func startMonitoringAll() {
    for geofence in geofences {
      locationManager.startMonitoring(region: geofence.region)
    }
    NSLog("[GeofenceManager] Started monitoring \(geofences.count) geofences")
  }
  
  // MARK: - Event Handlers
  
  private func handleEnterRegion(_ regionId: String) {
    guard let geofence = geofences.first(where: { $0.id.uuidString == regionId }) else {
      return
    }
    
    NSLog("[GeofenceManager] 📍 ENTERED: \(geofence.name)")
    lastEvent = GeofenceEvent(type: .enter, geofence: geofence, timestamp: Date())
    onEnterGeofence?(geofence)
  }
  
  private func handleExitRegion(_ regionId: String) {
    guard let geofence = geofences.first(where: { $0.id.uuidString == regionId }) else {
      return
    }
    
    NSLog("[GeofenceManager] 📍 EXITED: \(geofence.name)")
    lastEvent = GeofenceEvent(type: .exit, geofence: geofence, timestamp: Date())
    
    if let exitMessage = geofence.exitMessage {
      exitAlertMessage = exitMessage
      showExitAlert = true
    }
    
    onExitGeofence?(geofence)
  }
  
  // MARK: - Test Helpers
  
  func addTestGeofence(name: String, latitude: Double, longitude: Double, enterMessage: String?, exitMessage: String?) {
    let geofence = Geofence(
      name: name,
      latitude: latitude,
      longitude: longitude,
      radius: 100,
      enterMessage: enterMessage,
      exitMessage: exitMessage
    )
    addGeofence(geofence)
  }
  
  // Create a geofence at the user's current physical location
  func addGeofenceAtCurrentLocation(name: String, enterMessage: String?, exitMessage: String?) -> Bool {
    guard let location = locationManager.currentLocation else {
      NSLog("[GeofenceManager] Cannot add - no current location")
      // Force an update for next time
      locationManager.requestSingleLocation() 
      return false
    }
    
    let geofence = Geofence(
      name: name,
      latitude: location.coordinate.latitude,
      longitude: location.coordinate.longitude,
      radius: 20,  // Reduced to 20m for small space/hackathon testing
      enterMessage: enterMessage,
      exitMessage: exitMessage
    )
    addGeofence(geofence)
    return true
  }
}

// MARK: - Event Model
struct GeofenceEvent: Equatable {
  enum EventType { case enter, exit }
  let type: EventType
  let geofence: Geofence
  let timestamp: Date
}
