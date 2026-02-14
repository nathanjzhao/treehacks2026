# Ray-Ban Meta Glasses Development Guide

## 📱 Existing Apps

### Meta AI App (Official)
- **What it is**: Meta's official app for Ray-Ban Meta glasses
- **Available**: App Store (iOS) / Google Play (Android)
- **Features**: 
  - Basic camera controls
  - Voice commands ("Hey Meta")
  - Photo/video capture
  - Settings management
- **Developer Mode**: Can be enabled for third-party app development

### Your Custom App (This Project)
- **What it is**: Custom iOS app using Meta Wearables DAT SDK
- **Status**: Sample app exists in `samples/CameraAccess/`
- **Features**: 
  - Video streaming from glasses
  - Photo capture
  - Backend integration for memory assistance
  - Custom features (PRD requirements)

---

## 🚀 Development Process: From Code to Your Phone

### Step 1: Prerequisites

#### Hardware Requirements:
- ✅ **Mac with macOS** (required for Xcode)
- ✅ **iPhone** (iOS 17.0+)
- ✅ **Ray-Ban Meta Glasses** (for testing)
- ✅ **USB Cable** (to connect iPhone to Mac)

#### Software Requirements:
- ✅ **Xcode** (free from Mac App Store)
  - Download: https://apps.apple.com/us/app/xcode/id497799835
  - Version: 14.0+ (latest recommended)
- ✅ **Apple Developer Account** (free for basic testing)
  - Sign up: https://developer.apple.com
  - Free tier allows testing on your own device
- ✅ **Meta AI App** (on your iPhone)
  - Download from App Store
  - Enable Developer Mode (see Step 2)

---

### Step 2: Enable Developer Mode

1. **Install Meta AI App** on your iPhone
   - Download from App Store
   - Sign in with your Meta account

2. **Enable Developer Mode**:
   - Open Meta AI app
   - Go to **Settings** → **Developer Mode**
   - Toggle **Developer Mode** ON
   - This allows third-party apps to connect to your glasses

3. **Pair Your Glasses**:
   - Make sure glasses are paired with Meta AI app
   - Glasses should be connected via Bluetooth

---

### Step 3: Set Up Xcode Project

1. **Open Project in Xcode**:
   ```bash
   cd /Users/sheilawang/Desktop/RealityHacks26/samples/CameraAccess
   open CameraAccess.xcodeproj
   ```

2. **Configure Signing**:
   - In Xcode, select the project
   - Go to **Signing & Capabilities** tab
   - Select your **Team** (your Apple ID)
   - Xcode will automatically create a provisioning profile

3. **Select Your iPhone**:
   - Connect iPhone to Mac via USB
   - In Xcode, select your iPhone from device dropdown (top toolbar)
   - Trust the computer on iPhone if prompted

---

### Step 4: Build and Install on iPhone

1. **Build the App**:
   - Press `Cmd + B` to build
   - Fix any errors if they appear

2. **Run on iPhone**:
   - Press `Cmd + R` to run
   - First time: iPhone will ask to trust developer
   - Go to **Settings → General → VPN & Device Management**
   - Trust your developer certificate

3. **App Installed**:
   - App appears on your iPhone home screen
   - You can now use it!

---

### Step 5: Connect to Ray-Ban Glasses

1. **Launch Your App** on iPhone

2. **Press "Connect" Button**:
   - App will open Meta AI app automatically
   - Meta AI app handles OAuth registration
   - You'll be redirected back to your app

3. **Grant Permissions**:
   - Camera permission (for streaming)
   - Microphone permission (for audio)
   - Bluetooth permission (for glasses connection)

4. **Start Streaming**:
   - Once connected, you'll see live video from glasses
   - Use controls to capture photos, set timers, etc.

---

## 🛠️ Development Workflow

### Daily Development Cycle:

```
1. Edit code in Xcode (or Cursor)
   ↓
2. Build: Cmd + B
   ↓
3. Run on iPhone: Cmd + R
   ↓
4. Test with Ray-Ban glasses
   ↓
5. Debug and iterate
```

### Testing Without Glasses (Development):

The project includes **Mock Device Kit** for testing without physical glasses:
- Simulates glasses connection
- Mock video streaming
- Test all features except hardware-specific ones

---

## 📝 Adding Features (Example: "Set Time to Send Audio")

### Current State:
- ✅ Video streaming works
- ✅ Photo capture works
- ❌ Scheduled audio sending - **NOT IMPLEMENTED**

### How to Add "Set Time to Send Audio":

#### 1. Create a ViewModel for Audio Settings:

```swift
// AudioSettingsViewModel.swift
@MainActor
class AudioSettingsViewModel: ObservableObject {
    @Published var scheduledTime: Date?
    @Published var isScheduled: Bool = false
    
    func scheduleAudio(time: Date) {
        scheduledTime = time
        isScheduled = true
        // Schedule notification/timer
    }
}
```

#### 2. Add UI Controls:

```swift
// In StreamView.swift or new SettingsView.swift
DatePicker("Schedule Audio", selection: $audioSettings.scheduledTime)
Button("Set Schedule") {
    audioSettings.scheduleAudio(time: selectedTime)
}
```

#### 3. Implement Audio Capture:

```swift
// In StreamSessionViewModel.swift
func startAudioRecording() {
    // Use AVAudioRecorder or DAT SDK audio APIs
}

func sendAudioToBackend(audioData: Data) async {
    // Upload to backend
    // Send via WebSocket
}
```

#### 4. Schedule Timer:

```swift
// Use Timer or async/await
Task {
    let delay = scheduledTime.timeIntervalSinceNow
    try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
    // Trigger audio capture and send
}
```

---

## 🔧 Available APIs (Meta Wearables DAT SDK)

### What You Can Do:

1. **Video Streaming**:
   ```swift
   streamSession.start()  // Start video stream
   streamSession.capturePhoto()  // Capture photo
   ```

2. **Device Management**:
   ```swift
   wearables.startRegistration()  // Connect to glasses
   wearables.devices  // Get connected devices
   ```

3. **Permissions**:
   ```swift
   wearables.requestPermission(.camera)
   wearables.requestPermission(.microphone)
   ```

4. **Audio** (if available in SDK):
   - Audio streaming
   - Audio capture
   - Voice commands

---

## 📱 Using the App on Your Phone

### First Time Setup:

1. **Install App** (via Xcode or TestFlight)
2. **Enable Developer Mode** in Meta AI app
3. **Launch Your App**
4. **Press "Connect"** → Redirects to Meta AI app
5. **Authorize** → Returns to your app
6. **Grant Permissions** (Camera, Microphone)
7. **Start Using!**

### Daily Use:

1. Open your app
2. Glasses should auto-connect (if paired)
3. Start streaming/capturing
4. Use your custom features

---

## 🐛 Troubleshooting

### "Device Not Found"
- Make sure glasses are paired with Meta AI app
- Check Bluetooth is enabled
- Restart both apps

### "Registration Failed"
- Check Developer Mode is ON in Meta AI app
- Make sure you're signed in to Meta AI app
- Try disconnecting and reconnecting

### "Permission Denied"
- Go to iPhone Settings → Your App
- Enable Camera and Microphone permissions

### "Build Failed"
- Check Xcode version (14.0+)
- Update Meta Wearables DAT SDK if needed
- Clean build folder: `Cmd + Shift + K`, then rebuild

---

## 📚 Resources

- **Meta Wearables DAT SDK Docs**: https://wearables.developer.meta.com/docs/develop/
- **Apple Developer Docs**: https://developer.apple.com/documentation/
- **Xcode Download**: https://apps.apple.com/us/app/xcode/id497799835

---

## 🎯 Next Steps

1. **Install Xcode** (if not already)
2. **Enable Developer Mode** in Meta AI app
3. **Open project** in Xcode
4. **Build and run** on your iPhone
5. **Test with glasses**!

Want help implementing a specific feature? Let me know!
