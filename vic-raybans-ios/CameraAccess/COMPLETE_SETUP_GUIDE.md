# Complete Setup Guide - Ray-Ban + iOS App + Backend

## 🎯 Overview

This guide will help you:
1. ✅ Set up Xcode project
2. ✅ Configure Meta Wearables DAT SDK (for Ray-Ban)
3. ✅ Connect iOS app to backend
4. ✅ Test Ray-Ban integration
5. ✅ Test full flow (Glasses → iOS → Backend)

---

## Step 1: Open Project in Xcode

```bash
cd "/Users/sheilawang/Desktop/RealityHacks26/samples/CameraAccess"
open CameraAccess.xcodeproj
```

---

## Step 2: Configure Xcode Project Settings

### 2.1 Signing & Capabilities

1. **Select Project** in Xcode (top of left sidebar)
2. **Select "CameraAccess" target**
3. **Go to "Signing & Capabilities" tab**
4. **Enable "Automatically manage signing"**
5. **Select your Team** (your Apple ID)
   - If you don't have a team, click "Add Account"
   - Sign in with your Apple ID
   - Free tier works for testing on your own device

### 2.2 Bundle Identifier

1. Still in **Signing & Capabilities**
2. **Bundle Identifier**: Change to something unique
   - Example: `com.yourname.realityhacks` or `com.yourname.cameraaccess`
   - Must be unique (not used by other apps)

### 2.3 Deployment Target

1. **General tab** → **Minimum Deployments**
2. Set **iOS** to **17.0** (required for Meta Wearables DAT SDK)

---

## Step 3: Configure Meta Wearables DAT SDK

### 3.1 Get Meta Developer Credentials

You need to register your app with Meta to get:
- **Client Token**
- **Meta App ID**

**Option A: Use Meta Developer Portal** (Recommended for production)
1. Go to: https://developers.facebook.com/
2. Create a Meta Developer account
3. Create a new app
4. Get your **App ID** and **Client Token**

**Option B: Use Default/Test Values** (For development/testing)
- For testing, you can use placeholder values initially
- The SDK may work with default values for development

### 3.2 Set Environment Variables in Xcode

1. **Select Project** → **CameraAccess target** → **Build Settings**
2. Search for: `CLIENT_TOKEN`
3. Add User-Defined Setting:
   - **Key**: `CLIENT_TOKEN`
   - **Value**: Your Meta Client Token (or placeholder for testing)

4. **Team ID** is automatically set from your Apple Developer Team

### 3.3 Update Info.plist (if needed)

The `Info.plist` already has the MWDAT configuration:
```xml
<key>MWDAT</key>
<dict>
    <key>AppLinkURLScheme</key>
    <string>cameraaccess://</string>
    <key>ClientToken</key>
    <string>$(CLIENT_TOKEN)</string>
    <key>MetaAppID</key>
    <string>0</string>  <!-- Update this with your App ID -->
    <key>TeamID</key>
    <string>$(DEVELOPMENT_TEAM)</string>
</dict>
```

**To update MetaAppID:**
1. Open `Info.plist` in Xcode
2. Find `MetaAppID` (currently `0`)
3. Change to your Meta App ID (if you have one)
4. For testing, `0` might work with Mock Device Kit

---

## Step 4: Connect iPhone to Mac

1. **Connect iPhone** to Mac via USB cable
2. **Unlock iPhone** and trust the computer if prompted
3. **In Xcode**: Select your iPhone from device dropdown (top toolbar)
   - Should show: "iPhone (iOS XX.X)"

---

## Step 5: Enable Developer Mode on iPhone

1. **On iPhone**: Go to **Settings** → **Privacy & Security**
2. Scroll down to **Developer Mode**
3. **Toggle ON** Developer Mode
4. **Restart iPhone** if prompted

---

## Step 6: Enable Meta AI Developer Mode

1. **Install Meta AI App** on iPhone (from App Store)
2. **Open Meta AI app**
3. **Sign in** with your Meta account
4. **Go to Settings** → **Developer Mode**
5. **Toggle Developer Mode ON**
6. **Pair your Ray-Ban glasses** with Meta AI app (if you have them)

---

## Step 7: Configure Backend Connection

The app already has backend integration code. You need to set the user ID:

### Option A: Hardcode User ID (Quick Test)

Edit `MemoryCaptureWebSocketClient.swift`:
```swift
var userId: String = "your_user_id_here"  // Line 24
```

### Option B: Add Settings UI (Better)

Create a settings screen to let user enter their ID.

**For now, let's update the default:**

1. Open: `samples/CameraAccess/CameraAccess/Utils/MemoryCaptureWebSocketClient.swift`
2. Find line 24: `var userId: String = ""`
3. Change to: `var userId: String = "test_user"` (or your actual user ID)

---

## Step 8: Build and Run on iPhone

1. **In Xcode**: Press `Cmd + B` to build
2. **Fix any errors** if they appear
3. **Press `Cmd + R`** to run on iPhone
4. **First time**: iPhone will ask to trust developer
   - Go to: **Settings** → **General** → **VPN & Device Management**
   - Trust your developer certificate
5. **App launches** on iPhone!

---

## Step 9: Connect to Ray-Ban Glasses

### 9.1 Using Real Glasses

1. **Launch your app** on iPhone
2. **Press "Connect" button**
3. **App redirects to Meta AI app**
4. **Authorize** the connection in Meta AI app
5. **Returns to your app**
6. **Grant permissions**:
   - Camera permission
   - Microphone permission
   - Bluetooth permission
7. **Start streaming** - you should see live video from glasses!

### 9.2 Using Mock Device (Testing Without Glasses)

The app includes **Mock Device Kit** for testing:

1. **Launch app** in DEBUG mode
2. **Look for debug menu** (usually a button or gesture)
3. **Enable Mock Device**
4. **Simulate glasses connection**
5. **Test all features** except hardware-specific ones

---

## Step 10: Test Backend Integration

### 10.1 Test Memory Capture

1. **Start streaming** from glasses
2. **Capture a photo** (camera button)
3. **App should**:
   - Upload photo to backend via `POST /upload/{captureId}`
   - Send memory capture via WebSocket `/ws/ios/{userId}`
   - Receive acknowledgment

### 10.2 Verify Backend Connection

Check backend logs or use the test script:
```bash
cd "/Users/sheilawang/Desktop/RealityHacks26/samples/CameraAccess"
swift test_backend.swift
```

---

## Step 11: Full Integration Test

### Test Flow:

```
1. Ray-Ban Glasses → Capture Photo
   ↓
2. iOS App → Upload to Backend
   ↓
3. Backend → Process with Gemini AI
   ↓
4. Backend → Update Contacts Database
   ↓
5. Caretaker App → View People Profiles
```

### Steps:

1. **Capture photo** from glasses in iOS app
2. **Check backend** - photo should be uploaded
3. **Wait for processing** - backend analyzes with Gemini
4. **Check contacts** - new people should appear
5. **Open Caretaker App** - should see updated people profiles

---

## 🔧 Configuration Checklist

- [ ] Xcode project opened
- [ ] Signing & Capabilities configured (Team selected)
- [ ] Bundle Identifier set (unique)
- [ ] iOS deployment target: 17.0+
- [ ] CLIENT_TOKEN set (or placeholder)
- [ ] MetaAppID set (or 0 for testing)
- [ ] iPhone connected to Mac
- [ ] Developer Mode enabled on iPhone
- [ ] Meta AI app installed on iPhone
- [ ] Developer Mode enabled in Meta AI app
- [ ] Ray-Ban glasses paired (or Mock Device enabled)
- [ ] Backend user ID configured
- [ ] App built and installed on iPhone
- [ ] Permissions granted (Camera, Microphone, Bluetooth)

---

## 🧪 Testing Scenarios

### Scenario 1: Test Without Glasses (Mock Device)

1. Use Mock Device Kit in DEBUG mode
2. Simulate photo capture
3. Test backend upload
4. Verify WebSocket connection

### Scenario 2: Test With Real Glasses

1. Connect real Ray-Ban glasses
2. Stream live video
3. Capture actual photos
4. Test full backend integration

### Scenario 3: Test Backend Only

1. Use test script: `swift test_backend.swift`
2. Test WebSocket connections
3. Verify API endpoints
4. Check data flow

---

## 🐛 Troubleshooting

### "Device Not Found"
- ✅ Check glasses are paired with Meta AI app
- ✅ Check Bluetooth is enabled
- ✅ Restart both apps
- ✅ Try Mock Device for testing

### "Registration Failed"
- ✅ Check Developer Mode is ON in Meta AI app
- ✅ Check you're signed in to Meta AI app
- ✅ Verify CLIENT_TOKEN is set (even if placeholder)
- ✅ Try disconnecting and reconnecting

### "Permission Denied"
- ✅ Go to iPhone Settings → Your App
- ✅ Enable Camera and Microphone permissions

### "Backend Connection Failed"
- ✅ Check backend URL is correct
- ✅ Verify user ID is set
- ✅ Check network connection
- ✅ Test backend with: `swift test_backend.swift`

### "Build Failed"
- ✅ Check Xcode version (14.0+)
- ✅ Update Meta Wearables DAT SDK if needed
- ✅ Clean build: `Cmd + Shift + K`, then rebuild

---

## 📱 Quick Test Commands

```bash
# Test backend connection
cd "/Users/sheilawang/Desktop/RealityHacks26/samples/CameraAccess"
swift test_backend.swift

# Check if backend is running
curl https://memory-backend-328251955578.us-east1.run.app/

# Test WebSocket (requires Node.js)
# Or use the Swift test script above
```

---

## 🎯 Next Steps After Setup

1. **Test basic connection**: Glasses → iOS app
2. **Test photo capture**: Capture → Upload → Backend
3. **Test memory capture**: WebSocket → Backend processing
4. **Test people detection**: Backend → Contacts update
5. **Test caretaker app**: View people profiles

---

## 📚 Resources

- **Meta Wearables DAT SDK Docs**: https://wearables.developer.meta.com/docs/develop/
- **Meta Developer Portal**: https://developers.facebook.com/
- **Apple Developer Docs**: https://developer.apple.com/documentation/
- **Backend API Docs**: See `Backend/send_data.md` and `Backend/query_data.md`

---

## ✅ Success Indicators

You'll know everything is working when:

1. ✅ App builds without errors
2. ✅ App installs on iPhone
3. ✅ "Connect" button works
4. ✅ Glasses connect (or Mock Device works)
5. ✅ Video streams from glasses
6. ✅ Photos can be captured
7. ✅ Backend receives uploads
8. ✅ WebSocket sends memory captures
9. ✅ Backend processes with Gemini
10. ✅ People profiles update in backend

---

Ready to test! Let me know if you hit any issues during setup. 🚀
