# 🚀 Quick Start - Ray-Ban + iOS + Backend Setup

## ✅ What's Already Done

1. ✅ **Backend Integration Code** - Already in the project
   - `MemoryCaptureWebSocketClient.swift` - Sends captures to backend
   - `GCPUploader.swift` - Uploads photos to backend
   - `MemoryCaptureManager.swift` - **NEW** - Integrates everything

2. ✅ **Photo Capture Integration** - **JUST ADDED**
   - Photos automatically upload to backend when captured
   - WebSocket sends memory captures automatically
   - No manual steps needed!

---

## 📋 Step-by-Step Setup

### Step 1: Open Project in Xcode

```bash
cd "/Users/sheilawang/Desktop/RealityHacks26/samples/CameraAccess"
open CameraAccess.xcodeproj
```

---

### Step 2: Configure Xcode Project

1. **Select Project** (top of left sidebar)
2. **Select "CameraAccess" target**
3. **Go to "Signing & Capabilities" tab**
4. **Enable "Automatically manage signing"**
5. **Select your Team**:
   - If you see "TX49M9562D" (Meta's team), you can use it for testing
   - OR add your own Apple ID team
   - Free tier works for testing on your own device

6. **Bundle Identifier** (if needed):
   - Current: `com.meta.wearables.external.CameraAccess-example`
   - You can keep this OR change to something unique

7. **Deployment Target**:
   - Set to **iOS 17.0+** (required for Meta Wearables DAT SDK)

---

### Step 3: Configure Meta Credentials (Optional for Testing)

**For Testing Without Real Glasses:**
- You can use **Mock Device Kit** (built into the app)
- No Meta credentials needed for Mock Device

**For Real Glasses:**
1. **Get Meta Developer Credentials**:
   - Go to: https://developers.facebook.com/
   - Create app → Get **App ID** and **Client Token

2. **Set in Xcode**:
   - Project → Build Settings → Search "CLIENT_TOKEN"
   - Add User-Defined Setting: `CLIENT_TOKEN` = your token
   - Or edit `Info.plist` → `MetaAppID` (currently `0`)

---

### Step 4: Connect iPhone

1. **Connect iPhone** to Mac via USB
2. **Unlock iPhone** and trust computer
3. **In Xcode**: Select your iPhone from device dropdown (top toolbar)

---

### Step 5: Enable Developer Mode

**On iPhone:**
1. **Settings** → **Privacy & Security** → **Developer Mode**
2. **Toggle ON** → Restart if prompted

**In Meta AI App (if using real glasses):**
1. **Install Meta AI app** from App Store
2. **Open Meta AI** → **Settings** → **Developer Mode** → **Toggle ON**
3. **Pair your Ray-Ban glasses** with Meta AI app

---

### Step 6: Configure User ID (Optional)

**Current Default:** `"cass"`

**To Change:**
1. Open: `CameraAccess/ViewModels/StreamSessionViewModel.swift`
2. Find line ~168: `MemoryCaptureManager(userId: "cass")`
3. Change to your user ID: `MemoryCaptureManager(userId: "your_user_id")`

**OR** update in:
- `GCPUploader.swift` line 28: `var userId: String = "cass"`
- `MemoryCaptureWebSocketClient.swift` line 24: `var userId: String = ""`

---

### Step 7: Build and Run

1. **Press `Cmd + B`** to build
2. **Fix any errors** if they appear
3. **Press `Cmd + R`** to run on iPhone
4. **First time**: Trust developer certificate on iPhone
   - Settings → General → VPN & Device Management → Trust

---

### Step 8: Test the App

#### Test Without Glasses (Mock Device):

1. **Launch app** on iPhone
2. **Look for debug menu** (usually a button or gesture)
3. **Enable Mock Device**
4. **Start streaming** → You'll see simulated video
5. **Capture photo** → Photo uploads automatically to backend!

#### Test With Real Glasses:

1. **Launch app** on iPhone
2. **Press "Connect"** → Redirects to Meta AI app
3. **Authorize** connection in Meta AI app
4. **Returns to your app** → Grant permissions
5. **Start streaming** → Live video from glasses!
6. **Capture photo** → Automatically uploads to backend!

---

## 🎯 What Happens When You Capture a Photo

```
1. You press camera button
   ↓
2. Photo captured from glasses
   ↓
3. Photo preview shows on screen
   ↓
4. MemoryCaptureManager automatically:
   - Uploads photo to backend (POST /upload/{captureId})
   - Sends memory capture via WebSocket (/ws/ios/{userId})
   - Backend processes with Gemini AI
   - Updates contacts database
   ↓
5. You see "Upload complete!" status
```

**All automatic!** No manual steps needed.

---

## 🧪 Testing Backend Connection

### Test WebSocket Connection:

```bash
cd "/Users/sheilawang/Desktop/RealityHacks26/samples/CameraAccess"
swift test_backend.swift
```

This tests:
- WebSocket connection to backend
- Memory capture sending
- Query WebSocket

### Check Backend Logs:

- Backend URL: `https://memory-backend-328251955578.us-east1.run.app`
- Check your backend logs to see incoming captures

---

## ✅ Success Checklist

- [ ] Xcode project opens without errors
- [ ] Signing configured (Team selected)
- [ ] iPhone connected and selected in Xcode
- [ ] Developer Mode enabled on iPhone
- [ ] App builds successfully (`Cmd + B`)
- [ ] App installs on iPhone (`Cmd + R`)
- [ ] App launches on iPhone
- [ ] Can connect to glasses (or Mock Device works)
- [ ] Video streams from glasses
- [ ] Can capture photos
- [ ] Photos upload to backend (check status message)
- [ ] Backend receives captures (check backend logs)

---

## 🐛 Common Issues

### "Device Not Found"
- ✅ Use Mock Device for testing (no glasses needed)
- ✅ Check glasses are paired with Meta AI app
- ✅ Check Bluetooth is enabled

### "Build Failed"
- ✅ Check Xcode version (14.0+)
- ✅ Clean build: `Cmd + Shift + K`, then rebuild
- ✅ Check all files are added to project

### "Permission Denied"
- ✅ iPhone Settings → Your App → Enable Camera & Microphone

### "Backend Connection Failed"
- ✅ Check network connection
- ✅ Verify backend URL is correct
- ✅ Test with: `swift test_backend.swift`

### "MemoryCaptureManager not found"
- ✅ Make sure `MemoryCaptureManager.swift` is added to Xcode project
- ✅ Check file is in "Utils" folder
- ✅ Clean and rebuild

---

## 📱 Next Steps After Setup

1. **Test basic flow**: Capture → Upload → Backend
2. **Check backend**: Verify captures are received
3. **Test people detection**: Backend should detect faces
4. **View in Caretaker App**: Check people profiles update

---

## 🎉 You're Ready!

Once you can:
- ✅ Connect to glasses (or Mock Device)
- ✅ Stream video
- ✅ Capture photos
- ✅ See upload status

**Everything is working!** Photos automatically go to backend and get processed.

---

## 📚 More Info

- **Full Setup Guide**: See `COMPLETE_SETUP_GUIDE.md`
- **Testing Guide**: See `TESTING_WITHOUT_XCODE.md`
- **Development Guide**: See `RAYBAN_DEVELOPMENT_GUIDE.md`

---

**Questions?** Check the troubleshooting section or the full guides above! 🚀
