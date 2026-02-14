# Testing Without Xcode - Workarounds Guide

Since you don't have Xcode installed, here are several workarounds to test and understand the iOS app functionality:

## ✅ What You CAN Do

### 1. **View & Edit Code in Cursor**
- Open any `.swift` file in Cursor
- See all the code structure
- Edit with AI assistance
- Navigate between files easily

### 2. **Test Backend Connections (Command Line)**
Run the test script to test WebSocket connections:

```bash
cd /Users/sheilawang/Desktop/RealityHacks26/samples/CameraAccess
swift test_backend.swift
```

This will:
- Test memory capture WebSocket (`/ws/ios/{userId}`)
- Test query WebSocket (`/ws/query/{userId}`)
- Show you what data is sent/received

### 3. **Use Backend API Directly**
You can test the backend endpoints directly using `curl` or any HTTP client:

```bash
# Test health check
curl https://memory-backend-328251955578.us-east1.run.app/

# Test upload endpoint (with a test image)
curl -X POST https://memory-backend-328251955578.us-east1.run.app/upload/test123 \
  -F "file=@test_image.jpg"
```

### 4. **Code Flow Visualization**
I can create diagrams showing:
- How memory captures flow from glasses → backend
- How queries work
- Where features are implemented
- What's missing

### 5. **Create Mock Tests**
I can create Swift test files that simulate app behavior without needing the UI.

## 🔧 Quick Test Commands

### Test Memory Capture WebSocket
```bash
# Using the test script
swift test_backend.swift
```

### Test Backend Health
```bash
curl https://memory-backend-328251955578.us-east1.run.app/
```

### Test Query Endpoint (via WebSocket)
The test script includes this - it sends a "What was I doing?" query.

## 📊 Understanding Code Flow

I can trace through the code and show you:

1. **Memory Capture Flow:**
   - `StreamSessionViewModel` captures photo
   - `GCPUploader` uploads to backend
   - `MemoryCaptureWebSocketClient` sends metadata
   - Backend processes with Gemini AI

2. **Query Flow:**
   - User asks "What was I doing?"
   - `QueryWebSocketClient` sends query
   - Backend searches memories
   - Returns summary answer

3. **Feature Status:**
   - ✅ What's implemented
   - ❌ What's missing
   - 🔨 How to add features

## 🎯 Best Workarounds for Your Situation

### Option 1: Code Analysis (Recommended)
I can walk you through the code step-by-step, showing:
- How each feature works
- Where to add new features
- What happens when you use the app

### Option 2: Backend Testing
Test the backend directly to see:
- If WebSocket connections work
- What responses you get
- How data flows

### Option 3: Create Feature Implementation Plan
I can create detailed implementation guides for missing features:
- Step-by-step code changes
- What files to modify
- How to test each piece

## 🚀 Next Steps

Would you like me to:
1. **Run the test script** to show backend connectivity?
2. **Trace through a specific feature** (e.g., "What just happened")?
3. **Create implementation code** for missing features?
4. **Generate a code flow diagram** showing how everything connects?

Let me know what you'd like to explore!
