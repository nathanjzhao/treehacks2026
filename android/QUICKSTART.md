# Quick Start Guide: Multi-Destination Video Streaming

Test the video streaming feature from Meta Ray-Ban glasses to computer and cloud.

---

## Prerequisites

- Meta Ray-Ban smart glasses (paired with phone)
- Android phone with app installed
- Laptop/computer on same network OR phone hotspot
- Python 3.8+ installed on laptop (for test receiver)

---

## Step 1: Set Up Test Receiver on Laptop

### Option A: Simple Flask Receiver (Recommended for Testing)

**1. Create a test directory:**
```bash
mkdir ~/raybans-receiver
cd ~/raybans-receiver
```

**2. Create `receiver.py`:**
```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Create frames directory
os.makedirs('frames', exist_ok=True)

frame_count = 0

@app.route('/frame', methods=['POST'])
def receive_frame():
    global frame_count

    # Extract frame data
    image_file = request.files.get('image')
    timestamp = request.form.get('timestamp', '')
    frame_number = int(request.form.get('frame_number', 0))
    width = int(request.form.get('width', 0))
    height = int(request.form.get('height', 0))

    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    # Convert to OpenCV format
    nparr = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    # Save frame
    filename = f'frames/frame_{frame_number:06d}.jpg'
    cv2.imwrite(filename, frame)

    # Display info
    print(f"📸 Frame #{frame_number} | {width}x{height} | {timestamp} | Saved: {filename}")

    # Optional: Display frame in window
    # cv2.imshow('Ray-Ban Stream', frame)
    # cv2.waitKey(1)

    frame_count += 1

    return jsonify({
        'status': 'ok',
        'frame_number': frame_number,
        'received_at': datetime.utcnow().isoformat() + 'Z',
        'total_frames': frame_count
    })

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'running',
        'frames_received': frame_count
    })

if __name__ == '__main__':
    print("🚀 Ray-Ban Video Receiver Starting...")
    print("📡 Listening on http://0.0.0.0:8080")
    print("💾 Frames will be saved to ./frames/")
    print("\nPress Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
```

**3. Install dependencies:**
```bash
pip install flask opencv-python numpy
```

**4. Run the receiver:**
```bash
python receiver.py
```

You should see:
```
🚀 Ray-Ban Video Receiver Starting...
📡 Listening on http://0.0.0.0:8080
💾 Frames will be saved to ./frames/

Press Ctrl+C to stop
```

---

## Step 2: Network Setup

### Option A: Phone Hotspot (Recommended for Stanford Wi-Fi)

**On Phone:**
1. Go to Settings → Personal Hotspot (iPhone) or Hotspot & Tethering (Android)
2. Enable Personal Hotspot
3. Note the hotspot name and password

**On Laptop:**
1. Connect to phone's hotspot network
2. Wait for connection to establish

**Find Phone's IP Address:**
- **iPhone hotspot:** Phone IP is always `172.20.10.1`
- **Android hotspot:** Phone IP is usually `192.168.43.1`

You can verify by pinging:
```bash
# Try iPhone default
ping 172.20.10.1

# Or try Android default
ping 192.168.43.1
```

**Test the receiver is accessible:**
```bash
curl http://172.20.10.1:8080/status
# Should return: {"status":"running","frames_received":0}
```

### Option B: Same Wi-Fi Network (May not work on Stanford Wi-Fi due to AP isolation)

**Find Laptop IP:**
```bash
# On Mac/Linux
ifconfig | grep "inet "

# On Windows
ipconfig
```

Look for an IP like `192.168.1.xxx` or `10.x.x.x`

---

## Step 3: Configure Android App

**1. Launch the app** on your Android phone

**2. Pair Ray-Ban glasses** (if not already paired)
   - Follow the in-app pairing flow
   - Grant camera permissions when prompted

**3. Start streaming from glasses**
   - Tap "Start Stream"
   - You should see live video from the glasses

**4. Open Streaming Settings**
   - Tap the ⚙️ (Settings) icon on the stream screen (top-left floating button)

**5. Configure Computer Destination**
   - Toggle "Enable Computer Streaming" ON
   - Enter IP Address: `172.20.10.1` (or your laptop IP)
   - Port: `8080`
   - Adjust Target FPS: `7` (recommended)
   - Adjust JPEG Quality: `70%`
   - Tap "Save Computer Settings"

**6. (Optional) Configure Cloud Destination**
   - Toggle "Enable Cloud Streaming" ON
   - User ID: Enter your test user ID (e.g., `test_user_1`)
   - Tap "Save Cloud Settings"

---

## Step 4: Start Streaming

**1. Return to Stream Screen**

**2. Check Streaming Status**
   - You should see a status badge at the top of the screen
   - Badge should show "Streaming: Computer" (and/or "Cloud" if enabled)
   - Should display FPS and bandwidth

**3. Open Debug Console** (to monitor activity)
   - Tap the 🐛 (Bug) icon in top-right corner
   - Toggle the debug switch to ON
   - You should see logs appearing:
     ```
     20:15:32.123 [INFO]  ComputerStream  Connecting to 172.20.10.1:8080
     20:15:32.234 [INFO]  ComputerStream  Connection established
     20:15:32.678 [DEBUG] ComputerStream  Sending frame #0 (42183 bytes)
     20:15:32.892 [INFO]  ComputerStream  Frame #0 sent successfully (200)
     ```

**4. Check Laptop Receiver**
   - You should see frames arriving in the terminal:
     ```
     📸 Frame #0 | 640x480 | 2026-02-13T20:15:32.678Z | Saved: frames/frame_000000.jpg
     📸 Frame #1 | 640x480 | 2026-02-13T20:15:33.123Z | Saved: frames/frame_000001.jpg
     📸 Frame #2 | 640x480 | 2026-02-13T20:15:33.678Z | Saved: frames/frame_000002.jpg
     ```

**5. Verify Frames on Disk**
   ```bash
   ls -lh frames/
   # Should show growing list of JPEG files
   ```

**6. View a frame**
   ```bash
   # On Mac
   open frames/frame_000000.jpg

   # On Linux
   xdg-open frames/frame_000000.jpg

   # On Windows
   start frames\frame_000000.jpg
   ```

---

## Step 5: Monitor Performance

**In the App:**
- Check streaming status badge (top of stream screen):
  ```
  Streaming: Computer+Cloud
  7.2 FPS | 512 KB/s
  ```

**In Debug Console:**
- Monitor FPS: Should be around 5-10 FPS to computer
- Check for errors (red lines)
- Watch network latency

**In Streaming Settings:**
- View connection status for each destination
- Monitor bandwidth, dropped frames, and uptime

**On Laptop:**
- Monitor frame arrival rate:
  ```bash
  watch -n 1 'ls frames/ | wc -l'
  # Should increment every ~140ms (7 FPS)
  ```

---

## Troubleshooting

### Problem: "Connection Failed" in app

**Check 1:** Is the receiver running?
```bash
curl http://172.20.10.1:8080/status
```
If this fails, receiver isn't accessible.

**Check 2:** Is laptop on phone's hotspot?
```bash
ping 172.20.10.1
```

**Check 3:** Firewall blocking port 8080?
```bash
# On Mac, temporarily disable firewall or add exception
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/python3
```

**Check 4:** Correct IP address?
- In Debug Console, look for connection attempts
- Verify IP matches your hotspot gateway

### Problem: Frames arriving slowly or not at all

**Check 1:** FPS setting too high?
- Lower FPS to 2-3 in settings
- Save and retry

**Check 2:** Network congestion?
- Disable cloud streaming temporarily
- Test computer streaming alone

**Check 3:** Phone battery saver mode?
- Disable battery optimization for the app
- Settings → Battery → App Battery Usage → CameraAccess → Unrestricted

### Problem: App crashes when enabling streaming

**Check 1:** Open Debug Console first
- Enable logging before starting stream
- Capture error logs

**Check 2:** Check Android Logcat
```bash
adb logcat | grep -i "cameraaccess"
```

### Problem: Debug console shows connection errors

**Common errors and fixes:**

- `Connection refused` → Receiver not running or wrong IP
- `Connection timeout` → Firewall blocking or laptop not on hotspot
- `HTTP 404` → Receiver running but wrong endpoint (should be `/frame`)
- `JSON parse error` → Receiver returning invalid response

---

## Testing Checklist

- [ ] Phone hotspot enabled
- [ ] Laptop connected to phone hotspot
- [ ] Python receiver running on `0.0.0.0:8080`
- [ ] Can curl `http://172.20.10.1:8080/status` successfully
- [ ] App configured with correct IP `172.20.10.1`
- [ ] Ray-Ban glasses paired and streaming
- [ ] Debug console enabled and showing logs
- [ ] Frames arriving on laptop (check `frames/` directory)
- [ ] Frame rate matches configured FPS (~7 FPS)
- [ ] No errors in debug console

---

## Next Steps: Integration with VGGT

Once frames are arriving successfully:

1. **Modify receiver to pass frames to VGGT:**
   ```python
   # In receiver.py, add VGGT processing
   import your_vggt_module

   @app.route('/frame', methods=['POST'])
   def receive_frame():
       # ... existing code to decode frame ...

       # Pass to VGGT for 3D reconstruction
       your_vggt_module.process_frame(
           image=frame,
           timestamp=timestamp,
           frame_number=frame_number
       )

       return jsonify({'status': 'ok', ...})
   ```

2. **Save frames with timestamps for batch processing:**
   ```python
   # Save with ISO timestamp in filename
   timestamp_clean = timestamp.replace(':', '-').replace('.', '-')
   filename = f'frames/frame_{timestamp_clean}.jpg'
   cv2.imwrite(filename, frame)
   ```

3. **Run FoundationPose on frames to detect objects:**
   - Load CAD models for objects you want to track (pills, glasses, etc.)
   - Process frames through FoundationPose to get 6DOF poses
   - Update object locations in your database

4. **Build 3D scene with VGGT:**
   - Use frame sequence for 3D point cloud reconstruction
   - Merge object poses from FoundationPose into 3D scene
   - Render in Vuer for visualization

5. **Query interface:**
   - User asks: "Where are my pills?"
   - Look up last known pose of "pills" object
   - Animate camera movement in Vuer to show location

---

## Performance Notes

**Expected Performance:**
- Computer streaming: 5-10 FPS at 640x480
- Bandwidth: ~500-1000 KB/s
- Battery drain: ~10-15% per hour
- Frame latency: 50-200ms (phone to laptop)

**Optimization Tips:**
- Lower FPS if battery draining too fast
- Reduce JPEG quality (50-60%) if bandwidth limited
- Disable cloud streaming during testing to isolate computer streaming
- Use debug console to identify bottlenecks

---

## Support

**Check logs:**
- Debug Console in app (real-time)
- Android Logcat: `adb logcat | grep -i cameraaccess`
- Receiver terminal output

**Common issues:**
- Network connectivity: Use phone hotspot
- Firewall: Temporarily disable or add port 8080 exception
- Performance: Lower FPS, reduce quality

**Test minimal setup:**
1. Phone hotspot only
2. Computer streaming only (disable cloud)
3. Low FPS (2-3)
4. Debug console enabled

Good luck! 🚀
