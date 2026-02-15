# Phone Camera → localhost Stream

Stream your iPhone camera to `localhost` on your computer for use in custom apps.

## Option A: Camo (same network / USB)

Best for when your phone is nearby. Free, no watermark, no time limit at 1080p.

### Prerequisites

- **Camo Camera** — iPhone app ([App Store](https://apps.apple.com/us/app/camo-webcam-for-mac-and-pc/id1514199064))
- **Camo Studio** — Mac desktop app ([camo.com](https://camo.com))
- **ffmpeg** — `brew install ffmpeg`
- **mediamtx** — `brew install mediamtx`

### Setup

1. Open Camo Studio on your Mac and connect your iPhone (USB or Wi-Fi).

2. Create a mediamtx config:

```yaml
# mediamtx.yml
paths:
  all_others:
```

3. Start mediamtx:

```bash
mediamtx mediamtx.yml
```

4. In a second terminal, pipe the Camo virtual webcam to mediamtx:

```bash
ffmpeg -f avfoundation -framerate 30 -video_size 1920x1080 -i "2:none" \
  -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp \
  rtsp://localhost:8554/live/stream
```

> **Note:** The device index `2` may differ on your machine. Run `ffmpeg -f avfoundation -list_devices true -i ""` to find `Camo Camera` in the list.

5. Verify: open `http://localhost:8889/live/stream/` in a browser.

### Endpoints

| Protocol | URL                                      | Latency |
|----------|------------------------------------------|---------|
| WebRTC   | `http://localhost:8889/live/stream/`     | ~200ms  |
| RTSP     | `rtsp://localhost:8554/live/stream`      | ~500ms  |
| HLS      | `http://localhost:8888/live/stream/`     | 2-6s    |

### Alternative: Python MJPEG server (no mediamtx needed)

```python
# pip install opencv-python flask
import cv2
from flask import Flask, Response

app = Flask(__name__)
cap = cv2.VideoCapture(2)  # Camo device index

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, jpg = cv2.imencode('.jpg', frame)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=8080)
```

Stream available at `http://localhost:8080/video`.

---

## Option B: Larix Broadcaster (remote / different networks)

Best for when your phone is far from your computer. Requires Tailscale for cross-network connectivity.

### Prerequisites

- **Larix Broadcaster** — iPhone app ([App Store](https://apps.apple.com/us/app/larix-broadcaster/id1042474385)) — free with watermark after 30 min
- **mediamtx** — `brew install mediamtx`
- **Tailscale** — installed on both devices, same account ([tailscale.com](https://tailscale.com))

### Setup

1. Get your computer's Tailscale IP:

```bash
tailscale ip -4
```

2. Create `mediamtx.yml` and start mediamtx (same config as Option A).

3. In Larix: **Settings → Connections → + → Connection**
   - URL: `rtmp://<tailscale-ip>:1935/live/stream`

4. Tap the red record button in Larix.

5. Same endpoints as Option A apply.

### Notes

- Tailscale adds ~1-5ms latency on LAN, negligible for video streaming
- Larix free tier watermarks after 30 min — restart the stream to reset
- Works from anywhere with internet