#!/usr/bin/env python3
"""
Ray-Ban Meta Glasses - Video Frame Receiver
Receives frames from Android app via HTTP for VGGT 3D reconstruction

Usage:
    pip install flask opencv-python numpy
    python test_receiver.py

The receiver will listen on http://0.0.0.0:8080/frame
Configure Android app to stream to your phone's hotspot IP (e.g., 172.20.10.1:8080)
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
from datetime import datetime
import os
import sys
import socket

app = Flask(__name__)

# Configuration
FRAMES_DIR = 'frames'
# Set to True to display frames in window (requires X11)
DISPLAY_FRAMES = False

# Statistics
frame_count = 0
start_time = datetime.utcnow()

# Create frames directory
os.makedirs(FRAMES_DIR, exist_ok=True)


@app.route('/frame', methods=['POST'])
def receive_frame():
    """
    Receive video frame from Android app

    Expected multipart/form-data:
        - image: JPEG binary data
        - timestamp: ISO 8601 timestamp
        - frame_number: Sequential frame counter
        - width: Frame width in pixels
        - height: Frame height in pixels
    """
    global frame_count

    try:
        # Extract frame data
        image_file = request.files.get('image')
        timestamp = request.form.get('timestamp', '')
        frame_number = int(request.form.get('frame_number', 0))
        width = int(request.form.get('width', 0))
        height = int(request.form.get('height', 0))

        if not image_file:
            return jsonify({'error': 'No image provided'}), 400

        # Convert to OpenCV format
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Save frame
        filename = f'{FRAMES_DIR}/frame_{frame_number:06d}.jpg'
        cv2.imwrite(filename, frame)

        # Calculate statistics
        frame_count += 1
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Display info
        print(f"📸 Frame #{frame_number:6d} | "
              f"{width}x{height} | "
              f"{len(image_bytes):6d} bytes | "
              f"{timestamp} | "
              f"FPS: {fps:4.1f} | "
              f"Saved: {filename}")

        # Optional: Display frame in window
        if DISPLAY_FRAMES:
            cv2.imshow('Ray-Ban Stream', frame)
            cv2.waitKey(1)

        # TODO: Pass to VGGT pipeline
        # vggt_processor.add_frame(frame, timestamp)

        return jsonify({
            'status': 'ok',
            'frame_number': frame_number,
            'received_at': datetime.utcnow().isoformat() + 'Z',
            'total_frames': frame_count,
            'current_fps': round(fps, 2)
        })

    except Exception as e:
        print(f"❌ Error processing frame: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Health check endpoint"""
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    fps = frame_count / elapsed if elapsed > 0 else 0

    return jsonify({
        'status': 'running',
        'frames_received': frame_count,
        'uptime_seconds': round(elapsed, 1),
        'current_fps': round(fps, 2),
        'frames_directory': FRAMES_DIR
    })


@app.route('/stats', methods=['GET'])
def stats():
    """Detailed statistics endpoint"""
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    fps = frame_count / elapsed if elapsed > 0 else 0

    # Count saved frames
    saved_frames = len(
        [f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])

    return jsonify({
        'status': 'running',
        'frames_received': frame_count,
        'frames_saved': saved_frames,
        'uptime_seconds': round(elapsed, 1),
        'current_fps': round(fps, 2),
        'frames_directory': os.path.abspath(FRAMES_DIR),
        'display_enabled': DISPLAY_FRAMES
    })


@app.route('/clear', methods=['POST'])
def clear_frames():
    """Clear all saved frames"""
    try:
        files = [f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')]
        for f in files:
            os.remove(os.path.join(FRAMES_DIR, f))

        print(f"🗑️  Cleared {len(files)} frames")

        return jsonify({
            'status': 'ok',
            'frames_cleared': len(files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def print_banner(port: int):
    """Print startup banner"""
    print("=" * 60)
    print("🚀 Ray-Ban Meta Glasses - Video Frame Receiver")
    print("=" * 60)
    print(f"📡 Listening on: http://0.0.0.0:{port}")
    print(f"💾 Frames directory: {os.path.abspath(FRAMES_DIR)}")
    print(f"🖼️  Display frames: {DISPLAY_FRAMES}")
    print("")
    print("Endpoints:")
    print("  POST /frame   - Receive video frame")
    print("  GET  /status  - Health check")
    print("  GET  /stats   - Detailed statistics")
    print("  POST /clear   - Clear saved frames")
    print("")
    print("Phone Hotspot IPs:")
    print(f"  iOS:     172.20.10.1:{port}")
    print(f"  Android: 192.168.43.1:{port}")
    print("")
    print("Configure Android app:")
    print("  1. Enable phone hotspot")
    print("  2. Connect laptop to hotspot")
    print("  3. In app: Settings → Computer Streaming")
    print(f"  4. IP: 172.20.10.1, Port: {port}")
    print("  5. Enable streaming")
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print("")


_find_available_port = None


def _find_available_port(start: int, max_tries: int = 10, host: str = '0.0.0.0') -> int:
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f'No available port in range {start}-{start + max_tries - 1}')


if __name__ == '__main__':
    # pick a port (try next ports if 8080 is occupied)
    chosen_port = _find_available_port(8080, max_tries=10)
    if chosen_port != 8080:
        print(f"Port 8080 is in use — using {chosen_port} instead")

    print_banner(chosen_port)

    try:
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=chosen_port,
            threaded=True,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down receiver...")
        print(f"📊 Final stats: {frame_count} frames received")

        # Cleanup
        if DISPLAY_FRAMES:
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
