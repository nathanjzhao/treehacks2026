"""
SAM3 live streaming client — connects to the Modal WebSocket endpoint,
sends webcam frames, and displays annotated results.

Usage:
    python sam3/client.py --url wss://nathanjzhao--sam3-stream-segment.modal.run/ws --prompt "person"
"""

import argparse
import time

import cv2
import numpy as np
import websocket


def main():
    parser = argparse.ArgumentParser(description="SAM3 live streaming client")
    parser.add_argument("--url", required=True, help="WebSocket URL (wss://...modal.run/ws)")
    parser.add_argument("--prompt", default="object", help="Text prompt for segmentation")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality (1-100)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    # Append prompt as query param if not already in URL
    url = args.url
    if "?" not in url:
        url += f"?prompt={args.prompt}"
    elif "prompt=" not in url:
        url += f"&prompt={args.prompt}"

    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print(f"Connecting to {url}...")
    ws = websocket.create_connection(url, timeout=30)
    print("Connected! Press 'q' to quit.")

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            # Encode and send
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            ws.send_binary(buf.tobytes())

            # Receive annotated frame
            data = ws.recv()
            if isinstance(data, bytes):
                arr = np.frombuffer(data, dtype=np.uint8)
                annotated = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if annotated is not None:
                    # FPS counter
                    fps_counter += 1
                    elapsed = time.time() - fps_time
                    if elapsed >= 1.0:
                        fps_display = fps_counter / elapsed
                        fps_counter = 0
                        fps_time = time.time()

                    cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("SAM3 Live", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        ws.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
