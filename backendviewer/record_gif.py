#!/usr/bin/env python3
"""Records the flowchart data-flow animation as a GIF.

Usage:
    python record_gif.py

Requirements:
    pip install playwright pillow
    playwright install chromium
"""

import asyncio
import io
import os
import threading
import http.server
import functools
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

WIDTH = 1920
HEIGHT = 1080
FPS = 12
FRAME_DELAY_MS = 1000 // FPS
DURATION_SEC = 60
OUTPUT = Path(__file__).parent / "flowchart_dataflow.gif"
HTML_PATH = Path(__file__).parent / "flowchart.html"
SERVE_DIR = Path(__file__).parent
PORT = 8397


def start_server():
    """Start a local HTTP server so video files load properly."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SERVE_DIR))
    server = http.server.HTTPServer(("127.0.0.1", PORT), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def main():
    server = start_server()
    print(f"Serving on http://127.0.0.1:{PORT}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        await page.goto(f"http://127.0.0.1:{PORT}/flowchart.html", wait_until="networkidle")

        # Wait for entrance animations
        await asyncio.sleep(3)

        # Slow down flow animation for recording (2x slower)
        await page.evaluate("window.flowSpeedMultiplier = 2")

        # Switch to data flow mode
        await page.keyboard.press("2")
        # Hide mode toggle UI for clean recording
        await page.evaluate("""() => {
            const toggle = document.querySelector('.mode-toggle');
            if (toggle) toggle.style.display = 'none';
        }""")
        await asyncio.sleep(0.3)

        total_frames = DURATION_SEC * FPS
        frames: list[Image.Image] = []
        print(f"Recording {total_frames} frames at {FPS}fps ({DURATION_SEC}s)...")

        for i in range(total_frames):
            screenshot = await page.screenshot(type="png")
            img = Image.open(io.BytesIO(screenshot)).convert("RGBA")
            rgb = Image.new("RGB", img.size, (8, 9, 13))
            rgb.paste(img, mask=img.split()[3])
            frames.append(rgb)

            if i % FPS == 0:
                sec = i // FPS
                pct = round(i / total_frames * 100)
                print(f"  {sec}s / {DURATION_SEC}s ({pct}%)", end="\r")

            await asyncio.sleep(FRAME_DELAY_MS / 1000)

        await browser.close()

    server.shutdown()

    # Save as mp4 via ffmpeg (pipe PNGs in, get h264 out)
    mp4_out = OUTPUT.with_suffix('.mp4')
    print(f"\nEncoding MP4 ({len(frames)} frames)...")
    import subprocess
    proc = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264', '-crf', '20', '-preset', 'slow',
        '-pix_fmt', 'yuv420p',
        str(mp4_out),
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    import numpy as np
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    size_mb = mp4_out.stat().st_size / 1024 / 1024
    print(f"Saved to {mp4_out} ({size_mb:.1f} MB)")

    # Also save GIF (lower res for size)
    print("Encoding GIF...")
    gif_frames = [f.resize((960, 540), Image.LANCZOS) for f in frames]
    gif_frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=gif_frames[1:],
        duration=FRAME_DELAY_MS,
        loop=0,
        optimize=True,
    )
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"Saved to {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
