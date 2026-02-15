#!/usr/bin/env python3
"""Records the flowchart showcase + data-flow animation as MP4/GIF.

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
SHOWCASE_SEC = 14   # cycles through all 6 nodes quickly in headless
DATAFLOW_SEC = 50   # enough to cycle through flow paths
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


async def record_frames(page, duration_sec, label):
    """Record frames for a given duration."""
    total_frames = duration_sec * FPS
    frames: list[Image.Image] = []
    print(f"Recording {label}: {total_frames} frames at {FPS}fps ({duration_sec}s)...")

    for i in range(total_frames):
        screenshot = await page.screenshot(type="png")
        img = Image.open(io.BytesIO(screenshot)).convert("RGBA")
        rgb = Image.new("RGB", img.size, (8, 9, 13))
        rgb.paste(img, mask=img.split()[3])
        frames.append(rgb)

        if i % FPS == 0:
            sec = i // FPS
            pct = round(i / total_frames * 100)
            print(f"  [{label}] {sec}s / {duration_sec}s ({pct}%)", end="\r")

        await asyncio.sleep(FRAME_DELAY_MS / 1000)

    print()
    return frames


async def main():
    server = start_server()
    print(f"Serving on http://127.0.0.1:{PORT}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        await page.goto(f"http://127.0.0.1:{PORT}/flowchart.html", wait_until="networkidle")

        # Wait for entrance animations
        await asyncio.sleep(3)

        # Slow down animations for recording (3x slower)
        await page.evaluate("window.flowSpeedMultiplier = 3")

        # Hide mode toggle UI for clean recording
        await page.evaluate("""() => {
            const toggle = document.querySelector('.mode-toggle');
            if (toggle) toggle.style.display = 'none';
        }""")

        # Phase 1: Showcase mode — cycle through all 6 video nodes
        await page.keyboard.press("3")
        await asyncio.sleep(0.3)
        showcase_frames = await record_frames(page, SHOWCASE_SEC, "SHOWCASE")

        # Phase 2: Switch to data flow mode
        await page.keyboard.press("2")
        await asyncio.sleep(0.5)
        dataflow_frames = await record_frames(page, DATAFLOW_SEC, "DATA FLOW")

        await browser.close()

    server.shutdown()

    all_frames = showcase_frames + dataflow_frames
    total_duration = SHOWCASE_SEC + DATAFLOW_SEC

    # Save as mp4 via ffmpeg
    mp4_out = OUTPUT.with_suffix('.mp4')
    print(f"Encoding MP4 ({len(all_frames)} frames, {total_duration}s)...")
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
    for frame in all_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    size_mb = mp4_out.stat().st_size / 1024 / 1024
    print(f"Saved to {mp4_out} ({size_mb:.1f} MB)")

    # Also save GIF (lower res for size)
    print("Encoding GIF...")
    gif_frames = [f.resize((960, 540), Image.LANCZOS) for f in all_frames]
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
