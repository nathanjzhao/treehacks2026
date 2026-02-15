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
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

WIDTH = 1240
HEIGHT = 900
FPS = 8
FRAME_DELAY_MS = 1000 // FPS
DURATION_SEC = 30
OUTPUT = Path(__file__).parent / "flowchart.gif"
HTML_PATH = Path(__file__).parent / "flowchart.html"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

        await page.goto(f"file://{HTML_PATH.resolve()}", wait_until="networkidle")

        # Wait for entrance animations
        await asyncio.sleep(2.5)

        # Switch to data flow mode, hide chrome
        await page.keyboard.press("2")
        await page.evaluate("""() => {
            const toggle = document.querySelector('.mode-toggle');
            if (toggle) toggle.style.display = 'none';
            const nav = document.getElementById('flowNav');
            if (nav) nav.style.display = 'none';
        }""")
        await asyncio.sleep(0.3)

        total_frames = DURATION_SEC * FPS
        frames: list[Image.Image] = []
        print(f"Recording {total_frames} frames at {FPS}fps ({DURATION_SEC}s)...")

        for i in range(total_frames):
            screenshot = await page.screenshot(type="png")
            img = Image.open(io.BytesIO(screenshot)).convert("RGBA")
            # Convert to RGB for GIF (no alpha)
            rgb = Image.new("RGB", img.size, (8, 9, 13))  # --bg color
            rgb.paste(img, mask=img.split()[3])
            frames.append(rgb)

            if i % FPS == 0:
                pct = round(i / total_frames * 100)
                print(f"  Frame {i}/{total_frames} ({pct}%)", end="\r")

            await asyncio.sleep(FRAME_DELAY_MS / 1000)

        await browser.close()

    print(f"\nEncoding GIF...")
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DELAY_MS,
        loop=0,
        optimize=True,
    )
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"Saved to {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
