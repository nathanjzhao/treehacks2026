"""
Depth Anything V2 viewer — side-by-side video playback (original + depth).

Pre-decodes all frames into ImageBitmaps for smooth playback.
Arrow keys to step, Space to play/pause, dropdown to switch videos.

Usage:
  python viewer.py examples/ --source-dir ~/Downloads
"""

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame extraction (sequential — no seeking for HEVC compat)
# ---------------------------------------------------------------------------

def extract_matched_frames(depth_path, source_path, max_frames=200):
    dcap = cv2.VideoCapture(str(depth_path))
    dtotal = int(dcap.get(cv2.CAP_PROP_FRAME_COUNT))
    dinterval = max(1, dtotal // max_frames)

    depth_frames = []
    count = 0
    while True:
        ret, frame = dcap.read()
        if not ret:
            break
        if count % dinterval == 0:
            depth_frames.append(frame)
        count += 1
    dcap.release()

    if source_path:
        scap = cv2.VideoCapture(str(source_path))
        stotal = int(scap.get(cv2.CAP_PROP_FRAME_COUNT))

        wanted_src = {}
        for i in range(len(depth_frames)):
            src_idx = min(int(i * dinterval * stotal / max(dtotal, 1)), stotal - 1)
            wanted_src[i] = src_idx

        wanted_set = set(wanted_src.values())
        grabbed = {}
        scount = 0
        while True:
            ret, frame = scap.read()
            if not ret:
                break
            if scount in wanted_set:
                grabbed[scount] = frame
                if len(grabbed) == len(wanted_set):
                    break
            scount += 1
        scap.release()

        source_frames = [grabbed.get(wanted_src[i], depth_frames[i]) for i in range(len(depth_frames))]
        n = min(len(depth_frames), len(source_frames))
        depth_frames, source_frames = depth_frames[:n], source_frames[:n]
    else:
        source_frames = depth_frames

    print(f"  {Path(depth_path).stem}: {len(depth_frames)} frame pairs")
    return depth_frames, source_frames


def find_source_video(depth_path, source_dir):
    stem = Path(depth_path).stem.replace("_depth", "")
    search_dirs = [Path(depth_path).parent]
    if source_dir:
        search_dirs.insert(0, source_dir)
    for d in search_dirs:
        for ext in [".MOV", ".mov", ".mp4", ".MP4", ".avi", ".mkv"]:
            candidate = d / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Depth Anything V2</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#ddd;font-family:system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
.topbar{display:flex;align-items:center;gap:12px;padding:8px 16px;background:#151515;border-bottom:1px solid #2a2a2a;flex-shrink:0}
.topbar h1{font-size:14px;font-weight:600;color:#999}
.topbar select{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;padding:3px 8px;border-radius:3px;font-size:12px}
.status{margin-left:auto;font-size:11px;color:#555}
.viewer{flex:1;display:flex;gap:4px;padding:8px;min-height:0}
.vpanel{flex:1;display:flex;flex-direction:column;align-items:center}
.vpanel .lbl{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#555;margin-bottom:4px;flex-shrink:0}
.vpanel canvas{max-width:100%;max-height:100%;object-fit:contain;background:#000;border-radius:2px}
.controls{padding:6px 16px;background:#151515;border-top:1px solid #2a2a2a;display:flex;align-items:center;gap:8px;flex-shrink:0}
.controls button{background:#252525;color:#ccc;border:1px solid #3a3a3a;padding:4px 12px;border-radius:3px;cursor:pointer;font-size:12px}
.controls button:hover{background:#333}
.controls button.playing{background:#2563eb;border-color:#2563eb;color:#fff}
.controls input[type=range]{flex:1;accent-color:#2563eb}
.fnum{font-size:11px;color:#777;min-width:70px;text-align:center}
.controls label{font-size:10px;color:#666}
.controls .fpsctl{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;padding:2px 4px;border-radius:3px;font-size:11px;width:42px;text-align:center}
.shortcuts{font-size:9px;color:#444;padding:3px 16px;background:#151515}
kbd{background:#1e1e1e;padding:0 4px;border-radius:2px;border:1px solid #333}
</style>
</head>
<body>
<div class="topbar">
  <h1>Depth Anything V2</h1>
  <select id="vidSel">{{OPTIONS}}</select>
  <span class="status" id="status">Ready</span>
</div>
<div class="viewer">
  <div class="vpanel"><div class="lbl">Original</div><canvas id="srcCanvas"></canvas></div>
  <div class="vpanel"><div class="lbl">Depth</div><canvas id="depCanvas"></canvas></div>
</div>
<div class="controls">
  <button id="prevBtn">&larr;</button>
  <button id="playBtn">Play</button>
  <button id="nextBtn">&rarr;</button>
  <span class="fnum" id="frameNum">- / -</span>
  <input type="range" id="scrubber" min="0" max="0" value="0">
  <label>FPS</label>
  <input type="number" class="fpsctl" id="fpsCtl" value="15" min="1" max="60">
</div>
<div class="shortcuts">
  <kbd>Space</kbd> play/pause
  <kbd>&larr;</kbd><kbd>&rarr;</kbd> step
  <kbd>[</kbd><kbd>]</kbd> &plusmn;5
  <kbd>Home</kbd>/<kbd>End</kbd> first/last
</div>
<script>
const $ = id => document.getElementById(id);
const srcCanvas = $('srcCanvas'), depCanvas = $('depCanvas');
const srcCtx = srcCanvas.getContext('2d'), depCtx = depCanvas.getContext('2d');

let currentVideo = '', totalFrames = 0, frame = 0, playing = false, timer = null;
let srcBitmaps = [], depBitmaps = [];

function frameUrl(name, kind, idx) { return '/frame/' + name + '/' + kind + '/' + idx; }

function showFrame(idx) {
  idx = Math.max(0, Math.min(idx, totalFrames - 1));
  frame = idx;
  if (srcBitmaps[idx]) {
    srcCanvas.width = srcBitmaps[idx].width;
    srcCanvas.height = srcBitmaps[idx].height;
    srcCtx.drawImage(srcBitmaps[idx], 0, 0);
  }
  if (depBitmaps[idx]) {
    depCanvas.width = depBitmaps[idx].width;
    depCanvas.height = depBitmaps[idx].height;
    depCtx.drawImage(depBitmaps[idx], 0, 0);
  }
  $('scrubber').value = idx;
  $('frameNum').textContent = (idx+1) + ' / ' + totalFrames;
}

async function preloadAllFrames(name, count) {
  srcBitmaps = new Array(count).fill(null);
  depBitmaps = new Array(count).fill(null);

  const BATCH = 8;
  for (let start = 0; start < count; start += BATCH) {
    if (currentVideo !== name) return;
    const end = Math.min(start + BATCH, count);
    const promises = [];
    for (let i = start; i < end; i++) {
      promises.push(
        fetch(frameUrl(name, 'source', i)).then(r => r.blob()).then(b => createImageBitmap(b)).then(bm => { srcBitmaps[i] = bm; }),
        fetch(frameUrl(name, 'depth', i)).then(r => r.blob()).then(b => createImageBitmap(b)).then(bm => { depBitmaps[i] = bm; })
      );
    }
    await Promise.all(promises);
    $('status').textContent = 'Cached ' + end + '/' + count;
    if (start === 0 && srcBitmaps[0] && depBitmaps[0]) showFrame(0);
  }
  $('status').textContent = count + ' frames ready';
}

async function loadVideo(name) {
  currentVideo = name;
  srcBitmaps.forEach(b => { if(b) b.close(); });
  depBitmaps.forEach(b => { if(b) b.close(); });
  srcBitmaps = []; depBitmaps = [];
  stop();
  $('status').textContent = 'Loading...';
  const resp = await fetch('/info/' + name);
  const data = await resp.json();
  totalFrames = data.num_frames;
  $('scrubber').max = totalFrames - 1;
  preloadAllFrames(name, totalFrames);
}

let lastT = 0;
function play() {
  if (playing) return;
  playing = true;
  $('playBtn').textContent = 'Pause';
  $('playBtn').classList.add('playing');
  lastT = 0;
  timer = requestAnimationFrame(tick);
}
function stop() {
  playing = false;
  $('playBtn').textContent = 'Play';
  $('playBtn').classList.remove('playing');
  if (timer) { cancelAnimationFrame(timer); timer = null; }
}
function tick(ts) {
  if (!playing) return;
  if (!lastT) lastT = ts;
  const fps = parseInt($('fpsCtl').value) || 15;
  if (ts - lastT >= 1000/fps) {
    lastT = ts;
    if (frame >= totalFrames - 1) { stop(); return; }
    showFrame(frame + 1);
  }
  timer = requestAnimationFrame(tick);
}

$('playBtn').addEventListener('click', () => playing ? stop() : play());
$('prevBtn').addEventListener('click', () => { stop(); showFrame(frame-1); });
$('nextBtn').addEventListener('click', () => { stop(); showFrame(frame+1); });
$('scrubber').addEventListener('input', () => { stop(); showFrame(parseInt($('scrubber').value)); });
$('vidSel').addEventListener('change', () => loadVideo($('vidSel').value));

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' && e.target.type === 'number') return;
  switch(e.code) {
    case 'Space':       e.preventDefault(); playing ? stop() : play(); break;
    case 'ArrowLeft':   e.preventDefault(); stop(); showFrame(frame-1); break;
    case 'ArrowRight':  e.preventDefault(); stop(); showFrame(frame+1); break;
    case 'BracketLeft': e.preventDefault(); stop(); showFrame(frame-5); break;
    case 'BracketRight':e.preventDefault(); stop(); showFrame(frame+5); break;
    case 'Home':        e.preventDefault(); stop(); showFrame(0); break;
    case 'End':         e.preventDefault(); stop(); showFrame(totalFrames-1); break;
  }
});

if ($('vidSel').value) loadVideo($('vidSel').value);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 viewer")
    parser.add_argument("path", help="Depth video file or directory")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response

    app = FastAPI()

    input_path = Path(args.path)
    depth_files = sorted(input_path.glob("*_depth.mp4")) if input_path.is_dir() else [input_path]
    if not depth_files:
        print(f"No depth videos found at {args.path}")
        return

    file_map = {f.stem: f for f in depth_files}
    source_map = {}
    for df in depth_files:
        src = find_source_video(df, Path(args.source_dir) if args.source_dir else None)
        if src:
            source_map[df.stem] = src

    print(f"Found {len(depth_files)} depth videos")
    for n in sorted(file_map):
        src = source_map.get(n, "none")
        print(f"  {n} -> source: {Path(src).name if src != 'none' else 'none'}")

    # Lazy frame store
    frame_store = {}

    def ensure_loaded(name):
        if name in frame_store:
            return
        print(f"Loading {name}...")
        df = file_map[name]
        src = source_map.get(name)
        depth_frames, source_frames = extract_matched_frames(df, src, args.max_frames)

        depth_jpgs, source_jpgs = [], []
        for d, s in zip(depth_frames, source_frames):
            _, db = cv2.imencode(".jpg", d, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, sb = cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 85])
            depth_jpgs.append(db.tobytes())
            source_jpgs.append(sb.tobytes())

        frame_store[name] = {"depth": depth_jpgs, "source": source_jpgs}
        print(f"  {name}: {len(depth_frames)} frames loaded")

    options_html = "\n".join(
        f'<option value="{n}">{n.replace("_depth","")}</option>'
        for n in sorted(file_map.keys())
    )
    page_html = PAGE_HTML.replace("{{OPTIONS}}", options_html)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return page_html

    @app.get("/info/{name}")
    async def video_info(name: str):
        if name not in file_map:
            return {"num_frames": 0}
        ensure_loaded(name)
        return {"num_frames": len(frame_store[name]["depth"])}

    @app.get("/frame/{name}/{kind}/{idx}")
    async def serve_frame(name: str, kind: str, idx: int):
        if name not in file_map or kind not in ("depth", "source"):
            return Response(status_code=404)
        ensure_loaded(name)
        frames = frame_store[name][kind]
        idx = max(0, min(idx, len(frames) - 1))
        return Response(content=frames[idx], media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})

    print(f"\nViewer at http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
