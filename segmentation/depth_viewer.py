"""
Segmented Depth viewer — side-by-side video playback (original + masked depth + composite).

Pre-decodes all frames into ImageBitmaps for smooth playback.
Arrow keys to step, Space to play/pause, dropdown to switch videos.

Usage:
  python segmentation/depth_viewer.py data/segmentation/ --source-dir data/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path, max_frames=200):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // max_frames)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames, interval


def extract_matched_source(source_path, depth_count, depth_interval, source_total_hint=None, max_frames=200):
    """Extract source frames aligned to depth frame indices."""
    cap = cv2.VideoCapture(str(source_path))
    stotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    wanted = {}
    for i in range(depth_count):
        src_idx = min(int(i * depth_interval * stotal / max(stotal, 1)), stotal - 1)
        # Simple proportional mapping
        src_idx = min(int(i * stotal / depth_count), stotal - 1)
        wanted[i] = src_idx

    wanted_set = set(wanted.values())
    grabbed = {}
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count in wanted_set:
            grabbed[count] = frame
            if len(grabbed) == len(wanted_set):
                break
        count += 1
    cap.release()

    return [grabbed.get(wanted[i], np.zeros((100, 100, 3), dtype=np.uint8)) for i in range(depth_count)]


def find_source_video(name, source_dir):
    """Find the original video matching a stem like IMG_4723."""
    if not source_dir:
        return None
    for ext in [".MOV", ".mov", ".mp4", ".MP4"]:
        candidate = source_dir / f"{name}{ext}"
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
<title>Segmented Depth Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#ddd;font-family:system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
.topbar{display:flex;align-items:center;gap:12px;padding:8px 16px;background:#151515;border-bottom:1px solid #2a2a2a;flex-shrink:0}
.topbar h1{font-size:14px;font-weight:600;color:#999}
.topbar select{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;padding:3px 8px;border-radius:3px;font-size:12px}
.topbar .objects{font-size:11px;color:#6a9fff;margin-left:8px}
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
  <h1>Segmented Depth</h1>
  <select id="vidSel">{{OPTIONS}}</select>
  <span class="objects" id="objectsLabel"></span>
  <span class="status" id="status">Ready</span>
</div>
<div class="viewer">
  <div class="vpanel"><div class="lbl">Original</div><canvas id="srcCanvas"></canvas></div>
  <div class="vpanel"><div class="lbl">Object Depth</div><canvas id="maskedCanvas"></canvas></div>
  <div class="vpanel"><div class="lbl">Composite</div><canvas id="compCanvas"></canvas></div>
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
const srcCanvas = $('srcCanvas'), maskedCanvas = $('maskedCanvas'), compCanvas = $('compCanvas');
const srcCtx = srcCanvas.getContext('2d'), maskedCtx = maskedCanvas.getContext('2d'), compCtx = compCanvas.getContext('2d');

let currentVideo = '', totalFrames = 0, frame = 0, playing = false, timer = null;
let srcBitmaps = [], maskedBitmaps = [], compBitmaps = [];

function showFrame(idx) {
  idx = Math.max(0, Math.min(idx, totalFrames - 1));
  frame = idx;
  if (srcBitmaps[idx]) {
    srcCanvas.width = srcBitmaps[idx].width; srcCanvas.height = srcBitmaps[idx].height;
    srcCtx.drawImage(srcBitmaps[idx], 0, 0);
  }
  if (maskedBitmaps[idx]) {
    maskedCanvas.width = maskedBitmaps[idx].width; maskedCanvas.height = maskedBitmaps[idx].height;
    maskedCtx.drawImage(maskedBitmaps[idx], 0, 0);
  }
  if (compBitmaps[idx]) {
    compCanvas.width = compBitmaps[idx].width; compCanvas.height = compBitmaps[idx].height;
    compCtx.drawImage(compBitmaps[idx], 0, 0);
  }
  $('scrubber').value = idx;
  $('frameNum').textContent = (idx+1) + ' / ' + totalFrames;
}

async function preloadAllFrames(name, count) {
  srcBitmaps = new Array(count).fill(null);
  maskedBitmaps = new Array(count).fill(null);
  compBitmaps = new Array(count).fill(null);
  const BATCH = 6;
  for (let start = 0; start < count; start += BATCH) {
    if (currentVideo !== name) return;
    const end = Math.min(start + BATCH, count);
    const promises = [];
    for (let i = start; i < end; i++) {
      promises.push(
        fetch('/frame/' + name + '/source/' + i).then(r => r.blob()).then(b => createImageBitmap(b)).then(bm => { srcBitmaps[i] = bm; }),
        fetch('/frame/' + name + '/masked/' + i).then(r => r.blob()).then(b => createImageBitmap(b)).then(bm => { maskedBitmaps[i] = bm; }),
        fetch('/frame/' + name + '/composite/' + i).then(r => r.blob()).then(b => createImageBitmap(b)).then(bm => { compBitmaps[i] = bm; })
      );
    }
    await Promise.all(promises);
    $('status').textContent = 'Cached ' + end + '/' + count;
    if (start === 0 && srcBitmaps[0]) showFrame(0);
  }
  $('status').textContent = count + ' frames ready';
}

async function loadVideo(name) {
  currentVideo = name;
  [srcBitmaps, maskedBitmaps, compBitmaps].forEach(arr => arr.forEach(b => { if(b) b.close(); }));
  srcBitmaps = []; maskedBitmaps = []; compBitmaps = [];
  stop();
  $('status').textContent = 'Loading...';
  const resp = await fetch('/info/' + name);
  const data = await resp.json();
  totalFrames = data.num_frames;
  $('objectsLabel').textContent = data.objects ? data.objects.join(', ') : '';
  $('scrubber').max = totalFrames - 1;
  preloadAllFrames(name, totalFrames);
}

let lastT = 0;
function play() { if(playing) return; playing=true; $('playBtn').textContent='Pause'; $('playBtn').classList.add('playing'); lastT=0; timer=requestAnimationFrame(tick); }
function stop() { playing=false; $('playBtn').textContent='Play'; $('playBtn').classList.remove('playing'); if(timer){cancelAnimationFrame(timer);timer=null;} }
function tick(ts) {
  if(!playing) return;
  if(!lastT) lastT=ts;
  const fps=parseInt($('fpsCtl').value)||15;
  if(ts-lastT>=1000/fps){ lastT=ts; if(frame>=totalFrames-1){stop();return;} showFrame(frame+1); }
  timer=requestAnimationFrame(tick);
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
    parser = argparse.ArgumentParser(description="Segmented Depth viewer")
    parser.add_argument("path", help="Directory containing *_masked_depth.mp4 and *_composite.mp4")
    parser.add_argument("--source-dir", type=str, default=None, help="Directory with original .MOV files")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    import json as jsonmod

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response

    fapi = FastAPI()

    input_path = Path(args.path)
    source_dir = Path(args.source_dir) if args.source_dir else None

    # Find all video sets: look for *_masked_depth.mp4
    masked_files = sorted(input_path.glob("*_masked_depth.mp4"))
    if not masked_files:
        print(f"No *_masked_depth.mp4 files found in {args.path}")
        return

    # Build video sets keyed by stem (e.g. "IMG_4723")
    video_sets = {}
    for mf in masked_files:
        stem = mf.stem.replace("_masked_depth", "")
        comp = input_path / f"{stem}_composite.mp4"
        det_json = input_path / f"{stem}_seg_depth.json"
        src = find_source_video(stem, source_dir)

        if not comp.exists():
            print(f"  WARN: missing composite for {stem}, skipping")
            continue

        objects = []
        if det_json.exists():
            try:
                data = jsonmod.loads(det_json.read_text())
                if data:
                    first_frame = next(iter(data.values()))
                    objects = list({d["label"] for d in first_frame if "label" in d})
            except Exception:
                pass

        video_sets[stem] = {
            "masked": mf,
            "composite": comp,
            "source": src,
            "objects": objects,
        }

    print(f"Found {len(video_sets)} video sets")
    for stem, vs in sorted(video_sets.items()):
        src_name = Path(vs["source"]).name if vs["source"] else "none"
        print(f"  {stem}: objects={vs['objects']}, source={src_name}")

    # Lazy frame store
    frame_store = {}

    def ensure_loaded(stem):
        if stem in frame_store:
            return
        vs = video_sets[stem]
        print(f"Loading {stem}...")

        masked_frames, m_interval = extract_frames(vs["masked"], args.max_frames)
        comp_frames, _ = extract_frames(vs["composite"], args.max_frames)

        n = min(len(masked_frames), len(comp_frames))
        masked_frames, comp_frames = masked_frames[:n], comp_frames[:n]

        if vs["source"]:
            source_frames = extract_matched_source(vs["source"], n, m_interval, max_frames=args.max_frames)
        else:
            source_frames = masked_frames

        n = min(n, len(source_frames))
        masked_frames = masked_frames[:n]
        comp_frames = comp_frames[:n]
        source_frames = source_frames[:n]

        # Encode as JPEGs
        store = {"masked": [], "composite": [], "source": []}
        for m, c, s in zip(masked_frames, comp_frames, source_frames):
            _, mb = cv2.imencode(".jpg", m, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, cb = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, sb = cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 85])
            store["masked"].append(mb.tobytes())
            store["composite"].append(cb.tobytes())
            store["source"].append(sb.tobytes())

        frame_store[stem] = store
        print(f"  {stem}: {n} frame triples loaded")

    options_html = "\n".join(
        f'<option value="{stem}">{stem}</option>'
        for stem in sorted(video_sets.keys())
    )
    page_html = PAGE_HTML.replace("{{OPTIONS}}", options_html)

    @fapi.get("/", response_class=HTMLResponse)
    async def index():
        return page_html

    @fapi.get("/info/{stem}")
    async def video_info(stem: str):
        if stem not in video_sets:
            return {"num_frames": 0, "objects": []}
        ensure_loaded(stem)
        return {
            "num_frames": len(frame_store[stem]["masked"]),
            "objects": video_sets[stem]["objects"],
        }

    @fapi.get("/frame/{stem}/{kind}/{idx}")
    async def serve_frame(stem: str, kind: str, idx: int):
        if stem not in video_sets or kind not in ("masked", "composite", "source"):
            return Response(status_code=404)
        ensure_loaded(stem)
        frames = frame_store[stem][kind]
        idx = max(0, min(idx, len(frames) - 1))
        return Response(content=frames[idx], media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})

    print(f"\nViewer at http://localhost:{args.port}")
    uvicorn.run(fapi, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
