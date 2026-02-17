"""
Segmentation viewer — side-by-side video playback (original + tracked + masked depth).

Shows source video alongside SAM2 tracking overlay and depth-masked segmentation.
Arrow keys to step, Space to play/pause, dropdown to switch videos.

Usage:
  python segmentation/seg_viewer.py data/segmentation/ --source-dir data/
"""

import argparse
from pathlib import Path

import cv2


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
    return frames, interval, total


def find_source_video(stem, source_dir):
    if source_dir is None:
        return None
    for ext in [".MOV", ".mov", ".mp4", ".MP4"]:
        candidate = source_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Segmentation Viewer</title>
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
  <h1>Segmentation Viewer</h1>
  <select id="vidSel">{{OPTIONS}}</select>
  <span class="status" id="status">Ready</span>
</div>
<div class="viewer">
  <div class="vpanel"><div class="lbl">Original</div><canvas id="srcCanvas"></canvas></div>
  <div class="vpanel"><div class="lbl">Tracking</div><canvas id="trkCanvas"></canvas></div>
  <div class="vpanel"><div class="lbl">Masked Depth</div><canvas id="mdCanvas"></canvas></div>
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
const canvases = {src: $('srcCanvas'), trk: $('trkCanvas'), md: $('mdCanvas')};
const ctxs = {};
for (const k in canvases) ctxs[k] = canvases[k].getContext('2d');

let currentVideo = '', totalFrames = 0, frame = 0, playing = false, timer = null;
let bitmaps = {src: [], trk: [], md: []};

const KINDS = ['source', 'tracked', 'masked_depth'];
const KEYS = ['src', 'trk', 'md'];

function showFrame(idx) {
  idx = Math.max(0, Math.min(idx, totalFrames - 1));
  frame = idx;
  for (let i = 0; i < KEYS.length; i++) {
    const k = KEYS[i];
    if (bitmaps[k][idx]) {
      canvases[k].width = bitmaps[k][idx].width;
      canvases[k].height = bitmaps[k][idx].height;
      ctxs[k].drawImage(bitmaps[k][idx], 0, 0);
    }
  }
  $('scrubber').value = idx;
  $('frameNum').textContent = (idx+1) + ' / ' + totalFrames;
}

async function preloadAllFrames(name, count) {
  for (const k of KEYS) {
    bitmaps[k].forEach(b => { if(b) b.close(); });
    bitmaps[k] = new Array(count).fill(null);
  }
  const BATCH = 6;
  for (let start = 0; start < count; start += BATCH) {
    if (currentVideo !== name) return;
    const end = Math.min(start + BATCH, count);
    const promises = [];
    for (let i = start; i < end; i++) {
      for (let ki = 0; ki < KINDS.length; ki++) {
        const kind = KINDS[ki], key = KEYS[ki];
        promises.push(
          fetch('/frame/' + name + '/' + kind + '/' + i)
            .then(r => r.ok ? r.blob() : null)
            .then(b => b ? createImageBitmap(b) : null)
            .then(bm => { if(bm) bitmaps[key][i] = bm; })
        );
      }
    }
    await Promise.all(promises);
    $('status').textContent = 'Cached ' + end + '/' + count;
    if (start === 0) showFrame(0);
  }
  $('status').textContent = count + ' frames ready';
}

async function loadVideo(name) {
  currentVideo = name;
  for (const k of KEYS) { bitmaps[k].forEach(b => { if(b) b.close(); }); bitmaps[k] = []; }
  stop();
  $('status').textContent = 'Loading...';
  const resp = await fetch('/info/' + name);
  const data = await resp.json();
  totalFrames = data.num_frames;
  $('scrubber').max = totalFrames - 1;
  preloadAllFrames(name, totalFrames);
}

let lastT = 0;
function play() { if (playing) return; playing = true; $('playBtn').textContent = 'Pause'; $('playBtn').classList.add('playing'); lastT = 0; timer = requestAnimationFrame(tick); }
function stop() { playing = false; $('playBtn').textContent = 'Play'; $('playBtn').classList.remove('playing'); if (timer) { cancelAnimationFrame(timer); timer = null; } }
function tick(ts) {
  if (!playing) return;
  if (!lastT) lastT = ts;
  const fps = parseInt($('fpsCtl').value) || 15;
  if (ts - lastT >= 1000/fps) { lastT = ts; if (frame >= totalFrames - 1) { stop(); return; } showFrame(frame + 1); }
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


def main():
    parser = argparse.ArgumentParser(description="Segmentation viewer")
    parser.add_argument("results_dir", help="Directory with tracked/masked_depth/composite videos")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response

    app = FastAPI()

    results_dir = Path(args.results_dir)
    source_dir = Path(args.source_dir) if args.source_dir else None

    # Find all tracked videos and derive stems
    tracked_files = sorted(results_dir.glob("*_tracked.mp4"))
    if not tracked_files:
        print(f"No tracked videos found in {results_dir}")
        return

    stems = []
    video_map = {}  # stem -> {tracked, masked_depth, composite, source}
    for tf in tracked_files:
        stem = tf.stem.replace("_tracked", "")
        stems.append(stem)
        video_map[stem] = {
            "tracked": tf,
            "masked_depth": results_dir / f"{stem}_masked_depth.mp4",
            "composite": results_dir / f"{stem}_composite.mp4",
        }
        src = find_source_video(stem, source_dir)
        if src:
            video_map[stem]["source"] = src

    print(f"Found {len(stems)} segmentation results:")
    for s in stems:
        vm = video_map[s]
        has = [k for k, v in vm.items() if v is not None and (not isinstance(v, Path) or v.exists())]
        print(f"  {s}: {', '.join(has)}")

    frame_store = {}

    def ensure_loaded(stem):
        if stem in frame_store:
            return
        print(f"Loading {stem}...")
        vm = video_map[stem]

        store = {}
        # Load tracked video as reference for frame count
        tracked_frames, interval, total = extract_frames(vm["tracked"], args.max_frames)
        n = len(tracked_frames)
        store["tracked"] = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() for f in tracked_frames]

        # Load masked depth
        md_path = vm.get("masked_depth")
        if md_path and md_path.exists():
            md_frames, _, _ = extract_frames(md_path, args.max_frames)
            md_frames = md_frames[:n]
            store["masked_depth"] = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() for f in md_frames]

        # Load source
        src_path = vm.get("source")
        if src_path and src_path.exists():
            cap = cv2.VideoCapture(str(src_path))
            stotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_frames = []
            scount = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if scount % max(1, stotal // args.max_frames) == 0:
                    src_frames.append(frame)
                scount += 1
            cap.release()
            src_frames = src_frames[:n]
            store["source"] = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() for f in src_frames]

        frame_store[stem] = store
        print(f"  {stem}: {n} frames loaded ({', '.join(store.keys())})")

    default_stem = "IMG_4723" if "IMG_4723" in stems else stems[0]
    options_html = "\n".join(
        f'<option value="{s}"{" selected" if s == default_stem else ""}>{s}</option>'
        for s in stems
    )
    page_html = PAGE_HTML.replace("{{OPTIONS}}", options_html)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return page_html

    @app.get("/info/{stem}")
    async def video_info(stem: str):
        if stem not in video_map:
            return {"num_frames": 0}
        ensure_loaded(stem)
        tracked = frame_store[stem].get("tracked", [])
        return {"num_frames": len(tracked)}

    @app.get("/frame/{stem}/{kind}/{idx}")
    async def serve_frame(stem: str, kind: str, idx: int):
        if stem not in video_map:
            return Response(status_code=404)
        ensure_loaded(stem)
        frames = frame_store[stem].get(kind, [])
        if not frames:
            return Response(status_code=404)
        idx = max(0, min(idx, len(frames) - 1))
        return Response(content=frames[idx], media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})

    print(f"\nSegmentation viewer at http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
