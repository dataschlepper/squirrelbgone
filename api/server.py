#!/usr/bin/env python3
"""
SquirrelBGone — api/server.py
Dashboard for reviewing detections and testing hardware.

Usage (from repo root):
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Note: /api/spray writes a request file; detect.py picks it up on its next loop iteration.
Both processes can run simultaneously.
"""

import asyncio
import csv
import datetime
import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

LOG_DIR    = Path(os.environ.get("LOG_DIR",    "logs"))
FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", "frames"))

# Minimum confidence logged by detect.py
LOG_THRESHOLD  = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
# Threshold above which a detection is considered a reliable true positive
HIGH_CONF_THRESHOLD = float(os.environ.get("HIGH_CONF_THRESHOLD", "0.70"))

CORRECTIONS_PATH   = LOG_DIR / "corrections.csv"
CORRECTION_FIELDS  = ["flagged_at", "detection_timestamp", "class", "confidence", "frame_path"]
SPRAY_REQUEST_FILE  = LOG_DIR / "spray.request"
SOLENOID_STATE_FILE = LOG_DIR / "solenoid.state"
ZONE_FILE           = LOG_DIR / "feeder_zone.json"

app = FastAPI()

# ── MJPEG live stream ─────────────────────────────────────────────────────────
_frame_lock   = threading.Lock()
_latest_frame: bytes | None = None


def _rtsp_capture_loop() -> None:
    global _latest_frame
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return
    rtsp_url = os.environ.get("RTSP_URL", "")
    if not rtsp_url:
        return
    while True:
        cap = cv2.VideoCapture(rtsp_url)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                if w > 960:
                    scale = 960 / w
                    frame = cv2.resize(frame, (960, int(h * scale)))
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ok:
                    with _frame_lock:
                        _latest_frame = jpg.tobytes()
        finally:
            cap.release()
        time.sleep(2)


threading.Thread(target=_rtsp_capture_loop, daemon=True).start()


async def _mjpeg_gen():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        await asyncio.sleep(0.1)


@app.get("/api/stream")
async def mjpeg_stream():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


FRAMES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")


def _read_recent(minutes: int) -> list[dict]:
    cutoff = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
    rows = []

    # Check today's and yesterday's file so any window works near midnight
    for date in [datetime.date.today(), datetime.date.today() - datetime.timedelta(days=1)]:
        csv_path = LOG_DIR / f"detections_{date.isoformat()}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ts = datetime.datetime.fromisoformat(row["timestamp"])
                except (ValueError, KeyError):
                    continue
                if ts >= cutoff:
                    fp = Path(row.get("frame_path", ""))
                    row["image_url"] = f"/frames/{fp.name}" if fp.name else ""
                    rows.append(row)

    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    return rows


@app.get("/api/config")
def get_config():
    return {"log_threshold": LOG_THRESHOLD, "high_conf_threshold": HIGH_CONF_THRESHOLD}


@app.get("/api/detections")
def get_detections(minutes: int = 60):
    return _read_recent(minutes)


@app.post("/api/flag")
async def flag_detection(req: Request):
    body = await req.json()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not CORRECTIONS_PATH.exists()
    with open(CORRECTIONS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CORRECTION_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "flagged_at":          datetime.datetime.now().isoformat(timespec="seconds"),
            "detection_timestamp": body.get("timestamp", ""),
            "class":               body.get("class", ""),
            "confidence":          body.get("confidence", ""),
            "frame_path":          body.get("frame_path", ""),
        })
    return {"ok": True}


@app.post("/api/spray")
async def manual_spray(duration: float = 1.0):
    duration = max(0.1, min(duration, 10.0))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SPRAY_REQUEST_FILE.write_text(json.dumps({"duration": duration}))
    return {"ok": True, "duration": duration}


@app.get("/api/spray-status")
def spray_status():
    return {"pending": SPRAY_REQUEST_FILE.exists()}


@app.post("/api/solenoid/on")
def solenoid_on():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SOLENOID_STATE_FILE.write_text(json.dumps({"on": True}))
    return {"ok": True, "on": True}


@app.post("/api/solenoid/off")
def solenoid_off():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SOLENOID_STATE_FILE.write_text(json.dumps({"on": False}))
    return {"ok": True, "on": False}


@app.get("/api/zone")
def get_zone():
    try:
        return json.loads(ZONE_FILE.read_text())
    except Exception:
        return {"zone": None}


@app.post("/api/zone")
async def set_zone(req: Request):
    body = await req.json()
    zone = {
        "x1": float(body["x1"]),
        "y1": float(body["y1"]),
        "x2": float(body["x2"]),
        "y2": float(body["y2"]),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ZONE_FILE.write_text(json.dumps(zone))
    return {"ok": True, "zone": zone}


@app.delete("/api/zone")
def delete_zone():
    try:
        ZONE_FILE.unlink()
    except Exception:
        pass
    return {"ok": True}


@app.get("/api/solenoid-status")
def solenoid_status():
    try:
        data = json.loads(SOLENOID_STATE_FILE.read_text())
        return {"on": bool(data.get("on", False))}
    except Exception:
        return {"on": False}


MOBILE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>SquirrelBGone 💦</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Pacifico&family=Nunito:wght@400;700;900&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --pool:   #00B4D8;
      --sky:    #90E0EF;
      --deep:   #0077B6;
      --pink:   #FF006E;
      --coral:  #FF6B6B;
      --yellow: #FFD166;
      --mint:   #06D6A0;
      --cream:  #FFF9F0;
      --navy:   #023E8A;
    }

    body {
      font-family: 'Nunito', sans-serif;
      background: linear-gradient(160deg, #CAF0F8 0%, #90E0EF 40%, #48CAE4 70%, #00B4D8 100%);
      min-height: 100vh;
      color: var(--navy);
      -webkit-tap-highlight-color: transparent;
    }

    /* ── Header ─────────────────────────────────────────────────────────── */
    header {
      background: linear-gradient(135deg, #023E8A 0%, #0077B6 60%, #00B4D8 100%);
      padding: 18px 20px 22px;
      text-align: center;
      box-shadow: 0 4px 20px rgba(0,60,120,0.35);
      clip-path: ellipse(100% 100% at 50% 0%);
    }
    h1 {
      font-family: 'Pacifico', cursive;
      font-size: 2.2rem;
      color: var(--yellow);
      text-shadow: 3px 3px 0 rgba(0,0,0,0.25), 0 0 30px rgba(255,209,102,0.4);
    }
    .tagline {
      font-size: 0.72rem;
      color: var(--sky);
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 3px;
      margin-top: 4px;
    }

    /* ── Main ───────────────────────────────────────────────────────────── */
    main {
      max-width: 480px;
      margin: 0 auto;
      padding: 16px 14px 48px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    /* ── Card ───────────────────────────────────────────────────────────── */
    .card {
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border-radius: 22px;
      box-shadow: 0 4px 24px rgba(0, 100, 160, 0.13), 0 1px 4px rgba(0,0,0,0.06);
      overflow: hidden;
    }
    .card-inner { padding: 16px; }

    /* ── Live cam ───────────────────────────────────────────────────────── */
    .cam-header {
      padding: 10px 16px 8px;
      display: flex; align-items: center; gap: 8px;
      background: rgba(255,255,255,0.6);
      border-bottom: 1px solid rgba(0,180,216,0.2);
    }
    .live-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 8px #22c55e;
      animation: blink 1.4s ease-in-out infinite;
    }
    .live-label { font-weight: 900; font-size: 0.8rem; color: #16a34a; letter-spacing: 2px; }
    #stream-img {
      width: 100%; display: block;
      background: #0a0a1a;
      min-height: 200px;
      object-fit: contain;
    }
    .no-stream {
      min-height: 200px;
      display: none;
      flex-direction: column;
      align-items: center; justify-content: center;
      background: #0a0a1a;
      color: #555; gap: 8px;
      font-size: 0.85rem; font-weight: 700;
    }
    .no-stream-icon { font-size: 3rem; }

    /* ── Stats ──────────────────────────────────────────────────────────── */
    .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 14px; }
    .stat {
      background: linear-gradient(145deg, var(--deep), var(--pool));
      border-radius: 16px; padding: 14px 10px;
      text-align: center; color: white;
    }
    .stat-num {
      font-family: 'Pacifico', cursive;
      font-size: 2.2rem; color: var(--yellow);
      line-height: 1;
      text-shadow: 2px 2px 0 rgba(0,0,0,0.2);
    }
    .stat-label { font-size: 0.7rem; font-weight: 900; text-transform: uppercase; letter-spacing: 1px; opacity: 0.88; margin-top: 5px; }

    /* ── Blast button ───────────────────────────────────────────────────── */
    .blast-card .card-inner { text-align: center; padding: 20px 16px 22px; }
    .section-title {
      font-family: 'Pacifico', cursive;
      font-size: 1.05rem; color: var(--navy);
      margin-bottom: 18px;
    }

    #blast-btn {
      width: 190px; height: 190px;
      border-radius: 50%; border: none;
      background: linear-gradient(145deg, #FF6B6B, #FF006E);
      color: white;
      font-family: 'Nunito', sans-serif;
      font-weight: 900; font-size: 1.25rem;
      cursor: pointer;
      box-shadow: 0 8px 0 #990042, 0 14px 28px rgba(255,0,110,0.45);
      transform: translateY(0);
      transition: transform 0.08s, box-shadow 0.08s;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 2px;
      margin: 0 auto 18px;
      position: relative; overflow: hidden;
      -webkit-tap-highlight-color: transparent;
    }
    #blast-btn .btn-emoji { font-size: 3rem; line-height: 1; }
    #blast-btn .btn-text  { font-size: 1.1rem; letter-spacing: 1px; }

    #blast-btn:active:not(:disabled) {
      transform: translateY(6px);
      box-shadow: 0 2px 0 #990042, 0 4px 12px rgba(255,0,110,0.35);
    }
    #blast-btn:disabled {
      background: linear-gradient(145deg, #bbb, #999);
      box-shadow: 0 6px 0 #666, 0 10px 18px rgba(0,0,0,0.18);
      cursor: not-allowed;
    }
    #blast-btn:not(:disabled) {
      animation: idle-pulse 2.4s ease-in-out infinite;
    }
    @keyframes idle-pulse {
      0%, 100% { box-shadow: 0 8px 0 #990042, 0 14px 28px rgba(255,0,110,0.45); }
      50%       { box-shadow: 0 8px 0 #990042, 0 18px 42px rgba(255,0,110,0.7); }
    }

    /* water drop spawn animation */
    @keyframes drop-up {
      0%   { transform: translateY(0)    scale(1);   opacity: 1; }
      100% { transform: translateY(-90px) scale(0.4); opacity: 0; }
    }
    .drop {
      position: absolute; pointer-events: none;
      font-size: 1.6rem;
      animation: drop-up 0.75s ease-out forwards;
    }

    /* ── Duration row ───────────────────────────────────────────────────── */
    .dur-row {
      display: flex; align-items: center; justify-content: center;
      gap: 10px; margin-bottom: 14px;
    }
    .dur-label { font-weight: 900; color: var(--navy); font-size: 0.85rem; }
    #dur-input {
      width: 80px; padding: 8px 10px;
      border: 3px solid var(--pool); border-radius: 12px;
      font-family: 'Nunito', sans-serif; font-size: 0.95rem; font-weight: 700;
      color: var(--navy); text-align: center; background: white; outline: none;
    }
    #dur-input:focus { border-color: var(--deep); }

    /* ── Status ─────────────────────────────────────────────────────────── */
    #blast-status {
      min-height: 1.5em;
      font-weight: 700; font-size: 0.88rem;
      color: #555; transition: color 0.2s;
    }
    #blast-status.wait { color: var(--pool); }
    #blast-status.ok   { color: #16a34a; }
    #blast-status.err  { color: #dc2626; }

    /* ── Solenoid ───────────────────────────────────────────────────────── */
    .solenoid-inner {
      padding: 16px; text-align: center;
    }
    .solenoid-label {
      font-size: 0.7rem; font-weight: 900; text-transform: uppercase;
      letter-spacing: 2px; color: #888; margin-bottom: 12px;
    }
    .solenoid-row {
      display: flex; align-items: center; justify-content: center; gap: 14px;
    }
    .solenoid-dot {
      width: 14px; height: 14px; border-radius: 50%;
      background: #ccc; flex-shrink: 0;
      transition: background 0.3s, box-shadow 0.3s;
    }
    .solenoid-dot.on { background: #22c55e; box-shadow: 0 0 12px #22c55e; }
    #solenoid-btn {
      padding: 12px 28px; border-radius: 14px;
      border: 3px solid var(--deep);
      background: white; color: var(--deep);
      font-family: 'Nunito', sans-serif; font-size: 0.95rem; font-weight: 900;
      cursor: pointer; transition: all 0.2s;
    }
    #solenoid-btn.on {
      background: linear-gradient(135deg, #15803d, #22c55e);
      border-color: #15803d; color: white;
    }
    #solenoid-btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .solenoid-hint {
      font-size: 0.75rem; color: #888; margin-top: 10px; font-style: italic;
    }

    /* ── Recent detections ──────────────────────────────────────────────── */
    .recent-inner { padding: 14px 14px 16px; }
    .recent-label {
      font-size: 0.7rem; font-weight: 900; text-transform: uppercase;
      letter-spacing: 2px; color: #888; margin-bottom: 10px;
    }
    .det-scroll {
      display: flex; gap: 10px;
      overflow-x: auto; -webkit-overflow-scrolling: touch;
      padding-bottom: 4px; scrollbar-width: none;
    }
    .det-scroll::-webkit-scrollbar { display: none; }
    .det-chip {
      flex-shrink: 0;
      background: linear-gradient(135deg, var(--deep), var(--pool));
      border-radius: 12px; padding: 9px 14px;
      color: white; font-size: 0.8rem; font-weight: 700; white-space: nowrap;
    }
    .det-class { color: var(--yellow); display: block; }
    .det-empty { color: #aaa; font-size: 0.85rem; font-style: italic; }

    /* ── Animations ─────────────────────────────────────────────────────── */
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.3; }
    }

    /* ── Footer ─────────────────────────────────────────────────────────── */
    footer {
      text-align: center; padding: 12px 20px 24px;
      font-size: 0.68rem; color: rgba(0,80,140,0.55); font-weight: 700;
    }
    footer a { color: var(--deep); text-decoration: none; }
  </style>
</head>
<body>
<header>
  <h1>🐿️ SquirrelBGone 💦</h1>
  <p class="tagline">Spray First · Ask Questions Never</p>
</header>

<main>

  <!-- Live cam -->
  <div class="card">
    <div class="cam-header">
      <div class="live-dot"></div>
      <span class="live-label">LIVE</span>
    </div>
    <img id="stream-img" src="/api/stream" alt="Live cam"
         onerror="showNoStream()" onload="hideNoStream()">
    <div id="no-stream" class="no-stream">
      <span class="no-stream-icon">📷</span>
      <span>Camera offline</span>
    </div>
  </div>

  <!-- Stats -->
  <div class="card">
    <div class="stats-grid">
      <div class="stat">
        <div class="stat-num" id="squirrel-count">—</div>
        <div class="stat-label">🐿️ Squirrels Today</div>
      </div>
      <div class="stat">
        <div class="stat-num" id="last-seen" style="font-size:1.4rem">—</div>
        <div class="stat-label">⏱️ Last Spotted</div>
      </div>
    </div>
  </div>

  <!-- Blast button -->
  <div class="card blast-card">
    <div class="card-inner">
      <div class="section-title">🎯 Manual Fire Control</div>
      <button id="blast-btn" onclick="fireSpray()">
        <span class="btn-emoji">💦</span>
        <span class="btn-text">BLAST&nbsp;'EM</span>
      </button>
      <div class="dur-row">
        <span class="dur-label">Duration</span>
        <input id="dur-input" type="number" value="1.0" min="0.1" max="10" step="0.1">
        <span class="dur-label">sec</span>
      </div>
      <div id="blast-status">Ready to soak that fuzzy menace 🐿️</div>
    </div>
  </div>

  <!-- Solenoid hold-open -->
  <div class="card">
    <div class="solenoid-inner">
      <div class="solenoid-label">🚰 Solenoid — Hold Open</div>
      <div class="solenoid-row">
        <div class="solenoid-dot" id="solenoid-dot"></div>
        <button id="solenoid-btn" onclick="toggleSolenoid()">💧 Turn On</button>
      </div>
      <div class="solenoid-hint" id="solenoid-hint">Holds valve open until manually turned off</div>
    </div>
  </div>

  <!-- Recent detections -->
  <div class="card">
    <div class="recent-inner">
      <div class="recent-label">🔍 Recent Detections</div>
      <div class="det-scroll" id="det-scroll">
        <span class="det-empty">Loading…</span>
      </div>
    </div>
  </div>

</main>

<footer>
  SquirrelBGone &nbsp;·&nbsp; <a href="/">Full dashboard →</a>
</footer>

<script>
  // ── Stream error handling ────────────────────────────────────────────────
  function showNoStream() {
    document.getElementById('stream-img').style.display = 'none';
    document.getElementById('no-stream').style.display = 'flex';
  }
  function hideNoStream() {
    document.getElementById('stream-img').style.display = 'block';
    document.getElementById('no-stream').style.display = 'none';
  }

  // ── Water drop animation ─────────────────────────────────────────────────
  function spawnDrops(btn) {
    const emojis = ['💦', '💧', '🌊', '💦', '💧'];
    for (let i = 0; i < 7; i++) {
      const el = document.createElement('span');
      el.className = 'drop';
      el.textContent = emojis[Math.floor(Math.random() * emojis.length)];
      el.style.left = (15 + Math.random() * 70) + '%';
      el.style.top  = (20 + Math.random() * 60) + '%';
      el.style.animationDelay    = (Math.random() * 0.25) + 's';
      el.style.animationDuration = (0.55 + Math.random() * 0.4) + 's';
      btn.appendChild(el);
      el.addEventListener('animationend', () => el.remove());
    }
  }

  // ── Spray ────────────────────────────────────────────────────────────────
  async function fireSpray() {
    const duration = parseFloat(document.getElementById('dur-input').value) || 1.0;
    const btn    = document.getElementById('blast-btn');
    const status = document.getElementById('blast-status');
    btn.disabled = true;
    spawnDrops(btn);
    status.textContent = 'Firing! 💦💦💦';
    status.className = 'wait';
    try {
      const res  = await fetch('/api/spray?duration=' + duration, { method: 'POST' });
      const data = await res.json();
      if (data.ok) {
        status.textContent = `${data.duration}s spray queued — waiting for detect.py…`;
        status.className = 'wait';
        pollSpray(data.duration, btn, status);
      } else {
        status.textContent = 'Something went wrong 😬';
        status.className = 'err';
        btn.disabled = false;
      }
    } catch {
      status.textContent = 'Connection error 📡';
      status.className = 'err';
      btn.disabled = false;
    }
  }

  async function pollSpray(duration, btn, status) {
    let n = 0;
    const t = setInterval(async () => {
      n++;
      try {
        const data = await (await fetch('/api/spray-status')).json();
        if (!data.pending) {
          clearInterval(t);
          status.textContent = `GOTCHA! 🐿️💦 (${duration}s blast complete)`;
          status.className = 'ok';
          btn.disabled = false;
        } else if (n > 30) {
          clearInterval(t);
          status.textContent = 'Timed out — is detect.py running? 🤔';
          status.className = 'err';
          btn.disabled = false;
        }
      } catch { clearInterval(t); btn.disabled = false; }
    }, 500);
  }

  // ── Solenoid ─────────────────────────────────────────────────────────────
  let _solenoidOn = false;

  function updateSolenoidUI(on) {
    _solenoidOn = on;
    const btn  = document.getElementById('solenoid-btn');
    const dot  = document.getElementById('solenoid-dot');
    const hint = document.getElementById('solenoid-hint');
    btn.textContent = on ? '🔴 Turn Off' : '💧 Turn On';
    btn.classList.toggle('on', on);
    dot.classList.toggle('on', on);
    hint.textContent = on ? 'Valve is OPEN — water flowing! 🌊' : 'Holds valve open until manually turned off';
    document.getElementById('blast-btn').disabled = on;
  }

  async function toggleSolenoid() {
    const btn = document.getElementById('solenoid-btn');
    btn.disabled = true;
    try {
      const url  = _solenoidOn ? '/api/solenoid/off' : '/api/solenoid/on';
      const data = await (await fetch(url, { method: 'POST' })).json();
      updateSolenoidUI(data.on);
    } catch {}
    btn.disabled = false;
  }

  async function syncSolenoid() {
    try {
      const data = await (await fetch('/api/solenoid-status')).json();
      updateSolenoidUI(data.on);
    } catch {}
  }

  // ── Stats ────────────────────────────────────────────────────────────────
  function ago(isoStr) {
    const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000);
    if (diff < 60)   return diff + 's';
    if (diff < 3600) return Math.floor(diff / 60) + 'm';
    return Math.floor(diff / 3600) + 'h';
  }

  async function loadStats() {
    try {
      const n = new Date();
      const mins = n.getHours() * 60 + n.getMinutes() + 1;
      const data = await (await fetch('/api/detections?minutes=' + mins)).json();
      const squirrels = data.filter(d => d.class === 'squirrel');
      document.getElementById('squirrel-count').textContent = squirrels.length;
      const last = squirrels[0];
      document.getElementById('last-seen').textContent = last ? ago(last.timestamp) + ' ago' : 'None today';

      const scroll = document.getElementById('det-scroll');
      if (!data.length) {
        scroll.innerHTML = '<span class="det-empty">No detections today 🌤️</span>';
      } else {
        scroll.innerHTML = data.slice(0, 10).map(d => `
          <div class="det-chip">
            <span class="det-class">${d.class}</span>
            ${Math.round(d.confidence * 100)}% · ${ago(d.timestamp)} ago
          </div>`).join('');
      }
    } catch {}
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  syncSolenoid();
  loadStats();
  setInterval(syncSolenoid, 3000);
  setInterval(loadStats, 30000);
</script>
</body>
</html>"""


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SquirrelBGone</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #111; color: #eee; }

    /* ── Layout container ───────────────────────────────────────────────── */
    #page { max-width: 1400px; margin: 0 auto; }

    /* ── Header ─────────────────────────────────────────────────────────── */
    header {
      position: sticky; top: 0; z-index: 10;
      padding: 12px 16px; background: #1a1a1a;
      border-bottom: 1px solid #2a2a2a;
      display: flex; justify-content: space-between; align-items: center;
    }
    h1 { font-size: 1rem; font-weight: 600; }
    #meta { font-size: 0.75rem; color: #777; margin-top: 2px; }
    #refresh-btn {
      background: #2a2a2a; border: 1px solid #3a3a3a;
      color: #ccc; padding: 6px 14px; border-radius: 6px;
      font-size: 0.8rem; cursor: pointer; white-space: nowrap;
    }
    #refresh-btn:active { background: #333; }

    /* ── Controls ───────────────────────────────────────────────────────── */
    #controls {
      padding: 10px 12px; display: flex; flex-direction: column; gap: 8px;
      border-bottom: 1px solid #2a2a2a;
    }
    .ctrl-row { display: flex; gap: 6px; }
    .ctrl-btn {
      flex: 1; padding: 7px 0; border-radius: 6px;
      background: #1a1a1a; border: 1px solid #2a2a2a;
      color: #666; font-size: 0.8rem; cursor: pointer;
    }
    .ctrl-btn.active           { background: #2a2a2a; color: #eee;    border-color: #444; }
    .ctrl-btn.active.squirrel  { background: #451a03; color: #f59e0b; border-color: #7c3b0a; }
    .ctrl-btn.active.bird      { background: #0c1a40; color: #60a5fa; border-color: #1e3a8a; }
    .ctrl-btn.active.wildlife  { background: #052e16; color: #4ade80; border-color: #166534; }

    /* ── Summary pills ──────────────────────────────────────────────────── */
    #summary {
      display: flex; gap: 8px; padding: 10px 12px;
      font-size: 0.8rem;
    }
    .pill { background: #1e1e1e; border-radius: 20px; padding: 4px 12px; color: #aaa; }
    .pill.squirrel  { background: #451a03; color: #f59e0b; }
    .pill.bird      { background: #0c1a40; color: #60a5fa; }
    .pill.wildlife  { background: #052e16; color: #4ade80; }

    /* ── Cards grid ─────────────────────────────────────────────────────── */
    #cards {
      padding: 0 12px 24px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }

    .card {
      background: #1a1a1a; border-radius: 10px;
      overflow: hidden; border-left: 4px solid #333;
    }
    .card.squirrel { border-left-color: #f59e0b; }
    .card.bird     { border-left-color: #3b82f6; }
    .card.wildlife { border-left-color: #22c55e; }
    .card.flagged  { opacity: 0.45; }

    .img-wrap { position: relative; cursor: zoom-in; }
    .img-wrap img {
      width: 100%; display: block;
      max-height: 220px; object-fit: cover; background: #222;
    }
    .bbox-canvas {
      position: absolute; top: 0; left: 0;
      width: 100%; height: 100%; pointer-events: none;
    }

    .card-body {
      padding: 10px 12px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .cls { font-size: 0.9rem; font-weight: 600; text-transform: capitalize; }
    .cls.squirrel { color: #f59e0b; }
    .cls.bird     { color: #60a5fa; }
    .cls.wildlife { color: #4ade80; }
    .ts  { font-size: 0.7rem; color: #666; margin-top: 2px; }
    .right { display: flex; align-items: center; gap: 10px; }
    .conf { font-size: 1.2rem; font-weight: 700; }
    .flag-btn {
      background: none; border: 1px solid #333; border-radius: 6px;
      color: #555; padding: 5px 8px; font-size: 0.75rem; cursor: pointer;
    }
    .flag-btn:active  { color: #ef4444; border-color: #ef4444; }
    .flag-btn.flagged { color: #ef4444; border-color: #ef4444; cursor: default; }

    #empty { text-align: center; color: #555; padding: 60px 20px; font-size: 0.9rem; }

    /* ── Lightbox ───────────────────────────────────────────────────────── */
    #lightbox {
      display: none; position: fixed; inset: 0; z-index: 100;
      background: rgba(0,0,0,0.95); overflow: auto;
    }
    #lightbox.open { display: block; }
    #lb-wrap { position: relative; display: inline-block; min-width: 100%; min-height: 100%; }
    #lb-img { display: block; max-width: 100vw; }
    #lb-canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
    #lb-close {
      position: fixed; top: 12px; right: 12px; z-index: 101;
      background: rgba(0,0,0,0.7); color: #fff; border: none;
      border-radius: 50%; width: 36px; height: 36px;
      font-size: 1rem; cursor: pointer;
    }

    /* ── Desktop ────────────────────────────────────────────────────────── */
    @media (min-width: 700px) {
      #controls { flex-direction: row; }
      .ctrl-row { flex: 1; }

      #cards { grid-template-columns: 1fr 1fr; }
      .img-wrap img { max-height: none; }
    }

    @media (min-width: 1100px) {
      #cards { grid-template-columns: 1fr 1fr 1fr; }
    }

    /* ── Zone picker ────────────────────────────────────────────────────── */
    #zone-panel {
      margin: 12px; padding: 14px 16px;
      background: #1a1a1a; border-radius: 10px;
      border: 1px solid #2a2a2a;
    }
    #zone-panel h2 { font-size: 0.8rem; color: #777; margin-bottom: 6px; letter-spacing: 0.05em; text-transform: uppercase; }
    .zone-hint { font-size: 0.78rem; color: #555; margin-bottom: 10px; }
    .zone-wrap { position: relative; border-radius: 6px; overflow: hidden; width: 100%; }
    .zone-wrap img { display: block; width: 100%; background: #0a0a0a; min-height: 120px; }
    #zone-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
    .zone-controls { display: flex; align-items: center; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
    #zone-save-btn {
      padding: 8px 18px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
      background: #1a3a1a; border: 1px solid #2d6a2d; color: #4ade80; cursor: pointer;
    }
    #zone-save-btn:disabled { opacity: 0.4; cursor: default; }
    #zone-clear-btn {
      padding: 8px 18px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
      background: #3a1a1a; border: 1px solid #6a2d2d; color: #f87171; cursor: pointer;
    }
    #zone-status { font-size: 0.8rem; color: #555; }
    #zone-status.ok  { color: #4ade80; }
    #zone-status.err { color: #ef4444; }

    /* ── Hardware test panel ────────────────────────────────────────────── */
    #test-panel {
      margin: 12px; padding: 14px 16px;
      background: #1a1a1a; border-radius: 10px;
      border: 1px solid #2a2a2a;
    }
    #test-panel h2 { font-size: 0.8rem; color: #777; margin-bottom: 12px; letter-spacing: 0.05em; text-transform: uppercase; }
    .test-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    #spray-btn {
      padding: 10px 22px; border-radius: 8px; font-size: 0.9rem; font-weight: 600;
      background: #1a3a1a; border: 1px solid #2d6a2d; color: #4ade80; cursor: pointer;
    }
    #spray-btn:active { background: #2d6a2d; }
    #spray-btn:disabled { opacity: 0.4; cursor: default; }
    .dur-label { font-size: 0.8rem; color: #777; }
    #dur-input {
      width: 70px; padding: 6px 8px; border-radius: 6px;
      background: #2a2a2a; border: 1px solid #3a3a3a; color: #eee; font-size: 0.85rem;
    }
    #spray-status { font-size: 0.8rem; margin-top: 8px; color: #555; min-height: 1.2em; }
    #spray-status.ok      { color: #4ade80; }
    #spray-status.pending { color: #f59e0b; }
    #spray-status.err     { color: #ef4444; }
    .test-divider { border: none; border-top: 1px solid #2a2a2a; margin: 10px 0; }
    #toggle-btn {
      padding: 10px 22px; border-radius: 8px; font-size: 0.9rem; font-weight: 600;
      background: #1a1a2e; border: 1px solid #2d2d6a; color: #818cf8; cursor: pointer;
      min-width: 120px;
    }
    #toggle-btn.on  { background: #3a1a1a; border-color: #6a2d2d; color: #f87171; }
    #toggle-btn:disabled { opacity: 0.4; cursor: default; }
    .solenoid-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: #333; display: inline-block; margin-right: 6px; vertical-align: middle;
    }
    .solenoid-dot.on { background: #f87171; box-shadow: 0 0 6px #f87171; }
  </style>
</head>
<body>
<div id="page">
  <header>
    <div>
      <h1>SquirrelBGone</h1>
      <div id="meta">loading…</div>
    </div>
    <button id="refresh-btn" onclick="load()">Refresh</button>
  </header>

  <div id="controls">
    <div class="ctrl-row">
      <button class="ctrl-btn win-btn" data-w="15"    onclick="setWindow(15)">15m</button>
      <button class="ctrl-btn win-btn active" data-w="60" onclick="setWindow(60)">1h</button>
      <button class="ctrl-btn win-btn" data-w="today" onclick="setWindow('today')">Today</button>
    </div>
    <div class="ctrl-row">
      <button class="ctrl-btn cls-btn active"          data-f="all"      onclick="setFilter('all')">All</button>
      <button class="ctrl-btn cls-btn squirrel"        data-f="squirrel" onclick="setFilter('squirrel')">Squirrel</button>
      <button class="ctrl-btn cls-btn bird"            data-f="bird"     onclick="setFilter('bird')">Bird</button>
      <button class="ctrl-btn cls-btn wildlife"        data-f="wildlife" onclick="setFilter('wildlife')">Wildlife</button>
      <button class="ctrl-btn hc-btn"                  id="hc-btn"       onclick="toggleHighConf()">&#x2265;<span id="hc-label">70</span>%</button>
    </div>
  </div>

  <div id="summary"></div>

  <!-- Zone config -->
  <div id="zone-panel">
    <h2>Feeder Zone</h2>
    <p class="zone-hint">Click and drag on the live feed to define the feeder area. Only squirrels inside this zone will trigger the sprayer.</p>
    <div class="zone-wrap">
      <img id="zone-stream" src="/api/stream" alt="Live feed" onload="onZoneImgLoad()">
      <canvas id="zone-canvas"></canvas>
    </div>
    <div class="zone-controls">
      <button id="zone-save-btn" onclick="saveZone()" disabled>Save Zone</button>
      <button id="zone-clear-btn" onclick="clearZone()">Clear Zone</button>
      <span id="zone-status"></span>
    </div>
  </div>

  <div id="test-panel">
    <h2>Hardware Test</h2>
    <div class="test-row">
      <button id="spray-btn" onclick="fireSpray()">Fire Spray</button>
      <span class="dur-label">Duration (s)</span>
      <input id="dur-input" type="number" value="1.0" min="0.1" max="10" step="0.1">
    </div>
    <div id="spray-status">Queues a spray request — detect.py fires it on its next loop.</div>
    <hr class="test-divider">
    <div class="test-row">
      <button id="toggle-btn" onclick="toggleSolenoid()">
        <span class="solenoid-dot" id="solenoid-dot"></span>
        <span id="toggle-label">Turn On</span>
      </button>
      <span class="dur-label" id="toggle-hint">Hold open until turned off</span>
    </div>
  </div>

  <div id="cards"></div>

</div><!-- #page -->

  <div id="lightbox" onclick="handleLbClick(event)">
    <div id="lb-wrap">
      <img id="lb-img" onload="drawLbBox()">
      <canvas id="lb-canvas"></canvas>
    </div>
  </div>
  <button id="lb-close" style="display:none" onclick="closeLightbox()">✕</button>

  <script>
    const BOX_COLOR = { squirrel: '#f59e0b', bird: '#3b82f6' };
    function boxColor(cls) { return BOX_COLOR[cls] || '#fff'; }

    let currentWindow   = 60;
    let currentFilter   = 'all';
    let highConfOnly    = false;
    let highConfThresh  = 0.70;
    let allData = [];
    let lbData  = null;
    const flaggedSet = new Set();
    const detMap = {};

    const BIRD_CLASSES     = new Set(['bird','crow','pigeon','robin','sparrow']);
    const WILDLIFE_CLASSES = new Set([
      'deer','fawn','buck','doe',
      'fox','raccoon','rabbit','hog','boar',
      'bear','coyote','skunk','opossum','groundhog','turkey',
    ]);

    function windowMinutes() {
      if (currentWindow === 'today') {
        const n = new Date();
        return n.getHours() * 60 + n.getMinutes() + 1;
      }
      return currentWindow;
    }

    function setWindow(w) {
      currentWindow = w;
      document.querySelectorAll('.win-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.w === String(w))
      );
      load();
    }

    function setFilter(f) {
      currentFilter = f;
      document.querySelectorAll('.cls-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.f === f)
      );
      render();
    }

    function toggleHighConf() {
      highConfOnly = !highConfOnly;
      document.getElementById('hc-btn').classList.toggle('active', highConfOnly);
      render();
    }

    function cardClass(cls) {
      if (cls === 'squirrel') return 'squirrel';
      if (BIRD_CLASSES.has(cls)) return 'bird';
      if (WILDLIFE_CLASSES.has(cls)) return 'wildlife';
      return '';
    }

    function filtered() {
      let rows = allData;
      if (currentFilter === 'squirrel') rows = rows.filter(d => d.class === 'squirrel');
      else if (currentFilter === 'bird')     rows = rows.filter(d => BIRD_CLASSES.has(d.class));
      else if (currentFilter === 'wildlife') rows = rows.filter(d => WILDLIFE_CLASSES.has(d.class));
      if (highConfOnly) rows = rows.filter(d => parseFloat(d.confidence) >= highConfThresh);
      return rows;
    }

    function flagId(ts) { return 'flag-' + ts.replace(/:/g, '-'); }

    async function flagDetection(ts) {
      if (flaggedSet.has(ts)) return;
      const d = detMap[ts];
      if (!d) return;
      try {
        await fetch('/api/flag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            timestamp:  d.timestamp,
            class:      d.class,
            confidence: d.confidence,
            frame_path: d.frame_path || '',
          }),
        });
        flaggedSet.add(ts);
        const btn  = document.getElementById(flagId(ts));
        const card = btn && btn.closest('.card');
        if (btn)  { btn.classList.add('flagged'); btn.disabled = true; }
        if (card) card.classList.add('flagged');
      } catch (e) { console.error('Flag failed', e); }
    }

    // ── Bbox drawing ────────────────────────────────────────────────────────

    function drawBox(img) {
      const x1 = +img.dataset.x1, y1 = +img.dataset.y1;
      const w  = +img.dataset.w,  h  = +img.dataset.h;
      if (!w || !h) return;
      const canvas = img.nextElementSibling;
      canvas.width  = img.clientWidth;
      canvas.height = img.clientHeight;
      const sx = img.clientWidth  / img.naturalWidth;
      const sy = img.clientHeight / img.naturalHeight;
      const ctx = canvas.getContext('2d');
      // Draw feeder zone first (behind bbox)
      if (_feederZone) {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 2]);
        ctx.strokeRect(
          _feederZone.x1 * img.naturalWidth  * sx,
          _feederZone.y1 * img.naturalHeight * sy,
          (_feederZone.x2 - _feederZone.x1) * img.naturalWidth  * sx,
          (_feederZone.y2 - _feederZone.y1) * img.naturalHeight * sy,
        );
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = boxColor(img.dataset.cls);
      ctx.lineWidth = 2;
      ctx.strokeRect(x1 * sx, y1 * sy, w * sx, h * sy);
    }

    function drawLbBox() {
      if (!lbData || !lbData.w || !lbData.h) return;
      const img    = document.getElementById('lb-img');
      const canvas = document.getElementById('lb-canvas');
      canvas.width  = img.clientWidth;
      canvas.height = img.clientHeight;
      canvas.style.width  = img.clientWidth  + 'px';
      canvas.style.height = img.clientHeight + 'px';
      const sx = img.clientWidth  / img.naturalWidth;
      const sy = img.clientHeight / img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.strokeStyle = boxColor(lbData.cls);
      ctx.lineWidth = 4;
      ctx.strokeRect(lbData.x1 * sx, lbData.y1 * sy, lbData.w * sx, lbData.h * sy);
    }

    // ── Lightbox ─────────────────────────────────────────────────────────────

    function openLightbox(imgEl) {
      lbData = {
        src: imgEl.src,
        x1: +imgEl.dataset.x1, y1: +imgEl.dataset.y1,
        w:  +imgEl.dataset.w,  h:  +imgEl.dataset.h,
        cls: imgEl.dataset.cls,
      };
      const lb = document.getElementById('lightbox');
      document.getElementById('lb-img').src = lbData.src;
      lb.scrollTop = 0; lb.scrollLeft = 0;
      lb.classList.add('open');
      document.getElementById('lb-close').style.display = 'block';
    }

    function closeLightbox() {
      document.getElementById('lightbox').classList.remove('open');
      document.getElementById('lb-close').style.display = 'none';
      document.getElementById('lb-img').src = '';
      lbData = null;
    }

    function handleLbClick(e) {
      if (e.target.id === 'lightbox' || e.target.id === 'lb-wrap') closeLightbox();
    }

    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

    // ── Render ───────────────────────────────────────────────────────────────

    function ago(isoStr) {
      const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000);
      if (diff < 60)   return diff + 's ago';
      if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
      return Math.floor(diff / 3600) + 'h ago';
    }

    function render() {
      const data = filtered();
      const squirrels = allData.filter(d => d.class === 'squirrel').length;
      const birds     = allData.filter(d => BIRD_CLASSES.has(d.class)).length;
      const wildlife  = allData.filter(d => WILDLIFE_CLASSES.has(d.class)).length;
      const other     = allData.length - squirrels - birds - wildlife;

      const summary = document.getElementById('summary');
      summary.innerHTML =
        (squirrels ? `<span class="pill squirrel">${squirrels} squirrel${squirrels !== 1 ? 's' : ''}</span>` : '') +
        (birds     ? `<span class="pill bird">${birds} bird${birds !== 1 ? 's' : ''}</span>` : '') +
        (wildlife  ? `<span class="pill wildlife">${wildlife} wildlife</span>` : '') +
        (other     ? `<span class="pill">${other} other</span>` : '');

      const cards = document.getElementById('cards');
      if (!data.length) {
        cards.innerHTML = '<div id="empty">No detections in this window.</div>';
        return;
      }

      cards.innerHTML = data.map(d => {
        const cls      = (d.class || 'unknown').toLowerCase();
        const cc       = cardClass(cls);
        const conf     = Math.round(parseFloat(d.confidence) * 100);
        const isFlagged = flaggedSet.has(d.timestamp);
        const imgHtml  = d.image_url ? `
          <div class="img-wrap" onclick="openLightbox(this.querySelector('img'))">
            <img src="${d.image_url}" alt="${cls}" loading="lazy"
                 data-x1="${d.x1 || 0}" data-y1="${d.y1 || 0}"
                 data-w="${d.w || 0}"   data-h="${d.h || 0}"
                 data-cls="${cc}" onload="drawBox(this)">
            <canvas class="bbox-canvas"></canvas>
          </div>` : '';
        return `
          <div class="card ${cc}${isFlagged ? ' flagged' : ''}">
            ${imgHtml}
            <div class="card-body">
              <div>
                <div class="cls ${cc}">${cls}</div>
                <div class="ts">${ago(d.timestamp)} &middot; ${d.timestamp}</div>
              </div>
              <div class="right">
                <div class="conf">${conf}%</div>
                <button class="flag-btn${isFlagged ? ' flagged' : ''}"
                        id="${flagId(d.timestamp)}"
                        ${isFlagged ? 'disabled' : ''}
                        onclick="flagDetection('${d.timestamp}')">&#x1F6A9;</button>
              </div>
            </div>
          </div>`;
      }).join('');
    }

    async function load() {
      try {
        const res = await fetch('/api/detections?minutes=' + windowMinutes());
        allData = await res.json();
        allData.forEach(d => { detMap[d.timestamp] = d; });
      } catch (e) {
        document.getElementById('meta').textContent = 'Error loading detections.';
        return;
      }

      document.getElementById('meta').textContent =
        allData.length + ' detection' + (allData.length !== 1 ? 's' : '') +
        ' · ' + new Date().toLocaleTimeString();

      render();
    }

    async function init() {
      try {
        const cfg = await (await fetch('/api/config')).json();
        highConfThresh = cfg.high_conf_threshold;
        const pct = Math.round(highConfThresh * 100);
        document.getElementById('hc-label').textContent = pct;
      } catch (e) { /* use default */ }
      await loadFeederZone();
      initZonePicker();
      load();
    }

    init();
    setInterval(load, 30000);

    // ── Zone picker ──────────────────────────────────────────────────────────

    let _feederZone = null;
    let _zoneStart  = null;
    let _zoneDraft  = null;

    async function loadFeederZone() {
      try {
        const data = await (await fetch('/api/zone')).json();
        _feederZone = (data && data.x1 !== undefined) ? data : null;
      } catch {}
    }

    function onZoneImgLoad() {
      const canvas = document.getElementById('zone-canvas');
      const img    = document.getElementById('zone-stream');
      canvas.width  = img.clientWidth;
      canvas.height = img.clientHeight;
      drawZoneCanvas();
    }

    function drawZoneCanvas() {
      const canvas = document.getElementById('zone-canvas');
      const ctx    = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const zone = _zoneDraft || _feederZone;
      if (!zone) return;
      const x1 = zone.x1 * canvas.width,  y1 = zone.y1 * canvas.height;
      const x2 = zone.x2 * canvas.width,  y2 = zone.y2 * canvas.height;
      ctx.fillStyle   = 'rgba(245, 158, 11, 0.08)';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth   = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      ctx.fillStyle = '#f59e0b';
      ctx.font      = 'bold 11px system-ui';
      ctx.fillText('Feeder Zone', x1 + 5, y1 + 15);
    }

    function _canvasFrac(e, canvas) {
      const r = canvas.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (e.clientX - r.left)  / r.width)),
        y: Math.max(0, Math.min(1, (e.clientY - r.top)   / r.height)),
      };
    }

    function initZonePicker() {
      const canvas = document.getElementById('zone-canvas');

      canvas.addEventListener('mousedown', e => {
        _zoneStart = _canvasFrac(e, canvas);
        _zoneDraft = null;
        e.preventDefault();
      });

      canvas.addEventListener('mousemove', e => {
        if (!_zoneStart) return;
        const p = _canvasFrac(e, canvas);
        _zoneDraft = {
          x1: Math.min(_zoneStart.x, p.x), y1: Math.min(_zoneStart.y, p.y),
          x2: Math.max(_zoneStart.x, p.x), y2: Math.max(_zoneStart.y, p.y),
        };
        drawZoneCanvas();
        document.getElementById('zone-save-btn').disabled = false;
      });

      canvas.addEventListener('mouseup',    () => { _zoneStart = null; });
      canvas.addEventListener('mouseleave', () => { _zoneStart = null; });

      if (_feederZone) {
        document.getElementById('zone-status').textContent = 'Zone active';
        document.getElementById('zone-status').className  = 'ok';
      }
      drawZoneCanvas();
    }

    async function saveZone() {
      if (!_zoneDraft) return;
      const btn    = document.getElementById('zone-save-btn');
      const status = document.getElementById('zone-status');
      btn.disabled = true;
      try {
        const res  = await fetch('/api/zone', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(_zoneDraft),
        });
        const data = await res.json();
        if (data.ok) {
          _feederZone = _zoneDraft;
          _zoneDraft  = null;
          status.textContent = 'Zone saved ✓';
          status.className   = 'ok';
        } else {
          status.textContent = 'Save failed';
          status.className   = 'err';
        }
      } catch {
        status.textContent = 'Save failed';
        status.className   = 'err';
      }
    }

    async function clearZone() {
      try {
        await fetch('/api/zone', { method: 'DELETE' });
        _feederZone = null;
        _zoneDraft  = null;
        drawZoneCanvas();
        const status = document.getElementById('zone-status');
        status.textContent = 'Zone cleared';
        status.className   = '';
        document.getElementById('zone-save-btn').disabled = true;
      } catch {}
    }

    // ── Hardware test ────────────────────────────────────────────────────────

    async function fireSpray() {
      const duration = parseFloat(document.getElementById('dur-input').value) || 1.0;
      const btn    = document.getElementById('spray-btn');
      const status = document.getElementById('spray-status');
      btn.disabled = true;
      status.textContent = 'Queuing…';
      status.className = '';
      try {
        const res  = await fetch('/api/spray?duration=' + duration, { method: 'POST' });
        const data = await res.json();
        if (data.ok) {
          status.textContent = `Queued ${data.duration}s spray — waiting for detect.py…`;
          status.className = 'pending';
          pollSprayStatus(data.duration);
        } else {
          status.textContent = 'Error queuing request.';
          status.className = 'err';
          btn.disabled = false;
        }
      } catch (e) {
        status.textContent = 'Request failed.';
        status.className = 'err';
        btn.disabled = false;
      }
    }

    async function pollSprayStatus(duration) {
      const statusEl = document.getElementById('spray-status');
      let attempts = 0;
      const interval = setInterval(async () => {
        attempts++;
        try {
          const res  = await fetch('/api/spray-status');
          const data = await res.json();
          if (!data.pending) {
            clearInterval(interval);
            statusEl.textContent = `Fired ${duration}s pulse.`;
            statusEl.className = 'ok';
            document.getElementById('spray-btn').disabled = false;
          } else if (attempts > 20) {
            clearInterval(interval);
            statusEl.textContent = 'Timed out — is detect.py running?';
            statusEl.className = 'err';
            document.getElementById('spray-btn').disabled = false;
          }
        } catch (e) { clearInterval(interval); }
      }, 500);
    }

    let _solenoidOn = false;

    function updateToggleUI(on) {
      _solenoidOn = on;
      const btn   = document.getElementById('toggle-btn');
      const dot   = document.getElementById('solenoid-dot');
      const label = document.getElementById('toggle-label');
      const hint  = document.getElementById('toggle-hint');
      btn.classList.toggle('on', on);
      dot.classList.toggle('on', on);
      label.textContent = on ? 'Turn Off' : 'Turn On';
      hint.textContent  = on ? 'Solenoid is OPEN' : 'Hold open until turned off';
      document.getElementById('spray-btn').disabled = on;
    }

    async function toggleSolenoid() {
      const btn = document.getElementById('toggle-btn');
      btn.disabled = true;
      try {
        const endpoint = _solenoidOn ? '/api/solenoid/off' : '/api/solenoid/on';
        const res  = await fetch(endpoint, { method: 'POST' });
        const data = await res.json();
        updateToggleUI(data.on);
      } catch (e) { /* ignore */ }
      btn.disabled = false;
    }

    async function syncSolenoidStatus() {
      try {
        const res  = await fetch('/api/solenoid-status');
        const data = await res.json();
        updateToggleUI(data.on);
      } catch (e) { /* ignore */ }
    }

    syncSolenoidStatus();
    setInterval(syncSolenoidStatus, 3000);
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/mobile", response_class=HTMLResponse)
def mobile():
    return MOBILE_HTML
