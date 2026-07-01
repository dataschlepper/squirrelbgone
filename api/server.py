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
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

LOG_DIR    = Path(os.environ.get("LOG_DIR",    "logs"))
FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", "frames"))

# Minimum confidence logged by detect.py
LOG_THRESHOLD  = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
# Threshold above which a detection is considered a reliable true positive
HIGH_CONF_THRESHOLD = float(os.environ.get("HIGH_CONF_THRESHOLD", "0.70"))
# Confidence required to trigger spray (also used for "Squirrels Today" stat)
SPRAY_CONFIDENCE_THRESHOLD = float(os.environ.get("SPRAY_CONFIDENCE_THRESHOLD", "0.80"))

CORRECTIONS_PATH   = LOG_DIR / "corrections.csv"
CORRECTION_FIELDS  = ["flagged_at", "detection_timestamp", "class", "confidence", "frame_path"]
SPRAY_REQUEST_FILE  = LOG_DIR / "spray.request"
SOLENOID_STATE_FILE = LOG_DIR / "solenoid.state"
ZONE_FILE           = LOG_DIR / "feeder_zone.json"

FRAMES_KEEP_DAYS = int(os.environ.get("FRAMES_KEEP_DAYS", "7"))
LABELS_PATH      = LOG_DIR / "labels.csv"
LABEL_FIELDS     = ["labeled_at", "frame_path", "label", "predicted_class", "confidence"]

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


def _build_detection_index() -> dict:
    """Returns {frame_basename: best_detection_row} across the retention window."""
    index: dict = {}
    today = datetime.date.today()
    for offset in range(FRAMES_KEEP_DAYS + 1):
        csv_path = LOG_DIR / f"detections_{(today - datetime.timedelta(days=offset)).isoformat()}.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    name = Path(row.get("frame_path", "")).name
                    if not name:
                        continue
                    if name not in index:
                        index[name] = row
                    else:
                        # Prefer row whose class matches the filename suffix
                        suffix = name.rsplit("_", 1)[-1].replace(".jpg", "")
                        if row.get("class", "").lower() == suffix.lower():
                            index[name] = row
        except Exception:
            pass
    return index


def _read_labeled_set() -> set:
    """Returns set of frame basenames that have already been labeled."""
    labeled: set = set()
    if not LABELS_PATH.exists():
        return labeled
    try:
        with open(LABELS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                fp = row.get("frame_path", "")
                if fp:
                    labeled.add(Path(fp).name)
    except Exception:
        pass
    return labeled


def _date_from_filename(name: str):
    m = re.match(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else None


@app.get("/api/frames/dates")
def get_frame_dates():
    labeled = _read_labeled_set()
    counts: dict = {}
    for f in FRAMES_DIR.glob("*.jpg"):
        if f.name in labeled:
            continue
        d = _date_from_filename(f.name)
        if d:
            counts[d] = counts.get(d, 0) + 1
    dates = sorted(counts.keys(), reverse=True)
    return {"dates": [{"date": d, "pending": counts[d]} for d in dates]}


@app.get("/api/config")
def get_config():
    return {
        "log_threshold": LOG_THRESHOLD,
        "high_conf_threshold": HIGH_CONF_THRESHOLD,
        "spray_confidence_threshold": SPRAY_CONFIDENCE_THRESHOLD,
    }


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


@app.get("/api/frames/pending")
def get_pending_frames(date: str = Query(None)):
    labeled   = _read_labeled_set()
    det_idx   = _build_detection_index()
    all_files = sorted(FRAMES_DIR.glob("*.jpg"), reverse=True)
    if date:
        all_files = [f for f in all_files if f.name.startswith(date)]

    frames = []
    for f in all_files:
        if f.name in labeled:
            continue
        row = det_idx.get(f.name, {})
        frames.append({
            "frame_path":      f"frames/{f.name}",
            "image_url":       f"/frames/{f.name}",
            "filename":        f.name,
            "predicted_class": row.get("class", ""),
            "confidence":      row.get("confidence", ""),
            "x1": row.get("x1", ""), "y1": row.get("y1", ""),
            "w":  row.get("w",  ""), "h":  row.get("h",  ""),
            "timestamp":       row.get("timestamp", ""),
        })

    labeled_count = 0
    if LABELS_PATH.exists():
        try:
            with open(LABELS_PATH, newline="") as lf:
                for row in csv.DictReader(lf):
                    fp = row.get("frame_path", "")
                    if not fp:
                        continue
                    if date and date not in fp:
                        continue
                    labeled_count += 1
        except Exception:
            pass

    confs: list[float] = []
    class_dist: dict = {}
    for fr in frames:
        c = fr["confidence"]
        if c:
            try:
                confs.append(float(c))
            except ValueError:
                pass
        cls = (fr["predicted_class"] or "unknown").lower()
        class_dist[cls] = class_dist.get(cls, 0) + 1

    avg_conf = round(sum(confs) / len(confs), 3) if confs else None
    high_conf_count = sum(1 for c in confs if c >= HIGH_CONF_THRESHOLD)

    stats = {
        "pending_count":       len(frames),
        "labeled_count":       labeled_count,
        "class_distribution":  class_dist,
        "avg_confidence":      avg_conf,
        "high_conf_count":     high_conf_count,
        "high_conf_threshold": HIGH_CONF_THRESHOLD,
    }

    return {"count": len(frames), "frames": frames, "stats": stats}


@app.post("/api/frames/label")
async def label_frame(req: Request):
    body  = await req.json()
    fp    = body.get("frame_path", "").strip()
    label = body.get("label", "").strip()
    if not fp:
        return {"ok": False, "error": "frame_path required"}
    if label not in ("squirrel", "not_squirrel"):
        return {"ok": False, "error": "label must be 'squirrel' or 'not_squirrel'"}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not LABELS_PATH.exists()
    with open(LABELS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "labeled_at":      datetime.datetime.now().isoformat(timespec="seconds"),
            "frame_path":      fp,
            "label":           label,
            "predicted_class": body.get("predicted_class", ""),
            "confidence":      body.get("confidence", ""),
        })
    return {"ok": True}


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
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
      --yellow: #FFD166;
      --mint:   #06D6A0;
      --navy:   #023E8A;
      --glass:  rgba(255,255,255,0.85);
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
      padding: 14px 20px 28px;
      box-shadow: 0 4px 20px rgba(0,60,120,0.35);
      clip-path: ellipse(100% 100% at 50% 0%);
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .header-title { text-align: center; flex: 1; }
    h1 {
      font-family: 'Pacifico', cursive;
      font-size: clamp(1.6rem, 5vw, 2.4rem);
      color: var(--yellow);
      text-shadow: 3px 3px 0 rgba(0,0,0,0.25), 0 0 30px rgba(255,209,102,0.4);
    }
    .tagline {
      font-size: 0.68rem; color: var(--sky);
      font-weight: 900; text-transform: uppercase;
      letter-spacing: 3px; margin-top: 3px;
    }
    #meta {
      font-size: 0.68rem; color: rgba(144,224,239,0.7);
      font-weight: 700; margin-top: 3px;
    }
    #refresh-btn {
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.3);
      color: white; padding: 8px 14px; border-radius: 20px;
      font-family: 'Nunito', sans-serif;
      font-size: 0.78rem; font-weight: 700;
      cursor: pointer; white-space: nowrap;
      min-height: 38px; align-self: center;
    }
    #refresh-btn:active { background: rgba(255,255,255,0.28); }
    .nav-link {
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.3);
      color: rgba(255,255,255,0.9); padding: 8px 14px; border-radius: 20px;
      font-family: 'Nunito', sans-serif;
      font-size: 0.78rem; font-weight: 700;
      text-decoration: none; white-space: nowrap;
      min-height: 38px; align-self: center;
      display: inline-flex; align-items: center;
    }
    .nav-link:active { background: rgba(255,255,255,0.28); }

    /* ── Layout ──────────────────────────────────────────────────────────── */
    #layout {
      max-width: 1440px;
      margin: 0 auto;
      padding-bottom: 40px;
    }

    @media (min-width: 1100px) {
      #layout {
        display: grid;
        grid-template-columns: 390px 1fr;
        align-items: start;
      }
      #sidebar {
        position: sticky;
        top: 16px;
        max-height: calc(100vh - 32px);
        overflow-y: auto;
      }
    }

    /* ── Glass card ──────────────────────────────────────────────────────── */
    .card {
      background: var(--glass);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-radius: 22px;
      box-shadow: 0 4px 24px rgba(0,100,160,0.13), 0 1px 4px rgba(0,0,0,0.06);
      overflow: hidden;
      margin: 12px 14px;
    }
    .card-head {
      padding: 10px 16px 9px;
      display: flex; align-items: center; gap: 8px;
      border-bottom: 1px solid rgba(0,180,216,0.18);
      font-size: 0.7rem; font-weight: 900;
      text-transform: uppercase; letter-spacing: 2px; color: #888;
    }
    .card-body { padding: 16px; }

    /* ── Live stream + zone picker (combined) ────────────────────────────── */
    .live-dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: #22c55e; box-shadow: 0 0 7px #22c55e;
      animation: blink 1.4s ease-in-out infinite;
      flex-shrink: 0;
    }
    .live-label { color: #16a34a; letter-spacing: 2px; }

    #stream-wrap { position: relative; background: #0a0a1a; }
    #stream-img  { width: 100%; display: block; object-fit: contain; min-height: 180px; }
    #no-stream   {
      display: none; min-height: 180px;
      flex-direction: column; align-items: center; justify-content: center;
      background: #0a0a1a; color: #555; gap: 8px;
      font-size: 0.85rem; font-weight: 700;
    }
    #zone-canvas {
      position: absolute; top: 0; left: 0;
      width: 100%; height: 100%;
      cursor: default; pointer-events: none;
    }
    #zone-canvas.editing { cursor: crosshair; pointer-events: auto; }

    .zone-controls {
      padding: 10px 14px 12px;
      display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
      border-top: 1px solid rgba(0,180,216,0.18);
    }
    .zone-hint { font-size: 0.72rem; color: #888; font-weight: 700; font-style: italic; flex: 1; min-width: 80px; }
    #zone-save-btn, #zone-clear-btn, #zone-edit-btn, #zone-lock-btn, #zone-toggle-btn {
      padding: 6px 13px; border-radius: 12px; border: none;
      font-family: 'Nunito', sans-serif; font-size: 0.82rem; font-weight: 900;
      cursor: pointer; min-height: 34px;
    }
    #zone-save-btn  { background: linear-gradient(135deg, #15803d, #22c55e); color: white; }
    #zone-save-btn:disabled { opacity: 0.4; cursor: default; }
    #zone-clear-btn { background: linear-gradient(135deg, #9f1239, #f43f5e); color: white; }
    #zone-edit-btn  { background: linear-gradient(135deg, #0369a1, #38bdf8); color: white; }
    #zone-lock-btn  { background: linear-gradient(135deg, #78350f, #f59e0b); color: white; }
    #zone-toggle-btn { background: rgba(0,180,216,0.15); color: var(--navy); border: 1px solid rgba(0,180,216,0.3); }
    .zone-edit-group { display: none; align-items: center; gap: 8px; }
    .zone-edit-group.visible { display: flex; }
    #zone-status { font-size: 0.78rem; font-weight: 700; color: #aaa; }
    #zone-status.ok  { color: #16a34a; }
    #zone-status.err { color: #dc2626; }

    /* ── Stats ───────────────────────────────────────────────────────────── */
    .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .stat {
      background: linear-gradient(145deg, var(--deep), var(--pool));
      border-radius: 16px; padding: 14px 10px;
      text-align: center; color: white;
    }
    .stat-num {
      font-family: 'Pacifico', cursive;
      font-size: 2.2rem; color: var(--yellow);
      line-height: 1; text-shadow: 2px 2px 0 rgba(0,0,0,0.2);
    }
    .stat-label {
      font-size: 0.68rem; font-weight: 900;
      text-transform: uppercase; letter-spacing: 1px;
      opacity: 0.88; margin-top: 5px;
    }

    /* ── Blast button ────────────────────────────────────────────────────── */
    .blast-wrap { text-align: center; }
    .section-title {
      font-family: 'Pacifico', cursive;
      font-size: 1.05rem; color: var(--navy); margin-bottom: 16px;
    }
    #blast-btn {
      width: 160px; height: 160px; border-radius: 50%; border: none;
      background: linear-gradient(145deg, #FF6B6B, #FF006E);
      color: white; font-family: 'Nunito', sans-serif; font-weight: 900;
      cursor: pointer;
      box-shadow: 0 8px 0 #990042, 0 14px 28px rgba(255,0,110,0.45);
      transform: translateY(0);
      transition: transform 0.08s, box-shadow 0.08s;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 2px;
      margin: 0 auto 16px;
      position: relative; overflow: hidden;
      -webkit-tap-highlight-color: transparent;
    }
    #blast-btn .btn-emoji { font-size: 2.6rem; line-height: 1; }
    #blast-btn .btn-text  { font-size: 0.95rem; letter-spacing: 1px; }
    #blast-btn:active:not(:disabled) {
      transform: translateY(6px);
      box-shadow: 0 2px 0 #990042, 0 4px 12px rgba(255,0,110,0.35);
    }
    #blast-btn:disabled {
      background: linear-gradient(145deg, #bbb, #999);
      box-shadow: 0 6px 0 #666, 0 10px 18px rgba(0,0,0,0.18);
      cursor: not-allowed;
      animation: none;
    }
    #blast-btn:not(:disabled) { animation: idle-pulse 2.4s ease-in-out infinite; }
    @keyframes idle-pulse {
      0%, 100% { box-shadow: 0 8px 0 #990042, 0 14px 28px rgba(255,0,110,0.45); }
      50%       { box-shadow: 0 8px 0 #990042, 0 18px 42px rgba(255,0,110,0.7);  }
    }
    @keyframes drop-up {
      0%   { transform: translateY(0) scale(1);    opacity: 1; }
      100% { transform: translateY(-90px) scale(0.4); opacity: 0; }
    }
    .drop {
      position: absolute; pointer-events: none;
      font-size: 1.6rem; animation: drop-up 0.75s ease-out forwards;
    }
    .dur-row {
      display: flex; align-items: center; justify-content: center;
      gap: 10px; margin-bottom: 12px;
    }
    .dur-label { font-weight: 900; color: var(--navy); font-size: 0.85rem; }
    #dur-input {
      width: 76px; padding: 8px 10px;
      border: 3px solid var(--pool); border-radius: 12px;
      font-family: 'Nunito', sans-serif; font-size: 0.95rem; font-weight: 700;
      color: var(--navy); text-align: center; background: white; outline: none;
    }
    #dur-input:focus { border-color: var(--deep); }
    #blast-status {
      min-height: 1.5em; font-weight: 700; font-size: 0.85rem; color: #888;
    }
    #blast-status.wait { color: var(--pool); }
    #blast-status.ok   { color: #16a34a; }
    #blast-status.err  { color: #dc2626; }

    /* ── Solenoid ────────────────────────────────────────────────────────── */
    .solenoid-row {
      display: flex; align-items: center; justify-content: center; gap: 14px;
    }
    .solenoid-dot {
      width: 13px; height: 13px; border-radius: 50%;
      background: #ccc; flex-shrink: 0;
      transition: background 0.3s, box-shadow 0.3s;
    }
    .solenoid-dot.on { background: #22c55e; box-shadow: 0 0 10px #22c55e; }
    #solenoid-btn {
      padding: 11px 26px; border-radius: 14px;
      border: 3px solid var(--deep);
      background: white; color: var(--deep);
      font-family: 'Nunito', sans-serif; font-size: 0.92rem; font-weight: 900;
      cursor: pointer; transition: all 0.2s; min-height: 46px;
    }
    #solenoid-btn.on {
      background: linear-gradient(135deg, #15803d, #22c55e);
      border-color: #15803d; color: white;
    }
    #solenoid-btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .solenoid-hint {
      font-size: 0.73rem; color: #888; margin-top: 10px;
      font-style: italic; font-weight: 700; text-align: center;
    }

    /* ── Detection controls ──────────────────────────────────────────────── */
    .ctrl-group { display: flex; gap: 6px; margin-bottom: 8px; }
    .ctrl-group:last-child { margin-bottom: 0; }
    .ctrl-btn {
      flex: 1; padding: 9px 6px; border-radius: 12px;
      border: 2px solid rgba(0,119,182,0.2);
      background: rgba(255,255,255,0.55);
      color: #777; font-family: 'Nunito', sans-serif;
      font-size: 0.8rem; font-weight: 900;
      cursor: pointer; transition: all 0.15s; min-height: 44px;
    }
    .ctrl-btn.active           { background: var(--deep); color: white; border-color: var(--deep); }
    .ctrl-btn.active.squirrel  { background: #d97706; border-color: #d97706; }
    .ctrl-btn.active.bird      { background: #2563eb; border-color: #2563eb; }
    .ctrl-btn.active.wildlife  { background: #16a34a; border-color: #16a34a; }

    /* ── Summary pills ───────────────────────────────────────────────────── */
    #summary {
      display: flex; gap: 8px; padding: 4px 14px 8px;
      flex-wrap: wrap; font-size: 0.8rem;
    }
    .pill {
      border-radius: 20px; padding: 5px 14px;
      font-weight: 900; font-size: 0.78rem;
    }
    .pill.default  { background: rgba(0,119,182,0.12); color: var(--deep); }
    .pill.squirrel { background: #fef3c7; color: #d97706; }
    .pill.bird     { background: #dbeafe; color: #2563eb; }
    .pill.wildlife { background: #dcfce7; color: #16a34a; }

    /* ── Detection cards grid ────────────────────────────────────────────── */
    #cards {
      padding: 4px 14px 8px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    @media (min-width: 600px) and (max-width: 1099px) { #cards { grid-template-columns: 1fr 1fr; } }
    @media (min-width: 1100px)                         { #cards { grid-template-columns: 1fr 1fr; } }
    @media (min-width: 1440px)                         { #cards { grid-template-columns: 1fr 1fr 1fr; } }

    .det-card {
      background: var(--glass);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 4px 16px rgba(0,100,160,0.10);
      border-left: 4px solid rgba(0,119,182,0.18);
    }
    .det-card.squirrel { border-left-color: #f59e0b; }
    .det-card.bird     { border-left-color: #3b82f6; }
    .det-card.wildlife { border-left-color: #22c55e; }
    .det-card.flagged  { opacity: 0.42; }

    .img-wrap { position: relative; cursor: zoom-in; }
    .img-wrap img { width: 100%; display: block; object-fit: cover; background: #1a1a2e; }
    .bbox-canvas  { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }

    .det-card-body {
      padding: 11px 14px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .cls { font-size: 0.9rem; font-weight: 900; text-transform: capitalize; }
    .cls.squirrel { color: #d97706; }
    .cls.bird     { color: #2563eb; }
    .cls.wildlife { color: #16a34a; }
    .ts  { font-size: 0.66rem; color: #888; margin-top: 2px; font-weight: 700; }
    .card-right { display: flex; align-items: center; gap: 10px; }
    .conf { font-family: 'Pacifico', cursive; font-size: 1.25rem; color: var(--deep); }
    .flag-btn {
      background: none; border: 2px solid rgba(0,119,182,0.18);
      border-radius: 10px; color: #bbb; padding: 8px;
      font-size: 0.78rem; cursor: pointer; min-height: 44px; min-width: 44px;
    }
    .flag-btn:active  { color: #ef4444; border-color: #ef4444; }
    .flag-btn.flagged { color: #ef4444; border-color: #ef4444; cursor: default; }

    #empty {
      grid-column: 1 / -1; text-align: center;
      color: #888; padding: 60px 20px; font-size: 0.95rem; font-weight: 700;
    }

    /* ── Lightbox ────────────────────────────────────────────────────────── */
    #lightbox {
      display: none; position: fixed; inset: 0; z-index: 100;
      background: rgba(0,0,0,0.95); overflow: auto;
    }
    #lightbox.open { display: block; }
    #lb-wrap { position: relative; display: inline-block; min-width: 100%; min-height: 100%; }
    #lb-img  { display: block; max-width: 100vw; }
    #lb-canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
    #lb-close {
      position: fixed; top: 12px; right: 12px; z-index: 101;
      background: rgba(0,0,0,0.7); color: #fff; border: none;
      border-radius: 50%; width: 44px; height: 44px;
      font-size: 1.1rem; cursor: pointer;
    }

    /* ── Animations ──────────────────────────────────────────────────────── */
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.3; }
    }
  </style>
</head>
<body>

<header>
  <a href="/review" class="nav-link">🏷️ Label</a>
  <div class="header-title">
    <h1>🐿️ SquirrelBGone 💦</h1>
    <p class="tagline">Spray First · Ask Questions Never</p>
    <p id="meta"></p>
  </div>
  <button id="refresh-btn" onclick="load()">↺ Refresh</button>
</header>

<div id="layout">

  <!-- ── Sidebar ────────────────────────────────────────────────────────── -->
  <div id="sidebar">

    <!-- Live stream + zone picker -->
    <div class="card">
      <div class="card-head">
        <div class="live-dot"></div>
        <span class="live-label">LIVE</span>
        <span style="flex:1"></span>
        <button id="zone-toggle-btn" onclick="toggleZoneVisibility()" style="margin-right:4px">Hide Zone</button>
        <button id="zone-edit-btn" onclick="toggleZoneEdit()">Edit Zone</button>
        <button id="zone-lock-btn" onclick="toggleZoneEdit()" style="display:none">Lock Zone</button>
      </div>
      <div id="stream-wrap">
        <img id="stream-img" src="/api/stream" alt="Live feed"
             onerror="showNoStream()" onload="onStreamLoad()">
        <canvas id="zone-canvas"></canvas>
      </div>
      <div id="no-stream">
        <span style="font-size:2.5rem">📷</span>
        <span>Camera offline</span>
      </div>
      <div class="zone-controls">
        <span class="zone-hint" id="zone-hint">Zone locked — click Edit Zone to change</span>
        <div class="zone-edit-group" id="zone-edit-group">
          <button id="zone-save-btn" onclick="saveZone()" disabled>Save Zone</button>
          <button id="zone-clear-btn" onclick="clearZone()">Clear</button>
        </div>
        <span id="zone-status"></span>
      </div>
    </div>

    <!-- Stats -->
    <div class="card">
      <div class="card-body">
        <div class="stats-grid">
          <div class="stat">
            <div class="stat-num" id="squirrel-count">—</div>
            <div class="stat-label">🐿️ Squirrels Today <span id="squirrel-conf-label" style="opacity:0.7;font-size:0.6rem"></span></div>
          </div>
          <div class="stat">
            <div class="stat-num" id="last-seen" style="font-size:1.25rem">—</div>
            <div class="stat-label">⏱️ Last Spotted</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Blast button -->
    <div class="card">
      <div class="card-body blast-wrap">
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
      <div class="card-head" style="justify-content:center; border-bottom:none; padding-bottom:4px">
        🚰 Solenoid — Hold Open
      </div>
      <div class="card-body" style="text-align:center; padding-top:10px">
        <div class="solenoid-row">
          <div class="solenoid-dot" id="solenoid-dot"></div>
          <button id="solenoid-btn" onclick="toggleSolenoid()">💧 Turn On</button>
        </div>
        <div class="solenoid-hint" id="solenoid-hint">Holds valve open until manually turned off</div>
      </div>
    </div>

  </div><!-- #sidebar -->

  <!-- ── Main ──────────────────────────────────────────────────────────── -->
  <div id="main">

    <!-- Controls -->
    <div class="card">
      <div class="card-body">
        <div class="ctrl-group">
          <button class="ctrl-btn win-btn" data-w="15"    onclick="setWindow(15)">15m</button>
          <button class="ctrl-btn win-btn active" data-w="60" onclick="setWindow(60)">1h</button>
          <button class="ctrl-btn win-btn" data-w="today" onclick="setWindow('today')">Today</button>
        </div>
        <div class="ctrl-group">
          <button class="ctrl-btn cls-btn active"   data-f="all"      onclick="setFilter('all')">All</button>
          <button class="ctrl-btn cls-btn squirrel" data-f="squirrel" onclick="setFilter('squirrel')">🐿️ Squirrel</button>
          <button class="ctrl-btn cls-btn bird"     data-f="bird"     onclick="setFilter('bird')">🐦 Bird</button>
          <button class="ctrl-btn cls-btn wildlife" data-f="wildlife" onclick="setFilter('wildlife')">🦌 Wildlife</button>
          <button class="ctrl-btn hc-btn" id="hc-btn" onclick="toggleHighConf()">&#x2265;<span id="hc-label">70</span>%</button>
        </div>
      </div>
    </div>

    <div id="summary"></div>
    <div id="cards"></div>

  </div><!-- #main -->

</div><!-- #layout -->

<div id="lightbox" onclick="handleLbClick(event)">
  <div id="lb-wrap">
    <img id="lb-img" onload="drawLbBox()">
    <canvas id="lb-canvas"></canvas>
  </div>
</div>
<button id="lb-close" style="display:none" onclick="closeLightbox()">✕</button>

<script>
  // ── Stream ───────────────────────────────────────────────────────────────
  function showNoStream() {
    document.getElementById('stream-wrap').style.display = 'none';
    document.getElementById('no-stream').style.display  = 'flex';
  }
  function onStreamLoad() {
    document.getElementById('stream-wrap').style.display = '';
    document.getElementById('no-stream').style.display  = 'none';
    const img    = document.getElementById('stream-img');
    const canvas = document.getElementById('zone-canvas');
    canvas.width  = img.clientWidth;
    canvas.height = img.clientHeight;
    drawZoneCanvas();
  }

  // ── Zone picker ──────────────────────────────────────────────────────────
  let _feederZone = null, _zoneStart = null, _zoneDraft = null;
  let _zoneEditing = false;
  let _showZone    = true;

  async function loadFeederZone() {
    try {
      const data = await (await fetch('/api/zone')).json();
      _feederZone = (data && data.x1 !== undefined) ? data : null;
    } catch {}
  }

  function drawZoneCanvas() {
    const canvas = document.getElementById('zone-canvas');
    const ctx    = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!_showZone && !_zoneEditing) return;
    const zone = _zoneDraft || _feederZone;
    if (!zone) return;
    const x1 = zone.x1 * canvas.width,  y1 = zone.y1 * canvas.height;
    const x2 = zone.x2 * canvas.width,  y2 = zone.y2 * canvas.height;
    ctx.fillStyle   = 'rgba(255,209,102,0.1)';
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.strokeStyle = '#FFD166';
    ctx.lineWidth   = 2.5;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    ctx.fillStyle = '#FFD166';
    ctx.font      = 'bold 11px system-ui';
    ctx.fillText('Feeder Zone', x1 + 5, y1 + 15);
  }

  function _frac(clientX, clientY, canvas) {
    const r = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (clientX - r.left) / r.width)),
      y: Math.max(0, Math.min(1, (clientY - r.top)  / r.height)),
    };
  }

  function initZonePicker() {
    const canvas = document.getElementById('zone-canvas');

    canvas.addEventListener('mousedown', e => {
      if (!_zoneEditing) return;
      _zoneStart = _frac(e.clientX, e.clientY, canvas);
      _zoneDraft = null; e.preventDefault();
    });
    canvas.addEventListener('mousemove', e => {
      if (!_zoneEditing || !_zoneStart) return;
      const p = _frac(e.clientX, e.clientY, canvas);
      _zoneDraft = {
        x1: Math.min(_zoneStart.x, p.x), y1: Math.min(_zoneStart.y, p.y),
        x2: Math.max(_zoneStart.x, p.x), y2: Math.max(_zoneStart.y, p.y),
      };
      drawZoneCanvas();
      document.getElementById('zone-save-btn').disabled = false;
    });
    canvas.addEventListener('mouseup',    () => { _zoneStart = null; });
    canvas.addEventListener('mouseleave', () => { _zoneStart = null; });

    canvas.addEventListener('touchstart', e => {
      if (!_zoneEditing) return;
      const t = e.touches[0];
      _zoneStart = _frac(t.clientX, t.clientY, canvas);
      _zoneDraft = null; e.preventDefault();
    }, { passive: false });
    canvas.addEventListener('touchmove', e => {
      if (!_zoneEditing || !_zoneStart) return;
      const t = e.touches[0];
      const p = _frac(t.clientX, t.clientY, canvas);
      _zoneDraft = {
        x1: Math.min(_zoneStart.x, p.x), y1: Math.min(_zoneStart.y, p.y),
        x2: Math.max(_zoneStart.x, p.x), y2: Math.max(_zoneStart.y, p.y),
      };
      drawZoneCanvas();
      document.getElementById('zone-save-btn').disabled = false;
      e.preventDefault();
    }, { passive: false });
    canvas.addEventListener('touchend', () => { _zoneStart = null; });

    if (_feederZone) {
      document.getElementById('zone-status').textContent = 'Zone active ✓';
      document.getElementById('zone-status').className  = 'ok';
    }
    drawZoneCanvas();
  }

  function toggleZoneEdit() {
    _zoneEditing = !_zoneEditing;
    const canvas   = document.getElementById('zone-canvas');
    const editBtn  = document.getElementById('zone-edit-btn');
    const lockBtn  = document.getElementById('zone-lock-btn');
    const editGrp  = document.getElementById('zone-edit-group');
    const hint     = document.getElementById('zone-hint');
    if (_zoneEditing) {
      canvas.classList.add('editing');
      editBtn.style.display = 'none';
      lockBtn.style.display = '';
      editGrp.classList.add('visible');
      hint.textContent = 'Drag on the image to define the feeder area';
    } else {
      _zoneStart = null; _zoneDraft = null;
      canvas.classList.remove('editing');
      editBtn.style.display = '';
      lockBtn.style.display = 'none';
      editGrp.classList.remove('visible');
      hint.textContent = 'Zone locked — click Edit Zone to change';
      document.getElementById('zone-save-btn').disabled = true;
      drawZoneCanvas();
    }
  }

  function toggleZoneVisibility() {
    _showZone = !_showZone;
    document.getElementById('zone-toggle-btn').textContent = _showZone ? 'Hide Zone' : 'Show Zone';
    drawZoneCanvas();
    document.querySelectorAll('.det-card img[data-x1]').forEach(img => { if (img.complete) drawBox(img); });
  }

  async function saveZone() {
    if (!_zoneDraft) return;
    const btn = document.getElementById('zone-save-btn');
    const st  = document.getElementById('zone-status');
    btn.disabled = true;
    try {
      const res  = await fetch('/api/zone', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(_zoneDraft),
      });
      const data = await res.json();
      if (data.ok) {
        _feederZone = _zoneDraft; _zoneDraft = null;
        st.textContent = 'Zone saved ✓'; st.className = 'ok';
      } else {
        st.textContent = 'Save failed'; st.className = 'err';
      }
    } catch { st.textContent = 'Save failed'; st.className = 'err'; }
  }

  async function clearZone() {
    try {
      await fetch('/api/zone', { method: 'DELETE' });
      _feederZone = null; _zoneDraft = null; drawZoneCanvas();
      const st = document.getElementById('zone-status');
      st.textContent = 'Zone cleared'; st.className = '';
      document.getElementById('zone-save-btn').disabled = true;
    } catch {}
  }

  // ── Detection state ──────────────────────────────────────────────────────
  const BOX_COLOR = { squirrel: '#f59e0b', bird: '#3b82f6' };
  function boxColor(cls) { return BOX_COLOR[cls] || '#fff'; }

  let currentWindow  = 60;
  let currentFilter  = 'all';
  let highConfOnly   = false;
  let highConfThresh  = 0.70;
  let sprayConfThresh = 0.80;
  let allData = [];
  let lbData  = null;
  const flaggedSet = new Set();
  const detMap = {};

  const BIRD_CLASSES     = new Set(['bird','crow','pigeon','robin','sparrow']);
  const WILDLIFE_CLASSES = new Set([
    'deer','fawn','buck','doe','fox','raccoon','rabbit','hog','boar',
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
    if (cls === 'squirrel')        return 'squirrel';
    if (BIRD_CLASSES.has(cls))     return 'bird';
    if (WILDLIFE_CLASSES.has(cls)) return 'wildlife';
    return '';
  }

  function filtered() {
    let rows = allData;
    if (currentFilter === 'squirrel')     rows = rows.filter(d => d.class === 'squirrel');
    else if (currentFilter === 'bird')     rows = rows.filter(d => BIRD_CLASSES.has(d.class));
    else if (currentFilter === 'wildlife') rows = rows.filter(d => WILDLIFE_CLASSES.has(d.class));
    if (highConfOnly) rows = rows.filter(d => parseFloat(d.confidence) >= highConfThresh);
    return rows;
  }

  function ago(isoStr) {
    const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000);
    if (diff < 60)   return diff + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    return Math.floor(diff / 3600) + 'h ago';
  }

  // ── Stats (always full-day, independent of window filter) ────────────────
  async function updateStats() {
    try {
      const n    = new Date();
      const mins = n.getHours() * 60 + n.getMinutes() + 1;
      const data = await (await fetch('/api/detections?minutes=' + mins)).json();
      const squirrels = data.filter(d => d.class === 'squirrel' && parseFloat(d.confidence) >= sprayConfThresh);
      document.getElementById('squirrel-count').textContent = squirrels.length || '0';
      const last = squirrels[0];
      document.getElementById('last-seen').textContent = last ? ago(last.timestamp) : 'None today';
    } catch {}
  }

  // ── Flag ─────────────────────────────────────────────────────────────────
  function flagId(ts) { return 'flag-' + ts.replace(/[:.]/g, '-'); }

  async function flagDetection(ts) {
    if (flaggedSet.has(ts)) return;
    const d = detMap[ts];
    if (!d) return;
    try {
      await fetch('/api/flag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamp: d.timestamp, class: d.class,
          confidence: d.confidence, frame_path: d.frame_path || '',
        }),
      });
      flaggedSet.add(ts);
      const btn  = document.getElementById(flagId(ts));
      const card = btn?.closest('.det-card');
      if (btn)  { btn.classList.add('flagged'); btn.disabled = true; }
      if (card) card.classList.add('flagged');
    } catch {}
  }

  // ── Bbox ─────────────────────────────────────────────────────────────────
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
    if (_feederZone && _showZone) {
      ctx.strokeStyle = 'rgba(255,209,102,0.5)';
      ctx.lineWidth = 1; ctx.setLineDash([4, 2]);
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
    if (!lbData?.w || !lbData?.h) return;
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

  // ── Lightbox ──────────────────────────────────────────────────────────────
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

  // ── Render ────────────────────────────────────────────────────────────────
  function render() {
    const data      = filtered();
    const squirrels = allData.filter(d => d.class === 'squirrel').length;
    const birds     = allData.filter(d => BIRD_CLASSES.has(d.class)).length;
    const wildlife  = allData.filter(d => WILDLIFE_CLASSES.has(d.class)).length;
    const other     = allData.length - squirrels - birds - wildlife;

    document.getElementById('summary').innerHTML =
      (squirrels ? `<span class="pill squirrel">🐿️ ${squirrels}</span>` : '') +
      (birds     ? `<span class="pill bird">🐦 ${birds}</span>`         : '') +
      (wildlife  ? `<span class="pill wildlife">🦌 ${wildlife}</span>`  : '') +
      (other     ? `<span class="pill default">${other} other</span>`   : '');

    const cards = document.getElementById('cards');
    if (!data.length) {
      cards.innerHTML = '<div id="empty">No detections in this window 🌤️</div>';
      return;
    }

    cards.innerHTML = data.map(d => {
      const cls       = (d.class || 'unknown').toLowerCase();
      const cc        = cardClass(cls);
      const conf      = Math.round(parseFloat(d.confidence) * 100);
      const isFlagged = flaggedSet.has(d.timestamp);
      const imgHtml   = d.image_url ? `
        <div class="img-wrap" onclick="openLightbox(this.querySelector('img'))">
          <img src="${d.image_url}" alt="${cls}" loading="lazy"
               data-x1="${d.x1 || 0}" data-y1="${d.y1 || 0}"
               data-w="${d.w || 0}"   data-h="${d.h || 0}"
               data-cls="${cc}" onload="drawBox(this)">
          <canvas class="bbox-canvas"></canvas>
        </div>` : '';
      return `
        <div class="det-card ${cc}${isFlagged ? ' flagged' : ''}">
          ${imgHtml}
          <div class="det-card-body">
            <div>
              <div class="cls ${cc}">${cls}</div>
              <div class="ts">${ago(d.timestamp)} &middot; ${d.timestamp}</div>
            </div>
            <div class="card-right">
              <div class="conf">${conf}%</div>
              <button class="flag-btn${isFlagged ? ' flagged' : ''}"
                      id="${flagId(d.timestamp)}"
                      ${isFlagged ? 'disabled' : ''}
                      onclick="flagDetection('${d.timestamp}')">🚩</button>
            </div>
          </div>
        </div>`;
    }).join('');
  }

  async function load() {
    try {
      allData = await (await fetch('/api/detections?minutes=' + windowMinutes())).json();
      allData.forEach(d => { detMap[d.timestamp] = d; });
    } catch {
      document.getElementById('meta').textContent = 'Error loading detections.';
      return;
    }
    document.getElementById('meta').textContent =
      allData.length + ' detection' + (allData.length !== 1 ? 's' : '') +
      ' · ' + new Date().toLocaleTimeString();
    render();
  }

  // ── Spray ─────────────────────────────────────────────────────────────────
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

  async function fireSpray() {
    const duration = parseFloat(document.getElementById('dur-input').value) || 1.0;
    const btn    = document.getElementById('blast-btn');
    const status = document.getElementById('blast-status');
    btn.disabled = true;
    spawnDrops(btn);
    status.textContent = 'Firing! 💦💦💦'; status.className = 'wait';
    try {
      const data = await (await fetch('/api/spray?duration=' + duration, { method: 'POST' })).json();
      if (data.ok) {
        status.textContent = `${data.duration}s spray queued…`; status.className = 'wait';
        pollSpray(data.duration, btn, status);
      } else {
        status.textContent = 'Something went wrong 😬'; status.className = 'err';
        btn.disabled = false;
      }
    } catch {
      status.textContent = 'Connection error 📡'; status.className = 'err';
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
          status.className = 'ok'; btn.disabled = false;
        } else if (n > 30) {
          clearInterval(t);
          status.textContent = 'Timed out — is detect.py running? 🤔';
          status.className = 'err'; btn.disabled = false;
        }
      } catch { clearInterval(t); btn.disabled = false; }
    }, 500);
  }

  // ── Solenoid ──────────────────────────────────────────────────────────────
  let _solenoidOn = false;

  function updateSolenoidUI(on) {
    _solenoidOn = on;
    const btn  = document.getElementById('solenoid-btn');
    const dot  = document.getElementById('solenoid-dot');
    const hint = document.getElementById('solenoid-hint');
    btn.textContent = on ? '🔴 Turn Off' : '💧 Turn On';
    btn.classList.toggle('on', on);
    dot.classList.toggle('on', on);
    hint.textContent = on
      ? 'Valve is OPEN — water flowing! 🌊'
      : 'Holds valve open until manually turned off';
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

  // ── Init ──────────────────────────────────────────────────────────────────
  async function init() {
    try {
      const cfg = await (await fetch('/api/config')).json();
      highConfThresh  = cfg.high_conf_threshold;
      sprayConfThresh = cfg.spray_confidence_threshold ?? 0.80;
      document.getElementById('hc-label').textContent = Math.round(highConfThresh * 100);
      document.getElementById('squirrel-conf-label').textContent = '≥' + Math.round(sprayConfThresh * 100) + '%';
    } catch {}
    await loadFeederZone();
    initZonePicker();
    syncSolenoid();
    updateStats();
    load();
  }

  init();
  setInterval(load,        30000);
  setInterval(updateStats, 30000);
  setInterval(syncSolenoid, 3000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/mobile", response_class=HTMLResponse)
def mobile():
    return RedirectResponse(url="/")


REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
  <title>Label Frames — SquirrelBGone</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Pacifico&family=Nunito:wght@400;700;900&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --pool:   #00B4D8;
      --sky:    #90E0EF;
      --deep:   #0077B6;
      --pink:   #FF006E;
      --yellow: #FFD166;
      --mint:   #06D6A0;
      --navy:   #023E8A;
      --glass:  rgba(255,255,255,0.85);
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
      padding: 14px 20px 28px;
      box-shadow: 0 4px 20px rgba(0,60,120,0.35);
      clip-path: ellipse(100% 100% at 50% 0%);
      display: flex; align-items: flex-start;
      justify-content: space-between; gap: 12px;
    }
    .header-title { text-align: center; flex: 1; }
    h1 {
      font-family: 'Pacifico', cursive;
      font-size: clamp(1.4rem, 4vw, 2rem);
      color: var(--yellow);
      text-shadow: 3px 3px 0 rgba(0,0,0,0.25), 0 0 30px rgba(255,209,102,0.4);
    }
    .tagline { font-size: 0.68rem; color: var(--sky); font-weight: 900; text-transform: uppercase; letter-spacing: 3px; margin-top: 3px; }
    #counter { font-size: 0.75rem; color: var(--sky); font-weight: 700; margin-top: 4px; }
    .nav-back {
      background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
      color: rgba(255,255,255,0.9); padding: 8px 14px; border-radius: 20px;
      font-family: 'Nunito', sans-serif; font-size: 0.78rem; font-weight: 700;
      text-decoration: none; white-space: nowrap;
      min-height: 38px; align-self: center; display: inline-flex; align-items: center;
    }
    .nav-back:active { background: rgba(255,255,255,0.28); }

    /* ── Controls bar ───────────────────────────────────────────────────── */
    .controls-bar {
      max-width: 1200px; margin: 0 auto;
      padding: 16px 14px 0;
      display: flex; flex-direction: column; gap: 10px;
    }

    /* ── Metrics bar ────────────────────────────────────────────────────── */
    .metrics-bar {
      background: var(--glass);
      backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      border-radius: 16px; padding: 14px 18px;
      box-shadow: 0 2px 12px rgba(0,100,160,0.10);
      display: flex; flex-direction: column; gap: 8px;
    }
    .metrics-progress-row { display: flex; align-items: center; gap: 10px; }
    .progress-bar-wrap {
      flex: 1; height: 8px; background: rgba(0,119,182,0.15); border-radius: 4px; overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%; border-radius: 4px;
      background: linear-gradient(90deg, #06D6A0, #00B4D8);
      transition: width 0.4s ease;
    }
    .progress-label { font-size: 0.72rem; font-weight: 900; color: var(--deep); white-space: nowrap; }
    .metrics-pills { display: flex; flex-wrap: wrap; gap: 6px; min-height: 22px; }
    .metrics-pill { font-size: 0.72rem; font-weight: 900; padding: 3px 10px; border-radius: 20px; }
    .pill-squirrel { background: rgba(245,158,11,0.18); color: #b45309; }
    .pill-bird     { background: rgba(59,130,246,0.15); color: #1d4ed8; }
    .pill-other    { background: rgba(16,185,129,0.15); color: #065f46; }
    .metrics-conf-row {
      font-size: 0.72rem; font-weight: 700; color: #666;
      display: flex; flex-wrap: wrap; gap: 14px; min-height: 16px;
    }

    /* ── Date nav ───────────────────────────────────────────────────────── */
    .date-nav { display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap; }
    .date-nav-btn {
      background: rgba(255,255,255,0.7); border: 1.5px solid rgba(0,119,182,0.25);
      color: var(--deep); padding: 7px 14px; border-radius: 20px;
      font-family: 'Nunito', sans-serif; font-size: 0.82rem; font-weight: 900;
      cursor: pointer; min-height: 36px; transition: background 0.15s;
    }
    .date-nav-btn:hover:not(:disabled) { background: rgba(0,180,216,0.15); }
    .date-nav-btn:disabled { opacity: 0.4; cursor: default; }
    .today-btn { background: rgba(0,180,216,0.18); border-color: var(--pool); }
    #date-display { font-size: 0.9rem; font-weight: 900; color: var(--navy); min-width: 160px; text-align: center; }

    /* ── Confidence filter ──────────────────────────────────────────────── */
    .conf-filter { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .conf-filter-label { font-size: 0.72rem; font-weight: 900; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .conf-btn {
      background: rgba(255,255,255,0.7); border: 1.5px solid rgba(0,119,182,0.25);
      color: var(--deep); padding: 5px 12px; border-radius: 20px;
      font-family: 'Nunito', sans-serif; font-size: 0.78rem; font-weight: 900;
      cursor: pointer; min-height: 32px; transition: background 0.15s, border-color 0.15s;
    }
    .conf-btn.active { background: var(--deep); color: white; border-color: var(--deep); }

    /* ── Grid ────────────────────────────────────────────────────────────── */
    #frames-grid {
      max-width: 1200px; margin: 0 auto;
      padding: 14px 14px 48px;
      display: grid; grid-template-columns: 1fr; gap: 14px;
    }
    @media (min-width: 600px)  { #frames-grid { grid-template-columns: 1fr 1fr; } }
    @media (min-width: 1000px) { #frames-grid { grid-template-columns: 1fr 1fr 1fr; } }

    /* ── Review card ─────────────────────────────────────────────────────── */
    .review-card {
      background: var(--glass); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      border-radius: 18px; overflow: hidden;
      box-shadow: 0 4px 16px rgba(0,100,160,0.10);
      border-left: 4px solid rgba(0,119,182,0.18);
    }
    .review-card.squirrel { border-left-color: #f59e0b; }

    .img-wrap { position: relative; line-height: 0; background: #d0eaf5; cursor: zoom-in; }
    .img-wrap img { width: 100%; display: block; border-radius: 0; }
    .bbox-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }

    .card-meta { display: flex; align-items: center; justify-content: space-between; padding: 8px 12px 4px; gap: 8px; }
    .cls {
      font-size: 0.72rem; font-weight: 900; text-transform: uppercase;
      letter-spacing: 1px; padding: 3px 9px; border-radius: 30px;
      background: rgba(0,119,182,0.12); color: var(--deep);
    }
    .cls.squirrel { background: rgba(245,158,11,0.15); color: #b45309; }
    .cls.bird     { background: rgba(59,130,246,0.12); color: #1d4ed8; }
    .conf { font-size: 0.78rem; font-weight: 900; color: var(--deep); }
    .ts { font-size: 0.65rem; color: #888; padding: 0 12px 6px; font-weight: 700; }

    .label-btns { display: flex; gap: 8px; padding: 8px 12px 12px; }
    .label-btn {
      flex: 1; padding: 10px 8px; border-radius: 12px; border: none;
      font-family: 'Nunito', sans-serif; font-size: 0.85rem; font-weight: 900;
      cursor: pointer; min-height: 44px; transition: opacity 0.15s, transform 0.1s;
    }
    .label-btn:active { transform: scale(0.96); }
    .squirrel-btn { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; }
    .not-btn      { background: linear-gradient(135deg, #15803d, #22c55e); color: white; }
    .label-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    @keyframes card-exit {
      0%   { opacity: 1; transform: scale(1); }
      100% { opacity: 0; transform: scale(0.85); }
    }
    .card-exiting { animation: card-exit 0.25s ease-in forwards; pointer-events: none; }

    #empty-state { grid-column: 1 / -1; text-align: center; padding: 80px 20px; font-size: 1.1rem; font-weight: 700; color: #888; }

    /* ── Lightbox ────────────────────────────────────────────────────────── */
    #lb-overlay {
      display: none; position: fixed; inset: 0; z-index: 200;
      background: rgba(0,0,0,0.96); touch-action: none; overflow: hidden;
    }
    #lb-overlay.open { display: block; }
    #lb-container { position: absolute; top: 50%; left: 50%; transform-origin: 0 0; will-change: transform; }
    #lb-image { display: block; }
    #lb-bbox-canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
    #lb-close-btn {
      position: fixed; top: 12px; right: 12px; z-index: 201; display: none;
      background: rgba(0,0,0,0.7); color: white; border: none; border-radius: 50%;
      width: 48px; height: 48px; font-size: 1.2rem; cursor: pointer;
    }
    #lb-hint {
      position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
      background: rgba(0,0,0,0.6); color: rgba(255,255,255,0.75);
      padding: 6px 16px; border-radius: 20px;
      font-size: 0.75rem; font-weight: 700;
      pointer-events: none; z-index: 201; display: none;
    }
  </style>
</head>
<body>

<header>
  <a href="/" class="nav-back">← Dashboard</a>
  <div class="header-title">
    <h1>🏷️ Label Frames</h1>
    <p class="tagline">Build the training dataset</p>
    <p id="counter"></p>
  </div>
  <div style="width:90px"></div>
</header>

<div class="controls-bar">
  <div class="metrics-bar">
    <div class="metrics-progress-row">
      <div class="progress-bar-wrap">
        <div class="progress-bar-fill" id="progress-fill" style="width:0%"></div>
      </div>
      <span class="progress-label" id="progress-label">— / — labeled</span>
    </div>
    <div class="metrics-pills" id="metrics-pills"></div>
    <div class="metrics-conf-row" id="metrics-conf-row"></div>
  </div>

  <div class="date-nav">
    <button class="date-nav-btn" id="prev-date-btn" onclick="goDate(-1)">← Prev</button>
    <span id="date-display">—</span>
    <button class="date-nav-btn" id="next-date-btn" onclick="goDate(1)">Next →</button>
    <button class="date-nav-btn today-btn" onclick="goToday()">Today</button>
  </div>

  <div class="conf-filter">
    <span class="conf-filter-label">Confidence:</span>
    <button class="conf-btn active" data-conf="0"   onclick="setConf(0)">All</button>
    <button class="conf-btn"        data-conf="0.5" onclick="setConf(0.5)">≥ 0.50</button>
    <button class="conf-btn"        data-conf="0.7" onclick="setConf(0.7)">≥ 0.70</button>
    <button class="conf-btn"        data-conf="0.9" onclick="setConf(0.9)">≥ 0.90</button>
  </div>
</div>

<div id="frames-grid"></div>

<div id="lb-overlay">
  <div id="lb-container">
    <img id="lb-image" alt="Frame">
    <canvas id="lb-bbox-canvas"></canvas>
  </div>
</div>
<button id="lb-close-btn" onclick="closeLb()">✕</button>
<div id="lb-hint">Pinch to zoom · Double-tap to reset</div>

<script>
  // ── State ──────────────────────────────────────────────────────────────────
  let allFrames    = [];
  let currentStats = {};
  let currentDate  = null;
  let availableDates = [];
  let minConf      = 0;

  const BIRD_CLASSES = new Set([
    'bird','crow','sparrow','robin','hawk','eagle','dove','pigeon','owl',
    'heron','goose','duck','jay','finch','cardinal','wren','thrush',
    'blackbird','starling','swallow','mockingbird','nuthatch','chickadee',
    'woodpecker','kestrel','osprey','vulture','egret','ibis','grebe',
  ]);

  const BOX_COLORS = { squirrel: '#f59e0b', bird: '#3b82f6' };
  function boxColor(cls) { return BOX_COLORS[cls] || '#10b981'; }

  // ── Filtering ──────────────────────────────────────────────────────────────
  function filtered() {
    return allFrames.filter(f => {
      const c = parseFloat(f.confidence);
      return isNaN(c) ? minConf === 0 : c >= minConf;
    });
  }

  // ── Metrics ────────────────────────────────────────────────────────────────
  function renderMetrics(stats) {
    const labeled = stats.labeled_count || 0;
    const pending = stats.pending_count || 0;
    const total   = labeled + pending;
    const pct     = total > 0 ? Math.round((labeled / total) * 100) : 0;

    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('progress-label').textContent =
      labeled + ' / ' + total + ' labeled (' + pct + '%)';

    const dist = stats.class_distribution || {};
    let squirrelCount = 0, birdCount = 0, otherCount = 0;
    for (const [cls, n] of Object.entries(dist)) {
      if (cls === 'squirrel')       squirrelCount += n;
      else if (BIRD_CLASSES.has(cls)) birdCount   += n;
      else                            otherCount  += n;
    }

    const pillsEl = document.getElementById('metrics-pills');
    pillsEl.innerHTML = '';
    if (squirrelCount > 0)
      pillsEl.innerHTML += '<span class="metrics-pill pill-squirrel">🐿️ ' + squirrelCount + ' squirrel</span>';
    if (birdCount > 0)
      pillsEl.innerHTML += '<span class="metrics-pill pill-bird">🐦 ' + birdCount + ' bird</span>';
    if (otherCount > 0)
      pillsEl.innerHTML += '<span class="metrics-pill pill-other">🦌 ' + otherCount + ' other</span>';

    const confEl = document.getElementById('metrics-conf-row');
    const avg    = stats.avg_confidence;
    const high   = stats.high_conf_count || 0;
    const thresh = stats.high_conf_threshold || 0.70;
    if (avg !== null && avg !== undefined) {
      confEl.innerHTML =
        '<span>Avg confidence: <strong>' + avg.toFixed(2) + '</strong></span>' +
        '<span>' + high + ' frame' + (high !== 1 ? 's' : '') + ' ≥ ' + thresh.toFixed(2) + '</span>';
    } else {
      confEl.innerHTML = '';
    }
  }

  // ── Date navigation ────────────────────────────────────────────────────────
  function todayStr() { return new Date().toISOString().slice(0, 10); }

  function renderDateNav() {
    const today = todayStr();
    const idx   = availableDates.indexOf(currentDate);

    // availableDates sorted descending: older = higher index
    document.getElementById('prev-date-btn').disabled = idx < 0 || idx >= availableDates.length - 1;
    document.getElementById('next-date-btn').disabled = idx <= 0;

    if (currentDate) {
      const d     = new Date(currentDate + 'T12:00:00');
      const label = d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
      document.getElementById('date-display').textContent =
        currentDate === today ? label + ' (today)' : label;
    } else {
      document.getElementById('date-display').textContent = 'No dates';
    }
  }

  function goDate(dir) {
    // dir: -1 = Prev (older date = higher index in desc array)
    //      +1 = Next (newer date = lower index)
    const idx    = availableDates.indexOf(currentDate);
    if (idx < 0) return;
    const newIdx = dir === -1 ? idx + 1 : idx - 1;
    if (newIdx >= 0 && newIdx < availableDates.length) {
      currentDate = availableDates[newIdx];
      loadDate();
    }
  }

  function goToday() {
    currentDate = todayStr();
    loadDate();
  }

  // ── Grid rendering ─────────────────────────────────────────────────────────
  function renderGrid() {
    const frames = filtered();
    const total  = allFrames.length;
    const shown  = frames.length;

    if (total === 0) {
      document.getElementById('counter').textContent = 'No pending frames for this day';
    } else if (minConf > 0 && shown < total) {
      document.getElementById('counter').textContent =
        shown + ' of ' + total + ' shown (≥' + minConf.toFixed(2) + ' conf)';
    } else {
      document.getElementById('counter').textContent =
        total + ' frame' + (total !== 1 ? 's' : '') + ' pending review';
    }

    const grid = document.getElementById('frames-grid');
    grid.innerHTML = '';

    if (frames.length === 0) {
      grid.innerHTML = total === 0
        ? '<div id="empty-state">🎉 All frames labeled for this day!</div>'
        : '<div id="empty-state">No frames match the confidence filter.</div>';
      return;
    }
    frames.forEach(f => grid.appendChild(makeCard(f)));
  }

  // ── Confidence filter ──────────────────────────────────────────────────────
  function setConf(val) {
    minConf = val;
    document.querySelectorAll('.conf-btn').forEach(b =>
      b.classList.toggle('active', parseFloat(b.dataset.conf) === val));
    renderGrid();
  }

  // ── Box drawing ────────────────────────────────────────────────────────────
  function drawCardBox(img) {
    const x1 = +img.dataset.x1, y1 = +img.dataset.y1;
    const w  = +img.dataset.w,  h  = +img.dataset.h;
    if (!w || !h) return;
    const canvas = img.nextElementSibling;
    canvas.width  = img.clientWidth;
    canvas.height = img.clientHeight;
    const sx = img.clientWidth  / img.naturalWidth;
    const sy = img.clientHeight / img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = boxColor(img.dataset.cls);
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x1 * sx, y1 * sy, w * sx, h * sy);
  }

  // ── Card creation ──────────────────────────────────────────────────────────
  function ago(isoStr) {
    if (!isoStr) return '';
    const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000);
    if (diff < 60)    return diff + 's ago';
    if (diff < 3600)  return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    return Math.floor(diff / 86400) + 'd ago';
  }

  function makeCard(f) {
    const cls    = (f.predicted_class || '').toLowerCase();
    const conf   = f.confidence ? Math.round(parseFloat(f.confidence) * 100) + '%' : '—';
    const clsCss = cls || 'unknown';

    const card = document.createElement('div');
    card.className = 'review-card' + (cls === 'squirrel' ? ' squirrel' : '');

    card.innerHTML =
      '<div class="img-wrap">' +
        '<img src="' + f.image_url + '" alt="' + (cls || 'frame') + '" loading="lazy"' +
             ' data-x1="' + (f.x1||0) + '" data-y1="' + (f.y1||0) + '"' +
             ' data-w="'  + (f.w ||0) + '" data-h="'  + (f.h ||0) + '"' +
             ' data-cls="' + cls + '" onload="drawCardBox(this)">' +
        '<canvas class="bbox-canvas"></canvas>' +
      '</div>' +
      '<div class="card-meta">' +
        '<span class="cls ' + clsCss + '">' + (cls || 'unknown') + '</span>' +
        '<span class="conf">' + conf + '</span>' +
      '</div>' +
      '<div class="ts">' + ago(f.timestamp) + (f.timestamp ? ' · ' + f.timestamp : '') + '</div>' +
      '<div class="label-btns">' +
        '<button class="label-btn squirrel-btn">🐿️ Squirrel</button>' +
        '<button class="label-btn not-btn">✓ Not Squirrel</button>' +
      '</div>';

    const [sqBtn, notBtn] = card.querySelectorAll('.label-btn');
    sqBtn.addEventListener('click', () =>
      labelFrame(card, f.frame_path, 'squirrel', f.predicted_class, f.confidence));
    notBtn.addEventListener('click', () =>
      labelFrame(card, f.frame_path, 'not_squirrel', f.predicted_class, f.confidence));

    card.querySelector('.img-wrap').addEventListener('click', () => {
      openLb(card.querySelector('img'), {
        x1: +(f.x1||0), y1: +(f.y1||0), w: +(f.w||0), h: +(f.h||0), cls,
      });
    });

    return card;
  }

  // ── Labeling ───────────────────────────────────────────────────────────────
  async function labelFrame(card, framePath, label, predictedClass, confidence) {
    card.querySelectorAll('.label-btn').forEach(b => b.disabled = true);
    try {
      const res = await fetch('/api/frames/label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_path: framePath, label, predicted_class: predictedClass, confidence }),
      });
      const data = await res.json();
      if (data.ok) {
        allFrames = allFrames.filter(f => f.frame_path !== framePath);
        currentStats.labeled_count = (currentStats.labeled_count || 0) + 1;
        currentStats.pending_count = Math.max(0, (currentStats.pending_count || 0) - 1);
        if (currentStats.class_distribution && predictedClass) {
          const cls = predictedClass.toLowerCase();
          currentStats.class_distribution[cls] =
            Math.max(0, (currentStats.class_distribution[cls] || 0) - 1);
        }
        renderMetrics(currentStats);
        card.classList.add('card-exiting');
        card.addEventListener('animationend', () => { card.remove(); renderGrid(); }, { once: true });
      } else {
        card.querySelectorAll('.label-btn').forEach(b => b.disabled = false);
      }
    } catch {
      card.querySelectorAll('.label-btn').forEach(b => b.disabled = false);
    }
  }

  // ── Load a day ─────────────────────────────────────────────────────────────
  async function loadDate() {
    renderDateNav();
    let data;
    try {
      const qs = currentDate ? '?date=' + encodeURIComponent(currentDate) : '';
      data = await (await fetch('/api/frames/pending' + qs)).json();
    } catch {
      document.getElementById('frames-grid').innerHTML =
        '<div id="empty-state">Could not load frames — is the server running?</div>';
      return;
    }
    allFrames    = data.frames || [];
    currentStats = data.stats  || {};
    renderMetrics(currentStats);
    renderGrid();
  }

  // ── Lightbox ───────────────────────────────────────────────────────────────
  let _lbScale = 1, _lbX = 0, _lbY = 0;
  let _lbImgData = null;
  let _lbTouchDist = 0, _lbTouchMid = null, _lbScaleStart = 1, _lbPosStart = null;
  let _lbDragStart = null, _lbLastTap = 0;

  function _lbApply() {
    document.getElementById('lb-container').style.transform =
      `translate(calc(-50% + ${_lbX}px), calc(-50% + ${_lbY}px)) scale(${_lbScale})`;
  }

  function _lbDrawBox() {
    const img    = document.getElementById('lb-image');
    const canvas = document.getElementById('lb-bbox-canvas');
    if (!_lbImgData || !_lbImgData.w || !_lbImgData.h) { canvas.style.display = 'none'; return; }
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width  = img.naturalWidth  + 'px';
    canvas.style.height = img.naturalHeight + 'px';
    canvas.style.display = '';
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = boxColor(_lbImgData.cls);
    ctx.lineWidth = Math.max(2, img.naturalWidth / 200);
    ctx.strokeRect(_lbImgData.x1, _lbImgData.y1, _lbImgData.w, _lbImgData.h);
  }

  function openLb(imgEl, data) {
    _lbImgData = data; _lbScale = 1; _lbX = 0; _lbY = 0;
    const img = document.getElementById('lb-image');
    img.onload = () => {
      img.style.width  = img.naturalWidth  + 'px';
      img.style.height = img.naturalHeight + 'px';
      _lbDrawBox();
    };
    img.src = imgEl.src;
    document.getElementById('lb-overlay').classList.add('open');
    document.getElementById('lb-close-btn').style.display = 'block';
    document.getElementById('lb-hint').style.display = 'block';
    _lbApply();
  }

  function closeLb() {
    document.getElementById('lb-overlay').classList.remove('open');
    document.getElementById('lb-close-btn').style.display = 'none';
    document.getElementById('lb-hint').style.display = 'none';
    document.getElementById('lb-image').src = '';
    _lbImgData = null;
  }

  function _dist(t) { return Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY); }
  function _mid(t)  { return { x: (t[0].clientX + t[1].clientX) / 2, y: (t[0].clientY + t[1].clientY) / 2 }; }

  const lbOv = document.getElementById('lb-overlay');

  lbOv.addEventListener('touchstart', e => {
    if (e.touches.length === 2) {
      _lbTouchDist = _dist(e.touches); _lbTouchMid = _mid(e.touches);
      _lbScaleStart = _lbScale; _lbPosStart = { x: _lbX, y: _lbY }; _lbDragStart = null;
    } else if (e.touches.length === 1) {
      const now = Date.now();
      if (now - _lbLastTap < 300) { _lbScale = 1; _lbX = 0; _lbY = 0; _lbApply(); _lbLastTap = 0; return; }
      _lbLastTap   = now;
      _lbDragStart = { x: e.touches[0].clientX - _lbX, y: e.touches[0].clientY - _lbY };
    }
  }, { passive: true });

  lbOv.addEventListener('touchmove', e => {
    e.preventDefault();
    if (e.touches.length === 2) {
      const d = _dist(e.touches), m = _mid(e.touches);
      _lbScale = Math.max(0.5, Math.min(10, _lbScaleStart * (d / _lbTouchDist)));
      _lbX = _lbPosStart.x + (m.x - _lbTouchMid.x);
      _lbY = _lbPosStart.y + (m.y - _lbTouchMid.y);
      _lbApply();
    } else if (e.touches.length === 1 && _lbDragStart) {
      _lbX = e.touches[0].clientX - _lbDragStart.x;
      _lbY = e.touches[0].clientY - _lbDragStart.y;
      _lbApply();
    }
  }, { passive: false });

  lbOv.addEventListener('touchend', e => {
    if (e.touches.length < 2) { _lbTouchDist = 0; _lbTouchMid = null; }
    if (e.touches.length < 1)  _lbDragStart = null;
    if (e.changedTouches.length === 1 && e.touches.length === 0) {
      const t = e.changedTouches[0];
      if (t.target === lbOv && Math.abs(_lbX) < 4 && Math.abs(_lbY) < 4 && _lbScale < 1.05) closeLb();
    }
  }, { passive: true });

  lbOv.addEventListener('wheel', e => {
    e.preventDefault();
    _lbScale = Math.max(0.5, Math.min(10, _lbScale * (e.deltaY < 0 ? 1.12 : 0.89)));
    _lbApply();
  }, { passive: false });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeLb(); return; }
    if (!document.getElementById('lb-overlay').classList.contains('open')) {
      if (e.key === 'ArrowLeft')  goDate(-1);
      if (e.key === 'ArrowRight') goDate(1);
    }
  });

  // ── Boot ───────────────────────────────────────────────────────────────────
  async function init() {
    try {
      const datesData = await (await fetch('/api/frames/dates')).json();
      availableDates  = (datesData.dates || []).map(d => d.date);
    } catch {
      availableDates = [];
    }

    const today = todayStr();
    currentDate = availableDates.includes(today)
      ? today
      : (availableDates.length > 0 ? availableDates[0] : today);

    await loadDate();
  }

  init();
</script>
</body>
</html>"""


@app.get("/review", response_class=HTMLResponse)
def review():
    return REVIEW_HTML
