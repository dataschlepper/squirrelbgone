#!/usr/bin/env python3
"""
SquirrelBGone — api/server.py
Read-only mobile dashboard for reviewing recent detections.

Usage (from repo root):
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import csv
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

LOG_DIR    = Path(os.environ.get("LOG_DIR",    "logs"))
FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", "frames"))

CORRECTIONS_PATH   = LOG_DIR / "corrections.csv"
CORRECTION_FIELDS  = ["flagged_at", "detection_timestamp", "class", "confidence", "frame_path"]

app = FastAPI()

if FRAMES_DIR.exists():
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


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SquirrelBGone</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #111; color: #eee; }

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

    #summary {
      display: flex; gap: 8px; padding: 10px 12px;
      font-size: 0.8rem;
    }
    .pill { background: #1e1e1e; border-radius: 20px; padding: 4px 12px; color: #aaa; }
    .pill.squirrel { background: #451a03; color: #f59e0b; }
    .pill.bird     { background: #0c1a40; color: #60a5fa; }

    #cards { padding: 0 12px 24px; display: flex; flex-direction: column; gap: 10px; }

    .card {
      background: #1a1a1a; border-radius: 10px;
      overflow: hidden; border-left: 4px solid #333;
    }
    .card.squirrel { border-left-color: #f59e0b; }
    .card.bird     { border-left-color: #3b82f6; }
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

    /* Lightbox */
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
  </style>
</head>
<body>
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
    </div>
  </div>

  <div id="summary"></div>
  <div id="cards"></div>

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

    let currentWindow = 60;
    let currentFilter = 'all';
    let allData = [];
    let lbData  = null;
    const flaggedSet = new Set();
    const detMap = {};

    const BIRD_CLASSES = new Set(['bird','crow','pigeon','robin','sparrow']);

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

    function filtered() {
      if (currentFilter === 'squirrel') return allData.filter(d => d.class === 'squirrel');
      if (currentFilter === 'bird')     return allData.filter(d => BIRD_CLASSES.has(d.class));
      return allData;
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
      const other     = allData.length - squirrels - birds;

      const summary = document.getElementById('summary');
      summary.innerHTML =
        (squirrels ? `<span class="pill squirrel">${squirrels} squirrel${squirrels !== 1 ? 's' : ''}</span>` : '') +
        (birds     ? `<span class="pill bird">${birds} bird${birds !== 1 ? 's' : ''}</span>` : '') +
        (other     ? `<span class="pill">${other} other</span>` : '');

      const cards = document.getElementById('cards');
      if (!data.length) {
        cards.innerHTML = '<div id="empty">No detections in this window.</div>';
        return;
      }

      cards.innerHTML = data.map(d => {
        const cls      = (d.class || 'unknown').toLowerCase();
        const conf     = Math.round(parseFloat(d.confidence) * 100);
        const isFlagged = flaggedSet.has(d.timestamp);
        const imgHtml  = d.image_url ? `
          <div class="img-wrap" onclick="openLightbox(this.querySelector('img'))">
            <img src="${d.image_url}" alt="${cls}" loading="lazy"
                 data-x1="${d.x1 || 0}" data-y1="${d.y1 || 0}"
                 data-w="${d.w || 0}"   data-h="${d.h || 0}"
                 data-cls="${cls}" onload="drawBox(this)">
            <canvas class="bbox-canvas"></canvas>
          </div>` : '';
        return `
          <div class="card ${cls}${isFlagged ? ' flagged' : ''}">
            ${imgHtml}
            <div class="card-body">
              <div>
                <div class="cls ${cls}">${cls}</div>
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

    load();
    setInterval(load, 30000);
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
