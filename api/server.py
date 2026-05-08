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
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

LOG_DIR    = Path(os.environ.get("LOG_DIR",    "logs"))
FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", "frames"))

app = FastAPI()

if FRAMES_DIR.exists():
    app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")


def _read_recent(minutes: int) -> list[dict]:
    cutoff = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
    rows = []

    # Check today's and yesterday's file so an hour window works near midnight
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

    #summary {
      display: flex; gap: 8px; padding: 12px;
      font-size: 0.8rem;
    }
    .pill {
      background: #1e1e1e; border-radius: 20px;
      padding: 4px 12px; color: #aaa;
    }
    .pill.squirrel { background: #451a03; color: #f59e0b; }
    .pill.bird     { background: #0c1a40; color: #60a5fa; }

    #cards { padding: 0 12px 24px; display: flex; flex-direction: column; gap: 10px; }

    .card {
      background: #1a1a1a; border-radius: 10px;
      overflow: hidden; border-left: 4px solid #333;
    }
    .card.squirrel { border-left-color: #f59e0b; }
    .card.bird     { border-left-color: #3b82f6; }

    .card img {
      width: 100%; display: block;
      max-height: 220px; object-fit: cover; background: #222;
    }
    .card-body {
      padding: 10px 12px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .cls { font-size: 0.9rem; font-weight: 600; text-transform: capitalize; }
    .cls.squirrel { color: #f59e0b; }
    .cls.bird     { color: #60a5fa; }
    .ts  { font-size: 0.7rem; color: #666; margin-top: 2px; }
    .conf { font-size: 1.2rem; font-weight: 700; }

    #empty { text-align: center; color: #555; padding: 60px 20px; font-size: 0.9rem; }
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

  <div id="summary"></div>
  <div id="cards"></div>

  <script>
    function ago(isoStr) {
      const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000);
      if (diff < 60)   return diff + 's ago';
      if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
      return Math.floor(diff / 3600) + 'h ago';
    }

    async function load() {
      let data;
      try {
        const res = await fetch('/api/detections?minutes=60');
        data = await res.json();
      } catch (e) {
        document.getElementById('meta').textContent = 'Error loading detections.';
        return;
      }

      const squirrels = data.filter(d => d.class === 'squirrel').length;
      const birds     = data.filter(d => ['bird','crow','pigeon','robin','sparrow'].includes(d.class)).length;
      const other     = data.length - squirrels - birds;

      document.getElementById('meta').textContent =
        data.length + ' detection' + (data.length !== 1 ? 's' : '') +
        ' · last hour · ' + new Date().toLocaleTimeString();

      const summary = document.getElementById('summary');
      summary.innerHTML =
        (squirrels ? `<span class="pill squirrel">${squirrels} squirrel${squirrels !== 1 ? 's' : ''}</span>` : '') +
        (birds     ? `<span class="pill bird">${birds} bird${birds !== 1 ? 's' : ''}</span>` : '') +
        (other     ? `<span class="pill">${other} other</span>` : '');

      const cards = document.getElementById('cards');
      if (!data.length) {
        cards.innerHTML = '<div id="empty">No detections in the last hour.</div>';
        return;
      }

      cards.innerHTML = data.map(d => {
        const cls  = (d.class || 'unknown').toLowerCase();
        const conf = Math.round(parseFloat(d.confidence) * 100);
        const img  = d.image_url
          ? `<img src="${d.image_url}" alt="${cls}" loading="lazy">`
          : '';
        return `
          <div class="card ${cls}">
            ${img}
            <div class="card-body">
              <div>
                <div class="cls ${cls}">${cls}</div>
                <div class="ts">${ago(d.timestamp)} &middot; ${d.timestamp}</div>
              </div>
              <div class="conf">${conf}%</div>
            </div>
          </div>`;
      }).join('');
    }

    load();
    setInterval(load, 30000);
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
