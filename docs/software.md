# Software Architecture

## Overview

The core pipeline is a continuous loop: pull frames from RTSP → run YOLOv8 inference → evaluate detections → log everything.

```
┌─────────────────────────────────────────────────────────────────┐
│                        detect.py main loop                      │
│                                                                 │
│  RTSP stream (OpenCV)                                           │
│      │                                                          │
│      ▼                                                          │
│  Frame capture  ──── throttled to TARGET_FPS ────────────────► │
│      │                                                          │
│      ▼                                                          │
│  YOLOv8 inference (ultralytics)                                 │
│    ├── Squirrel model  (required)                               │
│    ├── Bird model      (optional, MODEL_PATH_BIRD)              │
│    └── Wildlife model  (optional, MODEL_PATH_WILDLIFE)          │
│      │                                                          │
│      ▼                                                          │
│  Squirrel boxes above CONFIDENCE_THRESHOLD?                     │
│      │                                                          │
│      ├── No ──► bird/wildlife present? log them; skip frame    │
│      │                                                          │
│      └── Yes ──► bird OR wildlife also in frame?               │
│                      │                                          │
│                      ├── Yes ──► triggered=False (suppressed)  │
│                      └── No  ──► squirrel center in feeder zone?
│                                      │                          │
│                                      ├── No  ──► triggered=False (outside feeder zone)
│                                      └── Yes ──► within DAY_START–DAY_END?
│                                                      │          │
│                                                      ├── No  ──► triggered=False (nighttime)
│                                                      └── Yes ──► cooldown elapsed?
│                                                                      │
│                                                                      ├── No  ──► triggered=False (cooldown)
│                                                                      └── Yes ──► conf ≥ SPRAY_CONFIDENCE_THRESHOLD?
│                                                                                      │
│                                                                                      ├── No  ──► triggered=False (low confidence)
│                                                                                      └── Yes ──► triggered=True → GPIO pulse
│                                                                 │
│  Save frame + write CSV row for every squirrel detection        │
└─────────────────────────────────────────────────────────────────┘
```

---

## File structure

```
squirrelbgone/
├── detect.py            # Entry point — inference loop, logging, GPIO
├── .env                 # Config (gitignored)
├── .env.example         # Config template
├── models/
│   └── squirrelbgone_best.pt   # Squirrel detector weights
├── api/
│   └── server.py        # FastAPI dashboard + MJPEG stream
├── systemd/
│   ├── squirrelbgone-detect.service   # systemd unit for detect.py
│   └── squirrelbgone-api.service      # systemd unit for uvicorn
├── logs/
│   ├── detections_YYYY-MM-DD.csv
│   ├── corrections.csv
│   ├── feeder_zone.json   # Saved feeder zone (fractions 0–1)
│   ├── spray.request      # Transient: dashboard → detect.py spray signal
│   └── solenoid.state     # Transient: dashboard → detect.py hold-open state
├── frames/              # JPEG saved per squirrel detection (auto-purged)
└── docs/
```

---

## Running as services (production)

Both processes run as systemd services so they start on boot and restart on failure.

```bash
# Install (one-time)
sudo cp systemd/squirrelbgone-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable squirrelbgone-api squirrelbgone-detect
sudo systemctl start  squirrelbgone-api squirrelbgone-detect

# Logs
journalctl -u squirrelbgone-detect -f
journalctl -u squirrelbgone-api -f

# Deploy after a code change: pull + sync service files + restart
# Add to ~/.bashrc for convenience:
alias sbg-deploy='cd ~/squirrelbgone && scripts/deploy.sh'
sbg-deploy

# Quick restart only (no pull, no service file sync)
alias sbg-restart='sudo systemctl restart squirrelbgone-detect squirrelbgone-api'
```

---

## detect.py

Single-file entry point. Manages stream, models, feeder zone, frame cleanup, CSV, and the inference loop.

**Key env vars** (all read at module load from `.env`):

| Var | Default | Notes |
|---|---|---|
| `RTSP_URL` | — | Required |
| `MODEL_PATH` | `squirrel_detector.pt` | Required |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum confidence to log a squirrel detection |
| `SPRAY_CONFIDENCE_THRESHOLD` | `0.80` | Minimum confidence to actually fire the solenoid |
| `MODEL_PATH_BIRD` | `` (disabled) | Leave empty to skip bird suppression model |
| `BIRD_CONFIDENCE_THRESHOLD` | `0.45` | |
| `MODEL_PATH_WILDLIFE` | `` (disabled) | Optional wildlife suppression model |
| `WILDLIFE_CONFIDENCE_THRESHOLD` | `0.45` | |
| `TARGET_FPS` | `5` | Frames processed per second |
| `LOG_DIR` | `logs` | |
| `FRAMES_DIR` | `frames` | |
| `FRAMES_KEEP_DAYS` | `7` | Frames older than this are deleted at startup and nightly |
| `GPIO_PIN` | `17` | BCM pin number |
| `COOLDOWN_SEC` | `10` | Minimum seconds between trigger events |
| `DAY_START` | `7` | Start hour (0–23, inclusive) |
| `DAY_END` | `20` | End hour (0–23, exclusive) |
| `SPRAY_DURATION_SEC` | `1.0` | Solenoid open duration in seconds |
| `AUTO_SPRAY_ENABLED` | `false` | Set to `true` to enable automatic squirrel-triggered spraying |

**Trigger guard chain** — squirrel detections pass all five guards before GPIO fires:

1. **Bird/wildlife suppression** — co-present in same frame → `triggered=False`
2. **Feeder zone** — squirrel bbox center outside `logs/feeder_zone.json` → `triggered=False`
3. **Day/night guard** — hour outside `[DAY_START, DAY_END)` → `triggered=False`
4. **Cooldown** — fewer than `COOLDOWN_SEC` seconds since last trigger → `triggered=False`
5. **Confidence** — below `SPRAY_CONFIDENCE_THRESHOLD` → `triggered=False`

**Feeder zone:** Loaded from `logs/feeder_zone.json` (set via dashboard zone picker). Stored as 0–1 fractions so it's resolution-independent. Reloaded automatically whenever the file changes — no restart needed after saving a new zone.

**Frame cleanup:** At startup and at midnight rollover, frames older than `FRAMES_KEEP_DAYS` are deleted automatically.

**Stream reconnect:** Exponential backoff (3s → 6s → 12s → … → 30s cap) on read failure.

**CSV rollover:** At midnight the current file handle is closed and a new dated file is opened automatically.

---

## Inference models

All models use **ultralytics YOLO** loaded with `YOLO(path)`.

| Model | Role | Format |
|---|---|---|
| Squirrel model | Primary detector, single class | Custom-trained YOLOv8 `.pt` |
| Bird model | Suppression — prevents false triggers when birds are present | COCO YOLOv8 (e.g. `yolov8n.pt`) |
| Wildlife model | Suppression — deer, fox, raccoon, etc. | Any YOLOv8 `.pt` with wildlife classes |

---

## Detection log schema

**`logs/detections_YYYY-MM-DD.csv`** — one row per bounding box, every frame with at least one detection:

| Column | Type | Example |
|---|---|---|
| `timestamp` | ISO 8601 (seconds) | `2026-05-10T08:59:18` |
| `class` | string | `squirrel`, `bird`, `deer` |
| `confidence` | float | `0.847` |
| `triggered` | bool | `True` |
| `x1` | int | `312` |
| `y1` | int | `140` |
| `w` | int | `94` |
| `h` | int | `87` |
| `frame_path` | string | `frames/2026-05-10T08-59-18_squirrel.jpg` |

If the schema changes, the old file is archived as `detections_YYYY-MM-DD_legacy_<epoch>.csv`.

**`logs/corrections.csv`** — user-flagged false positives from the dashboard:

| Column | Example |
|---|---|
| `flagged_at` | `2026-05-10T09:01:00` |
| `detection_timestamp` | `2026-05-10T08:59:18` |
| `class` | `squirrel` |
| `confidence` | `0.847` |
| `frame_path` | `frames/2026-05-10T08-59-18_squirrel.jpg` |

---

## api/server.py — Dashboard

FastAPI app serving the dashboard and MJPEG live stream. Runs alongside `detect.py` and communicates via files in `logs/`.

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Responsive dashboard (desktop + mobile) |
| `/mobile` | GET | Redirects to `/` |
| `/api/stream` | GET | MJPEG live stream (scaled to ≤960px wide) |
| `/api/detections?minutes=N` | GET | Detections from the last N minutes (default 60) |
| `/api/config` | GET | Returns `log_threshold`, `high_conf_threshold` |
| `/api/flag` | POST | Write a row to `corrections.csv` |
| `/api/spray?duration=N` | POST | Queue a timed spray pulse for detect.py to fire |
| `/api/spray-status` | GET | Whether a spray request is pending |
| `/api/solenoid/on` | POST | Hold solenoid open until turned off |
| `/api/solenoid/off` | POST | Release solenoid |
| `/api/solenoid-status` | GET | Current solenoid hold state |
| `/api/zone` | GET | Current feeder zone (fractions) or `{"zone": null}` |
| `/api/zone` | POST | Save feeder zone `{x1, y1, x2, y2}` as fractions |
| `/api/zone` | DELETE | Clear feeder zone |
| `/frames/<filename>` | GET | Static JPEG frames |

**Dashboard features:**
- MJPEG live stream with feeder zone overlay on detection cards
- Feeder zone picker — click-drag on live feed to define zone, saved immediately
- Time window filter: 15m / 1h / Today
- Class filter: All / Squirrel / Bird / Wildlife
- High-confidence filter (≥ threshold)
- Bounding box + zone overlay on card images and lightbox
- Flag button to mark false positives
- Hardware test panel: timed spray pulse + solenoid hold-open toggle
- Auto-refresh every 30s

**Inter-process communication:** The API server and detect.py share state via files in `logs/`:
- `spray.request` — API writes; detect.py reads and deletes on next loop
- `solenoid.state` — API writes; detect.py reads and applies on next loop
- `feeder_zone.json` — API writes; detect.py polls mtime and reloads when changed

---

## GPIO

Control scheme (verified on hardware):
- **Off / valve closed** → GPIO pin set to input, no pull (external 5V pull-up holds relay IN high)
- **On / valve open** → GPIO pin driven output low (pulls relay IN to 0V, energises relay)

Never drive the pin high to turn off: Pi 3.3V logic-high can't release a relay with a 5V pull-up on IN.

GPIO is managed via `pinctrl` subprocess calls (not gpiozero) for compatibility with Pi 5.

---

## Dependencies

```
ultralytics          # YOLOv8 inference
opencv-python        # RTSP capture, frame saving
python-dotenv        # .env config loading
fastapi              # Dashboard API
uvicorn              # ASGI server
```
