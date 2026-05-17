# Software Architecture

## Overview

The core pipeline is a continuous loop: pull frames from RTSP → run YOLOv8 inference → evaluate detections → log everything. GPIO triggering is Phase 2+.

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
│    └── Wildlife model  (optional, MODEL_PATH_WILDLIFE — backlog)│
│      │                                                          │
│      ▼                                                          │
│  Squirrel boxes above CONFIDENCE_THRESHOLD?                     │
│      │                                                          │
│      ├── No ──► bird/wildlife present? log them; skip frame    │
│      │                                                          │
│      └── Yes ──► bird OR wildlife also in frame?               │
│                      │                                          │
│                      ├── Yes ──► triggered=False (suppressed)  │
│                      └── No  ──► within DAY_START–DAY_END?     │
│                                      │                          │
│                                      ├── No  ──► triggered=False (nighttime)
│                                      └── Yes ──► cooldown elapsed?          │
│                                                      │          │
│                                                      ├── No  ──► triggered=False (cooldown)
│                                                      └── Yes ──► triggered=True
│                                                                  GPIO pulse  │
│                                                                 │
│  Save frame + write CSV row for every detection                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Actual file structure

```
squirrelbgone/
├── detect.py            # Entry point — inference loop, logging
├── .env                 # Config (gitignored)
├── .env.example         # Config template
├── models/
│   └── squirrelbgone_best.pt   # Squirrel detector weights
├── api/
│   └── server.py        # FastAPI dashboard (read-only, live)
├── logs/
│   └── detections_YYYY-MM-DD.csv
├── frames/              # One JPEG saved per detection event
└── docs/
```

---

## detect.py

Single-file entry point. Manages stream, models, CSV, and the inference loop.

**Run:**
```bash
python detect.py
```

**Key env vars** (all read at module load from `.env`):

| Var | Default | Notes |
|---|---|---|
| `RTSP_URL` | — | Required |
| `MODEL_PATH` | `squirrel_detector.pt` | Required |
| `CONFIDENCE_THRESHOLD` | `0.45` | Squirrel detection threshold |
| `MODEL_PATH_BIRD` | `` (disabled) | Leave empty to skip bird model |
| `BIRD_CONFIDENCE_THRESHOLD` | `0.45` | |
| `MODEL_PATH_WILDLIFE` | `` (disabled) | Backlog — see below |
| `WILDLIFE_CONFIDENCE_THRESHOLD` | `0.45` | |
| `TARGET_FPS` | `5` | Frames processed per second |
| `LOG_DIR` | `logs` | |
| `FRAMES_DIR` | `frames` | |
| `GPIO_PIN` | `18` | BCM pin; physical pin 12 on Pi 5 |
| `COOLDOWN_SEC` | `10` | Minimum seconds between trigger events |
| `DAY_START` | `7` | Start hour (0–23, inclusive) |
| `DAY_END` | `20` | End hour (0–23, exclusive) |
| `SPRAY_DURATION_SEC` | `1.0` | GPIO pulse duration in seconds |

**Suppression logic:** If a bird or wildlife detection co-occurs in the same frame as a squirrel, the squirrel row is written with `triggered=False`. The squirrel trigger is suppressed regardless of confidence. Bird and wildlife detections are always written with `triggered=False`.

**Stream reconnect:** Exponential backoff (3s → 6s → 12s → … → 30s cap) on read failure.

**CSV rollover:** At midnight the current file handle is closed and a new dated file is opened automatically.

---

## Inference models

All models use **ultralytics YOLO** loaded with `YOLO(path)`.

| Model | Role | Format |
|---|---|---|
| Squirrel model | Primary detector, single class | Custom-trained YOLOv8 `.pt` |
| Bird model | Suppression — prevents false triggers at feeders | COCO YOLOv8 (e.g. `yolov8n.pt`) |
| Wildlife model | Suppression — deer, fox, raccoon, etc. (backlog) | Any YOLOv8 `.pt` with wildlife classes |

**Wildlife model backlog:** The code in `detect.py` fully supports `MODEL_PATH_WILDLIFE`. The blocker is sourcing a reliable pre-trained `.pt` — the `animal-detection-yolov8` dataset from Roboflow Universe would need to be trained locally (on Mac, ~30 min with yolov8n) then copied to the Pi. See `WILDLIFE_SUPPRESS_CLASSES` in `detect.py` for the full list of suppressing class names.

---

## Detection log schema

**`logs/detections_YYYY-MM-DD.csv`** — one row per bounding box, every frame that has at least one detection:

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

If the schema changes, the old file is archived as `detections_YYYY-MM-DD_legacy_<epoch>.csv` automatically on next open.

**`logs/corrections.csv`** — user-flagged false positives from the dashboard flag button:

| Column | Example |
|---|---|
| `flagged_at` | `2026-05-10T09:01:00` |
| `detection_timestamp` | `2026-05-10T08:59:18` |
| `class` | `squirrel` |
| `confidence` | `0.847` |
| `frame_path` | `frames/2026-05-10T08-59-18_squirrel.jpg` |

---

## api/server.py — Dashboard

Read-only FastAPI app. Run alongside `detect.py`:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | HTML dashboard (single-page, inline) |
| `/api/detections?minutes=N` | GET | Detections from the last N minutes (default 60) |
| `/api/config` | GET | Returns `log_threshold`, `high_conf_threshold` |
| `/api/flag` | POST | Write a row to `corrections.csv` |
| `/frames/<filename>` | GET | Static JPEG frames |

**Dashboard features:**
- Time window filter: 15m / 1h / Today
- Class filter: All / Squirrel / Bird / Wildlife
- High-confidence filter (≥ threshold)
- Bounding box overlay on card images and lightbox
- Flag button to mark false positives
- Auto-refresh every 30s

---

## GPIO trigger (Phase 2+)

**Phase 2 (current):** `gpiozero.LED(GPIO_PIN)` drives the dry-run LED on BCM pin 18 (physical pin 12).  
**Phase 3:** Swap to `OutputDevice(GPIO_PIN, active_high=False)` for the JBtek relay (active-low).

### Guard chain

Every potential trigger passes three guards in order before `_fire_gpio()` is called:

1. **Suppression** — bird or wildlife detected in the same frame → `triggered=False`
2. **Day/night guard** — `datetime.now().hour` outside `[DAY_START, DAY_END)` → `triggered=False`, logs `suppressed — nighttime`
3. **Cooldown** — fewer than `COOLDOWN_SEC` seconds since last trigger → `triggered=False`, logs `suppressed — cooldown Ns`

If all three pass, `last_trigger_time` is updated and `_fire_gpio()` calls `blink(on_time=SPRAY_DURATION_SEC, n=1, background=True)` — non-blocking.

The `triggered` column in the CSV reflects the final outcome after all guards.

### Graceful degradation

`gpiozero` is imported lazily inside `_setup_gpio()`. If the import fails (e.g. running on a dev Mac), the script logs a warning and continues without hardware output. Inference, cooldown tracking, and CSV logging all run normally.

---

## Dependencies

```
ultralytics          # YOLOv8 inference
opencv-python        # RTSP capture, frame saving
python-dotenv        # .env config loading
fastapi              # Dashboard API
uvicorn              # ASGI server
gpiozero             # GPIO relay (Phase 2+)
RPi.GPIO             # gpiozero backend on Pi (Phase 2+)
```
