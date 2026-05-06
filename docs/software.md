# Software Architecture

## Overview

The core pipeline is a continuous loop: pull frames from RTSP → run YOLOv8 inference → evaluate detections → trigger GPIO if warranted → log everything.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Loop                                │
│                                                                 │
│  RTSP stream                                                    │
│      │                                                          │
│      ▼                                                          │
│  Frame capture  ──────────────────────────────► Skip if None   │
│      │                                                          │
│      ▼                                                          │
│  YOLOv8 inference (Roboflow Inference SDK)                      │
│      │                                                          │
│      ▼                                                          │
│  Filter detections                                              │
│    - class == "squirrel"?                                       │
│    - confidence >= threshold?                                   │
│      │                                                          │
│      ├── No match ──► log detection only, continue             │
│      │                                                          │
│      └── Match ──► cooldown check                              │
│                        │                                        │
│                        ├── In cooldown ──► skip, continue      │
│                        │                                        │
│                        └── Clear ──► schedule guard            │
│                                          │                      │
│                                          ├── Night ──► skip    │
│                                          │                      │
│                                          └── Day ──► TRIGGER   │
│                                                   │             │
│                                              GPIO pulse         │
│                                         (relay → solenoid)     │
│                                                   │             │
│                                              save frame         │
│                                              write log entry    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module structure

```
squirrelbgone/
├── main.py                  # Entry point; starts the main loop
├── config.py                # Loads and validates config.yaml
├── inference/
│   ├── stream.py            # RTSP stream capture (OpenCV)
│   └── detector.py          # Roboflow Inference SDK wrapper; returns detections
├── trigger/
│   ├── cooldown.py          # Cooldown timer logic
│   ├── schedule.py          # Day/night schedule guard
│   └── gpio_relay.py        # gpiozero relay pulse; spray duration
├── logging/
│   ├── detection_log.py     # Writes every inference result to CSV
│   └── trigger_log.py       # Writes spray events to separate CSV; saves frames
└── api/                     # Phase 4 — FastAPI backend
    ├── main.py
    ├── routes/
    │   ├── feed.py          # Live RTSP feed endpoint
    │   ├── detections.py    # Detection log endpoint
    │   └── sprays.py        # Spray history endpoint
    └── dashboard/           # Frontend with bounding box overlay
```

---

## Inference

Uses the **Roboflow Inference SDK** to run a hosted or locally-cached YOLOv8 model.

```python
from inference import InferencePipeline

# or for single-frame inference:
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",  # local inference server
    api_key=config.roboflow_api_key
)

result = client.infer(frame, model_id=config.roboflow_model_id)
```

Each result contains a list of predictions with:
- `class` — detected class label (e.g. `"squirrel"`, `"bird"`)
- `confidence` — float 0–1
- `x`, `y`, `width`, `height` — bounding box (center format)

**Model:** See `config.yaml` for `roboflow_model_id`. Find squirrel/bird models at [Roboflow Universe](https://universe.roboflow.com) — search "squirrel" or "squirrel bird". Note the workspace/model/version string (e.g. `"squirrels-birds/1"`).

---

## Detection filtering

```python
TRIGGER_CLASSES = {"squirrel"}
SUPPRESS_CLASSES = {"bird", "shadow"}  # logged but never trigger spray

def should_trigger(prediction, config) -> bool:
    return (
        prediction["class"] in TRIGGER_CLASSES
        and prediction["confidence"] >= config.confidence_threshold
    )
```

Non-trigger classes are logged but never actuate the relay.

---

## Cooldown logic

Prevents re-triggering on every frame during a single squirrel event.

```python
import time

class CooldownTimer:
    def __init__(self, cooldown_sec: float):
        self.cooldown_sec = cooldown_sec
        self._last_trigger: float = 0

    def is_clear(self) -> bool:
        return (time.monotonic() - self._last_trigger) >= self.cooldown_sec

    def reset(self):
        self._last_trigger = time.monotonic()
```

---

## Schedule guard

No triggers outside configured daytime hours.

```python
from datetime import datetime, time as dtime

def is_daytime(config) -> bool:
    now = datetime.now().time()
    return dtime(*config.day_start) <= now <= dtime(*config.day_end)
```

`day_start` and `day_end` are `[hour, minute]` pairs in `config.yaml`.

---

## GPIO relay control

```python
from gpiozero import OutputDevice
import time

relay = OutputDevice(config.gpio_pin, active_high=False)  # most relay modules are active-low

def spray(duration_sec: float):
    relay.on()
    time.sleep(duration_sec)
    relay.off()
```

> **Note:** Most relay modules (including JBtek) are active-low — the relay energizes when the GPIO pin goes LOW. Set `active_high=False` in gpiozero. Verify with the LED dry run before connecting the solenoid.

---

## Logging

### Detection log — `detections.csv`

Written on every inference pass (whether or not a spray is triggered).

| Column | Type | Example |
|---|---|---|
| `timestamp` | ISO 8601 string | `2025-03-14T09:26:53.441Z` |
| `class` | string | `squirrel` |
| `confidence` | float | `0.847` |
| `triggered` | bool | `True` |
| `frame_path` | string or empty | `frames/20250314_092653.jpg` |
| `bbox_x` | float | `412.3` |
| `bbox_y` | float | `208.7` |
| `bbox_w` | float | `94.1` |
| `bbox_h` | float | `87.5` |

### Trigger log — `triggers.csv`

Written only when a spray fires.

| Column | Type | Example |
|---|---|---|
| `timestamp` | ISO 8601 string | `2025-03-14T09:26:53.441Z` |
| `class` | string | `squirrel` |
| `confidence` | float | `0.847` |
| `spray_duration_sec` | float | `1.5` |
| `frame_path` | string | `frames/20250314_092653.jpg` |

### Frame saves

Save frames on every trigger (and optionally on low-confidence detections for review):

```python
import cv2

def save_frame(frame, label: str, confidence: float, timestamp: str) -> str:
    filename = f"frames/{timestamp}_{label}_{confidence:.2f}.jpg"
    cv2.imwrite(filename, frame)
    return filename
```

---

## Phase 4 — Nightly AI batch review

A separate script (`scripts/nightly_review.py`) runs on a cron job and sends flagged frames to Claude or GPT-4V for classification.

**Flagged frames** = triggers where confidence was below a review threshold (e.g. < 0.85), or any detection marked for manual review.

```
cron: 0 2 * * * python /home/pi/squirrelbgone/scripts/nightly_review.py
```

**Flow:**
1. Query `triggers.csv` for flagged entries since last run
2. For each flagged frame, send to vision API with prompt:
   > "What animal is in this image? Reply with one of: squirrel, bird, cat, other, none. Then rate your confidence 0–1."
3. Parse response into `corrections.csv`: `timestamp, original_class, ai_class, ai_confidence`
4. Use correction log to manually review and adjust `confidence_threshold` in `config.yaml`

---

## Phase 4 — FastAPI backend

Runs on the Pi, accessible at `http://<pi-ip>:8000` on the LAN.

| Endpoint | Method | Description |
|---|---|---|
| `/feed` | GET | MJPEG live stream with bounding box overlay |
| `/detections` | GET | Recent detection log (JSON, paginated) |
| `/sprays` | GET | Spray history (JSON, paginated) |
| `/status` | GET | System status: running, last trigger, cooldown state |

---

## Dependencies

```
# requirements.txt
inference-sdk          # Roboflow Inference SDK
opencv-python-headless # RTSP capture, frame saving
gpiozero               # GPIO relay control
RPi.GPIO               # gpiozero backend on Pi
fastapi                # Phase 4 API
uvicorn                # Phase 4 ASGI server
pyyaml                 # Config loading
requests               # Nightly AI batch (vision API calls)
```
