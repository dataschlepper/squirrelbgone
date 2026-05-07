# CLAUDE.md

Project context for Claude Code. Read this first. Full detail in `docs/`.

---

## What this is

**SquirrelBGone** — a Raspberry Pi 5 pulls an RTSP stream from an outdoor PoE camera, runs YOLOv8 inference via the Roboflow Inference SDK, and triggers a 12V normally-closed solenoid valve (via a GPIO-controlled relay) to spray water when a squirrel is detected. Birds and other classes are suppressed. Everything is logged with saved frames.

---

## Current phase

> **Update this section as phases are completed.**

- [ ] Phase 1 — Detection only (software setup, RTSP stream, inference loop, CSV logging)
- [ ] Phase 2 — GPIO dry run (LED stand-in, cooldown timer, day/night guard)
- [ ] Phase 3 — Hardware integration (relay, solenoid, water, outdoor enclosure)
- [ ] Phase 4 — Polish (FastAPI dashboard, nightly AI review, per-class logic, config hardening)

**Currently working on:** Phase 1

---

## Hardware (brief)

| Component | Detail |
|---|---|
| Raspberry Pi 5 4GB | Inference, GPIO, logging |
| Reolink RLC-811A | 4K PoE camera, IP67, RTSP, 5× optical zoom |
| JBtek 5V 4-ch relay | Active-low, **separate 5V supply** (not Pi GPIO rail) |
| Wengart 2W-160-15 | 12V NC solenoid, direct-acting, 0 PSI minimum, no polarity |
| IP65 enclosure | Houses Pi + relay outdoors |

Full wiring details: `docs/hardware.md`

---

## Software pipeline

```
RTSP stream
    │
    ▼
Frame capture (OpenCV)
    │
    ▼
YOLOv8 inference (Roboflow Inference SDK)
    │
    ▼
Filter: class == "squirrel" AND confidence >= threshold?
    │
    ├── No ──► log detection, continue
    │
    └── Yes ──► cooldown clear? ──► No: skip
                    │
                    └── Yes ──► daytime? ──► No: skip
                                    │
                                    └── Yes ──► GPIO pulse (relay → solenoid)
                                                    │
                                                save frame + write trigger log
```

Full module breakdown + code sketches: `docs/software.md`

---

## Repo structure

```
squirrelbgone/
├── CLAUDE.md                # ← you are here
├── README.md
├── config.yaml              # gitignored — copy from config.example.yaml
├── config.example.yaml      # all tunable params with comments
├── main.py                  # entry point
├── config.py                # loads + validates config.yaml
├── inference/
│   ├── stream.py            # RTSP capture via OpenCV
│   └── detector.py          # Roboflow Inference SDK wrapper
├── trigger/
│   ├── cooldown.py          # cooldown timer
│   ├── schedule.py          # day/night guard
│   └── gpio_relay.py        # gpiozero relay pulse
├── logging/
│   ├── detection_log.py     # writes detections.csv
│   └── trigger_log.py       # writes triggers.csv + saves frames
├── scripts/
│   └── nightly_review.py    # Phase 4: AI batch frame review (cron)
├── api/                     # Phase 4: FastAPI backend
└── docs/
    ├── hardware.md
    ├── network.md
    ├── phases.md
    ├── software.md
    ├── shopping-list.md
    └── decisions.md
```

---

## Key gotchas

**Relay is active-low.** The JBtek relay module energizes when the GPIO pin goes LOW. Use `active_high=False` in gpiozero. Verify with the Phase 2 LED dry run before connecting the solenoid.

**Relay needs a separate 5V supply.** Never power the relay from the Pi GPIO rail — it draws too much current and will destabilize the Pi. Shared ground with Pi is required for the signal line to work.

**Solenoid is NC (normally-closed).** The valve is shut by default and opens when energized. On any failure or power loss, water stops. Do not swap for an NO solenoid.

**Solenoid is direct-acting.** Works from 0 PSI — no minimum water pressure required.

**Cooldown timer is mandatory.** Without it, a squirrel sitting in frame triggers the solenoid on every inference pass. Set `cooldown_sec` in config and enforce it before any GPIO pulse.

**Day/night guard prevents nighttime false triggers.** Check `schedule.day_start` / `schedule.day_end` before every trigger.

**Config is gitignored.** `config.yaml` contains API keys — never commit it. The template is `config.example.yaml`.

---

## Config keys (quick ref)

```yaml
camera.rtsp_url                  # RTSP stream URL
inference.roboflow_api_key        # Roboflow API key
inference.roboflow_model_id       # e.g. "squirrels-birds/1"
inference.confidence_threshold    # min confidence to trigger (start ~0.70)
inference.review_threshold        # below this → flagged for AI review (0.85)
classes.trigger                   # ["squirrel"]
classes.suppress                  # ["bird", "shadow"]
gpio.pin                          # BCM pin number
gpio.active_high                  # false (JBtek relay is active-low)
spray.duration_sec                # 1.5 recommended
spray.cooldown_sec                # 30 recommended
schedule.day_start                # [7, 0]
schedule.day_end                  # [20, 0]
logging.detection_log             # "logs/detections.csv"
logging.trigger_log               # "logs/triggers.csv"
logging.frame_dir                 # "frames/"
```

Full config with comments: `config.example.yaml`

---

## Log schemas

**`logs/detections_YYYY-MM-DD.csv`** — every inference result (one file per day, rolls over at midnight):
`timestamp, class, confidence, triggered, frame_path, bbox_x, bbox_y, bbox_w, bbox_h`

**`logs/triggers.csv`** — spray events only:
`timestamp, class, confidence, spray_duration_sec, frame_path`

---

## Dependencies

```
inference-sdk              # Roboflow Inference SDK
opencv-python-headless     # RTSP capture, frame saving
gpiozero                   # GPIO relay control
RPi.GPIO                   # gpiozero backend on Pi
fastapi                    # Phase 4 API
uvicorn                    # Phase 4 ASGI server
pyyaml                     # config loading
requests                   # nightly AI batch (vision API calls)
```

---

## Docs index

| File | Contents |
|---|---|
| `docs/hardware.md` | Full component list, wiring diagram, design decisions, phase checklists |
| `docs/network.md` | Network topology, RTSP URL format, static IP notes, security |
| `docs/phases.md` | Task breakdown per phase, exit criteria, tunable params reference |
| `docs/software.md` | Module structure, code sketches, log schemas, FastAPI endpoints, AI review flow |
| `docs/shopping-list.md` | Full parts list with prices and purchase links |
| `docs/decisions.md` | Rationale for every major hardware and software decision |
