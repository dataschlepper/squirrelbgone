# CLAUDE.md

Project context for Claude Code. Read this first. Full detail in `docs/`.

---

## What this is

**SquirrelBGone** — a Raspberry Pi 5 pulls an RTSP stream from an outdoor PoE camera, runs YOLOv8 inference locally via **ultralytics**, and triggers a 12V normally-closed solenoid valve (via a GPIO-controlled relay) to spray water when a squirrel is detected. Suppression models (bird, wildlife) can be layered on to prevent false triggers. Everything is logged with saved frames.

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
YOLOv8 inference — up to 3 models per frame:
  ├── Squirrel model (ultralytics, required)
  ├── Bird suppression model (ultralytics, optional)
  └── Wildlife suppression model (ultralytics, optional — backlog)
    │
    ▼
Any squirrel detections above threshold?
    │
    ├── No ──► if bird/wildlife detected, log those; skip frame
    │
    └── Yes ──► bird or wildlife also in frame?
                    │
                    ├── Yes ──► log squirrel as suppressed (triggered=False)
                    │
                    └── No ──► log squirrel (triggered=True)
                               save frame
                               [Phase 2+: GPIO pulse → relay → solenoid]
```

Full details: `docs/software.md`

---

## Repo structure

```
squirrelbgone/
├── CLAUDE.md                # ← you are here
├── README.md
├── detect.py                # main entry point — inference loop, CSV logging
├── .env                     # gitignored — copy from .env.example
├── .env.example             # all config vars with comments
├── models/
│   └── squirrelbgone_best.pt  # squirrel detector weights
├── api/
│   └── server.py            # FastAPI read-only dashboard (live)
├── logs/                    # daily CSVs: detections_YYYY-MM-DD.csv
├── frames/                  # saved JPEG frames for each detection
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

All config is via `.env` (copy `.env.example`):

```
RTSP_URL                     # RTSP stream URL
MODEL_PATH                   # path to squirrel .pt weights (required)
CONFIDENCE_THRESHOLD         # squirrel min confidence, default 0.45
MODEL_PATH_BIRD              # bird suppression model (optional, leave empty to disable)
BIRD_CONFIDENCE_THRESHOLD    # default 0.45
MODEL_PATH_WILDLIFE          # wildlife suppression model (optional — backlog)
WILDLIFE_CONFIDENCE_THRESHOLD  # default 0.45
TARGET_FPS                   # frames to process per second, default 5
LOG_DIR                      # default "logs"
FRAMES_DIR                   # default "frames"
ROBOFLOW_API_KEY             # needed only to download models from Roboflow Universe
```

Full config with comments: `.env.example`

---

## Log schemas

**`logs/detections_YYYY-MM-DD.csv`** — every detection (squirrel, bird, wildlife), one file per day, rolls over at midnight:
`timestamp, class, confidence, triggered, x1, y1, w, h, frame_path`

- `triggered=True` means the squirrel detection was not suppressed (Phase 2+: this fires the relay)
- `triggered=False` for all bird/wildlife rows, and for squirrels suppressed by a co-present bird/wildlife

**`logs/corrections.csv`** — user-flagged false positives from the dashboard:
`flagged_at, detection_timestamp, class, confidence, frame_path`

---

## Dependencies

```
ultralytics          # YOLOv8 inference (squirrel + optional suppression models)
opencv-python        # RTSP capture, frame saving
python-dotenv        # .env config loading
fastapi              # dashboard API (live)
uvicorn              # ASGI server (live)
gpiozero             # GPIO relay control (Phase 2+)
RPi.GPIO             # gpiozero backend on Pi (Phase 2+)
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
