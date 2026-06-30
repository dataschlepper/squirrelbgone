# SquirrelBGone

Real-time computer vision squirrel deterrent. A Raspberry Pi 5 pulls an RTSP stream from an outdoor PoE camera, runs YOLOv8 inference, and triggers a 12V solenoid valve to spray water when a squirrel is detected. Birds and other classes are suppressed.

## Build phases

| Phase | Description | Status |
|---|---|---|
| 1 | Detection only — software setup, RTSP stream, inference loop, CSV logging | ✅ |
| 2 | GPIO dry run — LED stand-in, cooldown timer, day/night guard | ✅ |
| 3 | Hardware integration — relay, solenoid, water, weatherproof enclosure | ✅ |
| 4 | Polish — FastAPI dashboard, AI-assisted threshold tuning, per-class logic | 🔄 In progress |

See [`docs/phases.md`](docs/phases.md) for the full task breakdown.

## How it works

1. Reolink RLC-811A camera streams RTSP video over the local network
2. Raspberry Pi 5 pulls the stream and runs YOLOv8 inference locally via ultralytics
3. Squirrel detections above the confidence threshold are evaluated against a configurable feeder zone — only squirrels on the feeder trigger the sprayer
4. Optional suppression models (bird, wildlife) prevent false triggers from non-targets
5. A FastAPI dashboard on the Pi lets you review detections, define the feeder zone, and manually test the sprayer from your phone
6. A GPIO pin fires a relay → opens a 12V solenoid valve → water spray

## Software stack

| Component | Role |
|---|---|
| Raspberry Pi OS | Host OS |
| Python + ultralytics | YOLOv8 inference (squirrel model + optional suppression models) |
| OpenCV | RTSP stream capture, frame saving |
| FastAPI + uvicorn | Read-only detection dashboard (live) |
| gpiozero | GPIO relay control (Phase 2+) |

## Project structure

```
squirrelbgone/
├── detect.py            # Inference loop, GPIO, CSV logging
├── .env.example         # Config template (copy to .env)
├── models/              # YOLOv8 .pt weight files
├── api/
│   └── server.py        # FastAPI dashboard + MJPEG stream
├── systemd/             # systemd service units (auto-start on boot)
├── logs/                # Daily detection CSVs + runtime state files
├── frames/              # Saved JPEG frames (auto-purged per FRAMES_KEEP_DAYS)
└── docs/
    ├── hardware.md       # Wiring, components, key design decisions
    ├── network.md        # Topology, RTSP setup, PoE notes
    ├── phases.md         # 4-phase build plan with task breakdown
    ├── software.md       # Architecture, schemas, API endpoints
    └── shopping-list.md  # Parts list with prices and links
```

## Budget

~$455 total. See [`docs/shopping-list.md`](docs/shopping-list.md) for the full parts list.

| Category | Est. cost |
|---|---|
| Core electronics | ~$215–255 |
| Chassis & enclosure | ~$42–50 |
| Weatherproofing & wiring | ~$24–32 |
| Plumbing | ~$10–15 |
| Tools | ~$128–150 |
