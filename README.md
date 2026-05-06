# SquirrelBGone

Real-time computer vision squirrel deterrent. A Raspberry Pi 5 pulls an RTSP stream from an outdoor PoE camera, runs YOLOv8 inference, and triggers a 12V solenoid valve to spray water when a squirrel is detected. Birds and other classes are suppressed.

## How it works

1. Reolink RLC-811A camera streams RTSP video over the local network
2. Raspberry Pi 5 pulls the stream and runs YOLOv8 inference via the Roboflow Inference SDK
3. On a squirrel detection above the confidence threshold, a GPIO pin fires a relay
4. The relay opens a normally-closed 12V solenoid valve, triggering a water misting nozzle
5. A cooldown timer and day/night schedule guard prevent spurious triggers
6. All detections and spray events are logged with saved frames for review

## Software stack

| Component | Role |
|---|---|
| Raspberry Pi OS | Host OS |
| Python | Application runtime |
| Roboflow Inference SDK | YOLOv8 model serving (squirrel/bird model from Roboflow Universe) |
| gpiozero | GPIO relay control |
| FastAPI | Backend API — live feed, detection log, spray history (Phase 4) |
| Claude / GPT-4V | Nightly batch frame review for threshold tuning (Phase 4) |

## Project structure

```
squirrelbgone/
├── README.md
├── docs/
│   ├── hardware.md        # Wiring, components, key design decisions
│   ├── network.md         # Topology, RTSP setup, PoE notes
│   ├── phases.md          # 4-phase build plan with task breakdown
│   └── shopping-list.md   # Parts list with prices and links
├── inference/             # YOLOv8 inference loop, logging
├── gpio/                  # Relay control, cooldown, schedule guard
├── api/                   # FastAPI backend (Phase 4)
└── dashboard/             # Frontend with bounding box overlay (Phase 4)
```

## Build phases

| Phase | Description | Status |
|---|---|---|
| 1 | Detection only — software setup, RTSP stream, inference loop, CSV logging | — |
| 2 | GPIO dry run — LED stand-in, cooldown timer, day/night guard | — |
| 3 | Hardware integration — relay, solenoid, water, weatherproof enclosure | — |
| 4 | Polish — FastAPI dashboard, AI-assisted threshold tuning, per-class logic | — |

See [`docs/phases.md`](docs/phases.md) for the full task breakdown.

## Budget

~$455 total. See [`docs/shopping-list.md`](docs/shopping-list.md) for the full parts list.

| Category | Est. cost |
|---|---|
| Core electronics | ~$215–255 |
| Chassis & enclosure | ~$42–50 |
| Weatherproofing & wiring | ~$24–32 |
| Plumbing | ~$10–15 |
| Tools | ~$128–150 |
