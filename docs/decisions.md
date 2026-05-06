# Design Decisions

Key architectural and hardware decisions, with rationale. Useful reference if revisiting choices later.

---

## Hardware

### Normally-closed (NC) solenoid vs normally-open (NO)

**Decision:** NC solenoid (valve shut by default, opens when energized).

**Rationale:** Safe failure mode. If the Pi crashes, loses power, or the relay de-energizes for any reason, the valve closes automatically. An NO solenoid would leave water running indefinitely on any failure. For an unattended outdoor system, NC is the only reasonable choice.

---

### Direct-acting solenoid vs pilot-operated

**Decision:** Direct-acting (Wengart 2W-160-15).

**Rationale:** Pilot-operated solenoids require a minimum differential pressure (typically 0.5–1 bar) to open. A garden hose at low pressure or a gravity-fed setup may fall below this threshold. Direct-acting valves work from 0 PSI — no minimum pressure required. Slightly more expensive and less flow capacity at high pressure, but the reliability tradeoff is worth it for this application.

---

### Separate 5V supply for relay vs powering from Pi GPIO rail

**Decision:** Dedicated 5V wall adapter for the relay module, shared ground with Pi.

**Rationale:** The JBtek relay module draws significantly more current than the Pi's GPIO rail is rated to supply. Powering the relay from the GPIO rail causes voltage drops that destabilize the Pi and can corrupt SD card writes. Separate supply eliminates this entirely. Shared ground is required for the GPIO signal line to work correctly.

---

### PoE camera vs WiFi camera

**Decision:** PoE camera (Reolink RLC-811A with bundled PoE injector).

**Rationale:**
- Single Cat6 cable handles both power and video — only one cable run to the outdoor location
- No WiFi dead zones, interference, or dropped frames from signal issues
- PoE cameras are generally more reliable for continuous streaming than WiFi models
- IP67 rated — fully weatherproof
- RTSP support is standard and well-documented on Reolink cameras

Tradeoff: requires running Cat6 to the camera location. For most garden/yard setups this is straightforward.

---

### 4K camera vs lower resolution

**Decision:** 4K (RLC-811A) with 5× optical zoom.

**Rationale:** Squirrel detection at 15–20ft requires enough resolution to capture a small subject clearly. 4K + optical zoom allows the camera to be mounted further back while still giving YOLOv8 enough pixels to classify correctly. A 1080p camera at the same distance may produce detections but with lower confidence and more false positives. Inference runs on downscaled frames anyway, so the full 4K is mainly used for the optical zoom advantage.

---

## Software

### Roboflow Inference SDK vs running a local model directly

**Decision:** Roboflow Inference SDK (local inference server on Pi).

**Rationale:**
- Pre-trained squirrel/bird models available on Roboflow Universe — avoids collecting and labeling a custom dataset from scratch
- Inference SDK runs locally on the Pi (no cloud dependency in steady state)
- Easy model swapping: change `model_id` in config to try different models
- API key required for model download but not for inference once cached

Tradeoff: locked into Roboflow's SDK and model format. If switching to a custom-trained model later, the `detector.py` wrapper can be swapped out without touching the rest of the pipeline.

---

### Two separate log files (detections + triggers) vs one

**Decision:** `detections.csv` for all inference results, `triggers.csv` for spray events only.

**Rationale:** Detections are high-frequency (every frame with a detection above any threshold). Triggers are low-frequency (only when the solenoid fires). Keeping them separate makes it easy to query spray history without scanning the full detection log, and keeps file sizes manageable for the nightly AI review script which only needs trigger frames.

---

### Nightly AI batch review vs real-time AI classification

**Decision:** Nightly batch (cron job), not real-time.

**Rationale:**
- Real-time vision API calls introduce latency and cost per-frame — not suitable for a fast inference loop
- The nightly batch only processes flagged/low-confidence frames, keeping API costs minimal
- Corrections are used to tune thresholds over time, not to make per-frame decisions
- The primary classifier is always YOLOv8 running locally; the AI review is a tuning mechanism

---

### Config file (YAML) vs environment variables vs hardcoded constants

**Decision:** Single `config.yaml` file for all tunable parameters.

**Rationale:** Parameters like confidence threshold, spray duration, cooldown, and schedule need to be adjusted frequently during the tuning phase without touching code. A single YAML file is easy to edit on the Pi over SSH. API keys in the config file are gitignored. Environment variables are an acceptable alternative but less convenient for the Pi + SSH workflow.

---

## Build process

### 4-phase incremental build vs full build from the start

**Decision:** 4 phases: detection only → GPIO dry run → hardware integration → polish.

**Rationale:** Water + electronics + outdoors is a high-risk combination. Each phase is independently testable and verifiable before the next phase adds risk. Specifically:
- Phase 1 ensures the model works before any hardware is wired
- Phase 2 ensures GPIO logic is correct before any relay or solenoid is connected
- Phase 3 ensures water flow is controlled before the system is left unattended
- Phase 4 adds monitoring and tuning once the core system is proven

Skipping phases (e.g. going straight to solenoid wiring) risks water damage, burned GPIO pins, or a spray system that fires constantly due to untuned thresholds.
