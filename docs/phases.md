# Build Phases

The project is structured into four incremental phases. Each phase is fully testable before moving to the next — no wiring water until Phase 3, no water until GPIO logic is validated, etc.

---

## Phase 1 — Detection only

**Goal:** Get inference running on the live RTSP stream with acceptable accuracy. No hardware beyond the camera.

| Task | Type |
|---|---|
| Flash Raspberry Pi OS, install Python deps, configure SSH | software |
| Mount camera, confirm RTSP stream accessible on local network | hardware |
| Pull squirrel/bird YOLOv8 model from Roboflow Universe via Inference SDK | software |
| Run inference loop on live stream, print detections + confidence | software |
| Log detections to CSV with timestamp, class, confidence, saved frame path | software |
| Review logs, tune confidence threshold until false positive rate is acceptable | software |

**Exit criteria:** Inference loop running stably, squirrels detected reliably, false positive rate acceptable in logs.

---

## Phase 2 — GPIO dry run

**Goal:** Validate the full detection → trigger pipeline using an LED as a stand-in for the solenoid. No water involved.

| Task | Type | Status |
|---|---|---|
| Wire LED to GPIO pin on breadboard with current-limiting resistor | hardware | in progress |
| Wire detection → GPIO trigger logic using gpiozero | software | done |
| Implement cooldown timer (prevent re-trigger every frame) | software | done |
| Implement day/night schedule guard (no triggers at night) | software | done |
| Confirm LED flashes correctly on squirrel detection, stays off for birds | hardware | — |

**Exit criteria:** LED fires cleanly and only when expected. Cooldown and schedule guard both confirmed working.

---

## Phase 3 — Hardware integration

**Goal:** Replace LED with real relay + solenoid + water. Get the system running outdoors in its final enclosure.

| Task | Type |
|---|---|
| Wire relay module: separate 5V supply, shared ground with Pi, GPIO signal line | hardware |
| Wire NC solenoid to relay NO/COM terminals and 12V supply | hardware |
| Connect solenoid to hose line with Teflon-taped fittings + misting nozzle | hardware |
| Test water flow: confirm valve opens/closes on trigger signal | hardware |
| Set spray duration to 1–1.5 sec, confidence threshold to 0.7+ | software |
| Log all triggers with saved frames for first-week review | software |
| Mount Pi + relay in weatherproof IP65 enclosure with cable glands | hardware |

**Key parameters at this stage:**
- Spray duration: 1–1.5 seconds per trigger
- Confidence threshold: 0.7 or higher
- Cooldown: set to prevent re-trigger within N seconds of last spray

**Exit criteria:** Solenoid fires and cuts reliably on detection. System running outdoors, sealed, logged. One week of trigger frames collected for review.

---

## Phase 4 — Polish

**Goal:** Narrow triggers to feeder-only, train a camera-specific model to hit 90%+ precision, and automate ongoing improvement with AI-assisted labeling.

### Group A — Zone-based triggering *(immediate fix, independent of model quality)*

| Task | Type | Status |
|---|---|---|
| Add feeder zone picker to dashboard (drag to draw rectangle, save coords) | software | ✅ |
| Persist zone to `logs/feeder_zone.json` (as 0–1 fractions, resolution-independent) | software | ✅ |
| Update `detect.py` to skip trigger if squirrel center is outside feeder zone; log as `triggered=False` with reason `"outside feeder zone"` | software | ✅ |
| Show zone overlay on detection cards in dashboard | software | ✅ |

### Group B — AI-assisted pre-labeling script *(unlocks Group C without manual sifting)*

| Task | Type | Status |
|---|---|---|
| Write script that sends frames to Claude Vision and returns `squirrel / not-squirrel / uncertain` + explanation | AI | — |
| Output Roboflow-compatible labels (or CSV mapping frame → verdict) | software | — |
| Run on existing `frames/` archive to bootstrap training dataset | AI | — |
| Run nightly on new detections to keep dataset growing | AI | — |

### Group C — Custom model training *(core accuracy fix — target <5% FP in feeder zone)*

| Task | Type | Status |
|---|---|---|
| Set up Roboflow dataset, import pre-labeled frames from Group B | software | — |
| Human review pass — correct AI labels, add bboxes to true positives | manual | — |
| Fine-tune from existing `squirrelbgone_best.pt` weights (not from scratch) | ML | — |
| Evaluate: measure precision/recall on held-out set | ML | — |
| Deploy: swap `MODEL_PATH` in `.env`, monitor first week | software | — |

### Group D — Ongoing iteration loop

| Task | Type | Status |
|---|---|---|
| Nightly script flags uncertain detections for human review | AI | — |
| Monthly retrain cycle as labeled dataset grows | ML | — |
| Tune `SPRAY_CONFIDENCE_THRESHOLD` based on observed precision per confidence band | manual | — |

**Config file should expose at minimum:**
- `confidence_threshold` — minimum confidence to trigger spray
- `spray_duration_sec` — how long to open the solenoid
- `cooldown_sec` — minimum time between spray events
- `day_start` / `day_end` — schedule guard hours
- `camera_rtsp_url` — RTSP stream URL
- `gpio_pin` — relay control pin

**AI review loop (nightly batch):**
1. Collect frames flagged as low-confidence or misclassified during the day
2. Send to Claude or GPT-4V with a classification prompt
3. Parse results into a correction log
4. Use correction log to manually or automatically adjust thresholds
5. Repeat weekly until false positive/negative rate is at target

---

## Tunable parameters reference

| Parameter | Phase set | Notes |
|---|---|---|
| Confidence threshold | Phase 1 (initial), Phase 3 (tightened), Phase 4 (AI-tuned) | Start permissive, tighten over time |
| Spray duration | Phase 3 | 1–1.5 sec recommended starting point |
| Cooldown timer | Phase 2 | Prevents spray every frame during a single detection event |
| Day/night schedule | Phase 2 (basic), Phase 4 (hardened) | No triggers at night |
| Per-class suppression | Phase 4 | Birds, shadows, other non-squirrel classes suppressed |

---

## Backlog

Improvements that were scoped out or deferred. No phase assignment yet.

| Item | Notes |
|---|---|
| Real-time bounding box overlay on live feed | Current dashboard draws boxes on saved frames only. Live MJPEG stream has no overlay. Would require detect.py to publish current bbox coords (e.g. via a shared file or memory) so the API thread can composite them onto stream frames before encoding. |
