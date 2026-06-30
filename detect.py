#!/usr/bin/env python3
"""
SquirrelBGone — detect.py
Phase 1: Detection only. Reads RTSP stream, runs YOLOv8 inference locally,
prints detections + logs to CSV with saved frames.

Runs up to three models per frame:
  - Primary squirrel model (MODEL_PATH) — single-class squirrel detector
  - Bird suppression model (MODEL_PATH_BIRD, default yolov8n.pt) — COCO model;
    if a bird is present in the same frame as a squirrel, the squirrel trigger
    is suppressed (both are still logged)
  - Wildlife suppression model (MODEL_PATH_WILDLIFE, optional) — e.g. a Roboflow
    backyard-wildlife model; if deer/fox/raccoon/etc. are present, the squirrel
    trigger is suppressed and the wildlife class is logged

Requirements:
    pip install ultralytics opencv-python python-dotenv

Usage:
    cp .env.example .env && nano .env
    python detect.py
"""

import csv
import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RTSP_URL                  = os.environ.get("RTSP_URL", "")
MODEL_PATH                = os.environ.get("MODEL_PATH", "squirrel_detector.pt")
MODEL_PATH_BIRD           = os.environ.get("MODEL_PATH_BIRD", "")
TARGET_FPS                = int(os.environ.get("TARGET_FPS", "5"))
CONFIDENCE_THRESHOLD         = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
BIRD_CONFIDENCE_THRESHOLD    = float(os.environ.get("BIRD_CONFIDENCE_THRESHOLD", "0.45"))
MODEL_PATH_WILDLIFE          = os.environ.get("MODEL_PATH_WILDLIFE", "")
WILDLIFE_CONFIDENCE_THRESHOLD = float(os.environ.get("WILDLIFE_CONFIDENCE_THRESHOLD", "0.45"))

# Where to write logs and saved frames
LOG_DIR         = Path(os.environ.get("LOG_DIR",    "logs"))
FRAMES_DIR      = Path(os.environ.get("FRAMES_DIR", "frames"))
FRAMES_KEEP_DAYS = int(os.environ.get("FRAMES_KEEP_DAYS", "7"))

# Written by api/server.py for dashboard controls
SPRAY_REQUEST_FILE   = LOG_DIR / "spray.request"
SOLENOID_STATE_FILE  = LOG_DIR / "solenoid.state"
ZONE_FILE            = LOG_DIR / "feeder_zone.json"

# Phase 2+: GPIO trigger
GPIO_PIN           = int(os.environ.get("GPIO_PIN", "17"))
COOLDOWN_SEC       = float(os.environ.get("COOLDOWN_SEC", "10"))
DAY_START          = int(os.environ.get("DAY_START", "7"))   # hour 0–23, inclusive
DAY_END            = int(os.environ.get("DAY_END", "20"))    # hour 0–23, exclusive
SPRAY_DURATION_SEC        = float(os.environ.get("SPRAY_DURATION_SEC", "1.0"))
SPRAY_CONFIDENCE_THRESHOLD = float(os.environ.get("SPRAY_CONFIDENCE_THRESHOLD", "0.80"))
SPRAY_ENABLED             = os.environ.get("AUTO_SPRAY_ENABLED", "false").lower() == "true"

# Classes that should trigger the sprayer (Phase 2+)
TRIGGER_CLASSES = {"squirrel"}

# Classes we want logged but never triggered on
BENIGN_CLASSES  = {"bird", "crow", "pigeon", "robin", "sparrow"}

# Wildlife model classes that suppress a squirrel trigger
WILDLIFE_SUPPRESS_CLASSES = {
    "deer", "fawn", "buck", "doe",
    "fox", "raccoon", "rabbit", "hog", "boar",
    "bear", "coyote", "skunk", "opossum", "groundhog", "turkey",
}

# ─── FRAME CLEANUP ────────────────────────────────────────────────────────────

def _cleanup_old_frames() -> None:
    cutoff = datetime.datetime.now() - datetime.timedelta(days=FRAMES_KEEP_DAYS)
    deleted = 0
    for f in FRAMES_DIR.glob("*.jpg"):
        try:
            if datetime.datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                f.unlink()
                deleted += 1
        except Exception:
            pass
    if deleted:
        log.info(f"Cleanup: removed {deleted} frames older than {FRAMES_KEEP_DAYS}d.")


# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sbg")

# ─── GPIO ─────────────────────────────────────────────────────────────────────
#
# Control scheme (verified on hardware):
#   OFF / valve closed  →  GPIO17 = input, no pull  (external 5V pull-up holds relay IN high)
#   ON  / valve open    →  GPIO17 = output, drive low  (pulls relay IN to 0V, energises relay)
#
# Never drive the pin high to turn off: Pi 3.3V logic-high can't release a relay
# with a 5V pull-up on IN.  "Off" means releasing the pin to input.

_gpio_available = False


def _gpio_release() -> None:
    """Set GPIO17 to input/no-pull — external pull-up closes the relay (fail-safe off)."""
    subprocess.run(
        ["pinctrl", "set", str(GPIO_PIN), "ip", "pn"],
        check=False, capture_output=True,
    )


def _gpio_drive_low() -> None:
    """Drive GPIO17 output low — pulls relay IN to 0V, energises relay, opens valve."""
    subprocess.run(
        ["pinctrl", "set", str(GPIO_PIN), "op", "dl"],
        check=False, capture_output=True,
    )


def _setup_gpio() -> None:
    global _gpio_available
    try:
        subprocess.run(
            ["pinctrl", "set", str(GPIO_PIN), "ip", "pn"],
            check=True, capture_output=True,
        )
        _gpio_available = True
        log.info(f"GPIO ready: relay on pin {GPIO_PIN} (pinctrl, active-low)")
    except Exception as exc:
        log.warning(f"GPIO unavailable ({exc}) — running without hardware output")


_solenoid_toggled_on = False


def _fire_gpio(duration: float | None = None) -> None:
    """Open valve for `duration` seconds, then release. try/finally ensures valve closes on crash."""
    if not _gpio_available:
        return
    dur = duration or SPRAY_DURATION_SEC
    try:
        _gpio_drive_low()
        time.sleep(dur)
    finally:
        _gpio_release()


def _check_spray_request() -> None:
    global _solenoid_toggled_on

    # Toggle state — persist until changed
    if SOLENOID_STATE_FILE.exists():
        try:
            data = json.loads(SOLENOID_STATE_FILE.read_text())
            want_on = bool(data.get("on", False))
            if want_on != _solenoid_toggled_on:
                _solenoid_toggled_on = want_on
                if _gpio_available:
                    _gpio_drive_low() if want_on else _gpio_release()
                log.info(f"Solenoid toggled {'ON' if want_on else 'OFF'} (dashboard)")
        except Exception:
            pass

    # Timed pulse — ignored while toggled on
    if not _solenoid_toggled_on and SPRAY_REQUEST_FILE.exists():
        try:
            data = json.loads(SPRAY_REQUEST_FILE.read_text())
            duration = float(data.get("duration", SPRAY_DURATION_SEC))
        except Exception:
            duration = SPRAY_DURATION_SEC
        try:
            SPRAY_REQUEST_FILE.unlink()
        except Exception:
            pass
        _fire_gpio(duration)
        log.info(f"Manual spray {duration}s (requested via dashboard)")


def _cleanup_gpio() -> None:
    """Ensure valve is closed on exit — safe to call even if GPIO never initialised."""
    _gpio_release()

# ─── FEEDER ZONE ──────────────────────────────────────────────────────────────
#
# Zone is stored as fractions (0–1) so it's resolution-independent.
# Reloaded whenever the file changes (set via dashboard zone picker).

_feeder_zone: dict | None = None
_feeder_zone_mtime: float = 0.0


def _reload_zone_if_changed() -> None:
    global _feeder_zone, _feeder_zone_mtime
    try:
        mtime = ZONE_FILE.stat().st_mtime
        if mtime != _feeder_zone_mtime:
            _feeder_zone = json.loads(ZONE_FILE.read_text())
            _feeder_zone_mtime = mtime
            log.info(f"Feeder zone loaded: {_feeder_zone}")
    except FileNotFoundError:
        if _feeder_zone is not None:
            log.info("Feeder zone cleared.")
        _feeder_zone = None
        _feeder_zone_mtime = 0.0
    except Exception:
        pass


def _in_feeder_zone(x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> bool:
    """Returns True if the bbox center falls inside the configured feeder zone.
    Returns True unconditionally when no zone is set (trigger everywhere)."""
    if not _feeder_zone:
        return True
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (
        _feeder_zone["x1"] * frame_w <= cx <= _feeder_zone["x2"] * frame_w
        and _feeder_zone["y1"] * frame_h <= cy <= _feeder_zone["y2"] * frame_h
    )


# ─── SHUTDOWN ─────────────────────────────────────────────────────────────────

_running = True

def _shutdown(sig, frame):
    global _running
    log.info("Interrupt received — shutting down.")
    _running = False

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# ─── CSV ──────────────────────────────────────────────────────────────────────

CSV_FIELDS = ["timestamp", "class", "confidence", "triggered", "x1", "y1", "w", "h", "frame_path"]

def open_csv(log_dir: Path):
    """Open (or append to) today's detection log. Returns (file_handle, csv_writer).
    If an existing file has a mismatched header, archives it and starts fresh."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    csv_path = log_dir / f"detections_{date_str}.csv"

    if csv_path.exists():
        with open(csv_path, newline="") as chk:
            existing_header = next(csv.reader(chk), [])
        if existing_header != CSV_FIELDS:
            archive = csv_path.with_name(f"detections_{date_str}_legacy_{int(time.time())}.csv")
            csv_path.rename(archive)
            log.warning(f"CSV schema changed — archived old log to {archive.name}")
            is_new = True
        else:
            is_new = False
    else:
        is_new = True

    fh = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if is_new:
        writer.writeheader()
    log.info(f"Logging detections to: {csv_path}")
    return fh, writer

# ─── STREAM ───────────────────────────────────────────────────────────────────

_STREAM_PARAMS = [
    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000,
    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000,
]

def _open_stream(rtsp_url: str) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG, _STREAM_PARAMS)
    return cap if cap.isOpened() else None


def _reconnect(rtsp_url: str, attempt: int) -> tuple[cv2.VideoCapture | None, int]:
    """Exponential backoff reconnect. Returns (cap_or_None, next_attempt)."""
    delay = min(3 * 2 ** attempt, 30)
    log.warning(f"Stream lost — reconnect attempt {attempt + 1} in {delay}s…")
    time.sleep(delay)
    cap = _open_stream(rtsp_url)
    if cap:
        log.info("Stream reconnected.")
        return cap, 0
    return None, attempt + 1


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if not RTSP_URL:
        log.error("RTSP_URL is not set. Copy .env.example → .env and fill it in.")
        sys.exit(1)

    if not Path(MODEL_PATH).exists():
        log.error(
            f"Model weights not found at '{MODEL_PATH}'.\n"
            "  Set MODEL_PATH in .env to point at your .pt file."
        )
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_frames()

    from ultralytics import YOLO

    log.info(f"Loading squirrel model: {MODEL_PATH}")
    squirrel_model = YOLO(MODEL_PATH)
    log.info("Squirrel model loaded.")

    bird_model = None
    if MODEL_PATH_BIRD:
        if not Path(MODEL_PATH_BIRD).exists():
            log.warning(f"Bird model not found at '{MODEL_PATH_BIRD}' — bird suppression disabled.")
        else:
            log.info(f"Loading bird model: {MODEL_PATH_BIRD}")
            bird_model = YOLO(MODEL_PATH_BIRD)
            log.info("Bird model loaded.")

    wildlife_model = None
    if MODEL_PATH_WILDLIFE:
        if not Path(MODEL_PATH_WILDLIFE).exists():
            log.warning(f"Wildlife model not found at '{MODEL_PATH_WILDLIFE}' — wildlife suppression disabled.")
        else:
            log.info(f"Loading wildlife model: {MODEL_PATH_WILDLIFE}")
            wildlife_model = YOLO(MODEL_PATH_WILDLIFE)
            log.info("Wildlife model loaded.")

    csv_fh, csv_writer = open_csv(LOG_DIR)

    _setup_gpio()

    log.info(f"Connecting to stream: {RTSP_URL}")
    cap = _open_stream(RTSP_URL)
    if cap is None:
        log.error(
            "Could not open RTSP stream. Check:\n"
            "  1. Camera IP and credentials in RTSP_URL\n"
            "  2. Camera is on the same LAN as the Pi\n"
            "  3. Test with: ffprobe <RTSP_URL>"
        )
        sys.exit(1)

    log.info(
        f"Stream open. {TARGET_FPS}fps · "
        f"squirrel threshold={CONFIDENCE_THRESHOLD} · "
        f"bird threshold={BIRD_CONFIDENCE_THRESHOLD}"
    )
    log.info("Watching… (Ctrl+C to stop)\n")

    frame_interval    = 1.0 / TARGET_FPS
    last_frame_time   = 0.0
    last_trigger_time = 0.0
    current_date      = datetime.date.today()
    reconnect_attempt = 0

    try:
        while _running:
            now = time.monotonic()

            _reload_zone_if_changed()
            _check_spray_request()

            if cap is None:
                cap, reconnect_attempt = _reconnect(RTSP_URL, reconnect_attempt)
                if cap is None:
                    continue
                last_frame_time = 0.0

            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                continue

            reconnect_attempt = 0

            if (now - last_frame_time) < frame_interval:
                continue
            last_frame_time = now

            # Roll CSV over at midnight
            today = datetime.date.today()
            if today != current_date:
                csv_fh.close()
                csv_fh, csv_writer = open_csv(LOG_DIR)
                current_date = today
                _cleanup_old_frames()

            # ── Inference ────────────────────────────────────────────────────
            squirrel_results = squirrel_model(frame, verbose=False)[0]

            squirrel_boxes = [
                box for box in squirrel_results.boxes
                if float(box.conf[0]) >= CONFIDENCE_THRESHOLD
                and squirrel_model.names[int(box.cls[0])].lower() == "squirrel"
            ]

            bird_boxes = []
            if bird_model:
                bird_results = bird_model(frame, verbose=False)[0]
                bird_boxes = [
                    box for box in bird_results.boxes
                    if float(box.conf[0]) >= BIRD_CONFIDENCE_THRESHOLD
                    and bird_model.names[int(box.cls[0])].lower() == "bird"
                ]
                # DEBUG: log raw COCO bird detections to help tune BIRD_CONFIDENCE_THRESHOLD
                for box in bird_results.boxes:
                    if bird_model.names[int(box.cls[0])].lower() == "bird":
                        log.debug(f"[coco-raw] bird conf={float(box.conf[0]):.3f} (threshold={BIRD_CONFIDENCE_THRESHOLD})")

            wildlife_boxes = []
            if wildlife_model:
                wildlife_results = wildlife_model(frame, verbose=False)[0]
                wildlife_boxes = [
                    box for box in wildlife_results.boxes
                    if float(box.conf[0]) >= WILDLIFE_CONFIDENCE_THRESHOLD
                    and wildlife_model.names[int(box.cls[0])].lower() in WILDLIFE_SUPPRESS_CLASSES
                ]

            if not squirrel_boxes and not bird_boxes and not wildlife_boxes:
                continue

            frame_has_bird     = len(bird_boxes) > 0
            frame_has_wildlife = len(wildlife_boxes) > 0
            frame_h, frame_w   = frame.shape[:2]
            ts = datetime.datetime.now().isoformat(timespec="seconds")

            # Save one frame per inference pass; name by dominant class
            if squirrel_boxes:
                prefix = "squirrel"
            elif wildlife_boxes:
                prefix = wildlife_model.names[int(wildlife_boxes[0].cls[0])].lower()
            else:
                prefix = "bird"
            frame_filename = f"{ts.replace(':', '-')}_{prefix}.jpg"
            frame_path = FRAMES_DIR / frame_filename
            if not cv2.imwrite(str(frame_path), frame):
                log.warning(f"Failed to write frame: {frame_path}")
                frame_path = Path("")

            # ── Log squirrel detections ───────────────────────────────────────
            for box in squirrel_boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                triggered = not frame_has_bird and not frame_has_wildlife
                suppress_reason = ""

                if not triggered:
                    if frame_has_wildlife:
                        suppress_reason = ", ".join(
                            wildlife_model.names[int(b.cls[0])].lower() for b in wildlife_boxes
                        )
                    else:
                        suppress_reason = "bird present"
                elif not _in_feeder_zone(x1, y1, x2, y2, frame_w, frame_h):
                    triggered = False
                    suppress_reason = "outside feeder zone"
                else:
                    now_mono = time.monotonic()
                    if not (DAY_START <= datetime.datetime.now().hour < DAY_END):
                        triggered = False
                        suppress_reason = "nighttime"
                    elif (now_mono - last_trigger_time) < COOLDOWN_SEC:
                        triggered = False
                        remaining = COOLDOWN_SEC - (now_mono - last_trigger_time)
                        suppress_reason = f"cooldown {remaining:.0f}s"
                    elif conf < SPRAY_CONFIDENCE_THRESHOLD:
                        triggered = False
                        suppress_reason = f"low confidence ({conf:.0%} < {SPRAY_CONFIDENCE_THRESHOLD:.0%})"

                if triggered:
                    last_trigger_time = time.monotonic()
                    if SPRAY_ENABLED:
                        _fire_gpio()
                    else:
                        suppress_reason = "spray disabled"
                        triggered = False

                csv_writer.writerow({
                    "timestamp":  ts,
                    "class":      "squirrel",
                    "confidence": round(conf, 4),
                    "triggered":  triggered,
                    "x1": x1, "y1": y1, "w": w, "h": h,
                    "frame_path": str(frame_path),
                })
                csv_fh.flush()

                if triggered:
                    log.info(f"🐿️  SQUIRREL          conf={conf:.2f}  bbox=({x1},{y1},{w}×{h})")
                else:
                    log.info(f"🐿️  squirrel [suppressed — {suppress_reason}]  conf={conf:.2f}  bbox=({x1},{y1},{w}×{h})")

            # ── Log bird detections ───────────────────────────────────────────
            for box in bird_boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                csv_writer.writerow({
                    "timestamp":  ts,
                    "class":      "bird",
                    "confidence": round(conf, 4),
                    "triggered":  False,
                    "x1": x1, "y1": y1, "w": w, "h": h,
                    "frame_path": str(frame_path),
                })
                csv_fh.flush()

                log.info(f"🐦  bird               conf={conf:.2f}  bbox=({x1},{y1},{w}×{h})")

            # ── Log wildlife detections ───────────────────────────────────────
            for box in wildlife_boxes:
                cls_name = wildlife_model.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                csv_writer.writerow({
                    "timestamp":  ts,
                    "class":      cls_name,
                    "confidence": round(conf, 4),
                    "triggered":  False,
                    "x1": x1, "y1": y1, "w": w, "h": h,
                    "frame_path": str(frame_path),
                })
                csv_fh.flush()

                log.info(f"🦌  {cls_name:<18}  conf={conf:.2f}  bbox=({x1},{y1},{w}×{h})")

    finally:
        if cap is not None:
            cap.release()
        csv_fh.close()
        _cleanup_gpio()
        log.info("Done.")


if __name__ == "__main__":
    main()
