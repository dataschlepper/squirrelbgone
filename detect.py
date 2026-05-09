#!/usr/bin/env python3
"""
SquirrelBGone — detect.py
Phase 1: Detection only. Reads RTSP stream, runs YOLOv8 inference locally,
prints detections + logs to CSV with saved frames.

Runs two models per frame:
  - Primary squirrel model (MODEL_PATH) — single-class squirrel detector
  - Bird suppression model (MODEL_PATH_BIRD, default yolov8n.pt) — COCO model;
    if a bird is present in the same frame as a squirrel, the squirrel trigger
    is suppressed (both are still logged)

Requirements:
    pip install ultralytics opencv-python python-dotenv

Usage:
    cp .env.example .env && nano .env
    python detect.py
"""

import csv
import datetime
import logging
import os
import signal
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RTSP_URL                  = os.environ.get("RTSP_URL", "")
MODEL_PATH                = os.environ.get("MODEL_PATH", "squirrel_detector.pt")
MODEL_PATH_BIRD           = os.environ.get("MODEL_PATH_BIRD", "yolov8n.pt")
TARGET_FPS                = int(os.environ.get("TARGET_FPS", "5"))
CONFIDENCE_THRESHOLD      = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
BIRD_CONFIDENCE_THRESHOLD = float(os.environ.get("BIRD_CONFIDENCE_THRESHOLD", "0.45"))

# Where to write logs and saved frames
LOG_DIR    = Path(os.environ.get("LOG_DIR", "logs"))
FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", "frames"))

# Classes that should trigger the sprayer (Phase 2+)
TRIGGER_CLASSES = {"squirrel"}

# Classes we want logged but never triggered on
BENIGN_CLASSES  = {"bird", "crow", "pigeon", "robin", "sparrow"}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sbg")

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

    from ultralytics import YOLO

    log.info(f"Loading squirrel model: {MODEL_PATH}")
    squirrel_model = YOLO(MODEL_PATH)
    log.info("Squirrel model loaded.")

    log.info(f"Loading bird model: {MODEL_PATH_BIRD}")
    bird_model = YOLO(MODEL_PATH_BIRD)
    log.info("Bird model loaded.")

    csv_fh, csv_writer = open_csv(LOG_DIR)

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
    current_date      = datetime.date.today()
    reconnect_attempt = 0

    try:
        while _running:
            now = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap, reconnect_attempt = _reconnect(RTSP_URL, reconnect_attempt)
                if cap is None:
                    continue
                last_frame_time = 0.0
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

            # ── Inference ────────────────────────────────────────────────────
            squirrel_results = squirrel_model(frame, verbose=False)[0]
            bird_results     = bird_model(frame, verbose=False)[0]

            squirrel_boxes = [
                box for box in squirrel_results.boxes
                if float(box.conf[0]) >= CONFIDENCE_THRESHOLD
                and squirrel_model.names[int(box.cls[0])].lower() == "squirrel"
            ]

            # Filter COCO output to "bird" class only
            bird_boxes = [
                box for box in bird_results.boxes
                if float(box.conf[0]) >= BIRD_CONFIDENCE_THRESHOLD
                and bird_model.names[int(box.cls[0])].lower() == "bird"
            ]

            # DEBUG: log all COCO detections to see what the model is finding
            for box in bird_results.boxes:
                name = bird_model.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])
                if name == "bird":
                    log.debug(f"[coco-raw] bird conf={conf:.3f} (threshold={BIRD_CONFIDENCE_THRESHOLD})")

            if not squirrel_boxes and not bird_boxes:
                continue

            frame_has_bird = len(bird_boxes) > 0
            ts = datetime.datetime.now().isoformat(timespec="seconds")

            # Save one frame per inference pass; name by dominant class
            prefix = "squirrel" if squirrel_boxes else "bird"
            frame_filename = f"{ts.replace(':', '-')}_{prefix}.jpg"
            frame_path = FRAMES_DIR / frame_filename
            cv2.imwrite(str(frame_path), frame)

            # ── Log squirrel detections ───────────────────────────────────────
            for box in squirrel_boxes:
                conf = float(box.conf[0])
                triggered = not frame_has_bird  # suppress if bird co-present
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

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
                    log.info(f"🐿️  squirrel [suppressed — bird present]  conf={conf:.2f}  bbox=({x1},{y1},{w}×{h})")

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

    finally:
        cap.release()
        csv_fh.close()
        log.info("Done.")


if __name__ == "__main__":
    main()
