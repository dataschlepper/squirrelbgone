#!/usr/bin/env python3
"""
SquirrelBGone — detect.py
Phase 1: Detection only. Reads RTSP stream, runs YOLOv8 inference locally,
prints detections + logs to CSV with saved frames.

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

RTSP_URL             = os.environ.get("RTSP_URL", "")
MODEL_PATH           = os.environ.get("MODEL_PATH", "squirrel_detector.pt")
TARGET_FPS           = int(os.environ.get("TARGET_FPS", "5"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))

# Where to write logs and saved frames
LOG_DIR    = Path(os.environ.get("LOG_DIR", "logs"))
FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", "frames"))

# Classes that should trigger the sprayer (Phase 2+)
TRIGGER_CLASSES = {"squirrel"}

# Classes we want logged but never triggered on
BENIGN_CLASSES  = {"bird", "crow", "pigeon", "robin", "sparrow"}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
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

CSV_FIELDS = ["timestamp", "class", "confidence", "x1", "y1", "w", "h", "frame_path"]

def open_csv(log_dir: Path):
    """Open (or append to) today's detection log. Returns (file_handle, csv_writer)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    csv_path = log_dir / f"detections_{date_str}.csv"
    is_new = not csv_path.exists()
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

    # Create output dirs
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    log.info(f"Loading model: {MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    log.info("Model loaded.")

    # Open CSV
    csv_fh, csv_writer = open_csv(LOG_DIR)

    # Open stream
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

    log.info(f"Stream open. {TARGET_FPS}fps · threshold={CONFIDENCE_THRESHOLD}")
    log.info("Watching… (Ctrl+C to stop)\n")

    frame_interval  = 1.0 / TARGET_FPS
    last_frame_time = 0.0
    current_date    = datetime.date.today()
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
                last_frame_time = 0.0  # drain any buffered frames after reconnect
                continue

            reconnect_attempt = 0  # reset on any successful read

            if (now - last_frame_time) < frame_interval:
                continue
            last_frame_time = now

            # Roll CSV over at midnight
            today = datetime.date.today()
            if today != current_date:
                csv_fh.close()
                csv_fh, csv_writer = open_csv(LOG_DIR)
                current_date = today

            # Inference
            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                cls_name = model.names[int(box.cls[0])].lower()
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                ts = datetime.datetime.now().isoformat(timespec="seconds")

                # Save frame
                frame_filename = f"{ts.replace(':', '-')}_{cls_name}_{conf:.2f}.jpg"
                frame_path = FRAMES_DIR / frame_filename
                cv2.imwrite(str(frame_path), frame)

                # Write CSV row
                csv_writer.writerow({
                    "timestamp":  ts,
                    "class":      cls_name,
                    "confidence": round(conf, 4),
                    "x1": x1, "y1": y1, "w": w, "h": h,
                    "frame_path": str(frame_path),
                })
                csv_fh.flush()

                # Console
                if cls_name in TRIGGER_CLASSES:
                    icon = "🐿️  SQUIRREL"
                elif cls_name in BENIGN_CLASSES:
                    icon = "🐦  bird"
                else:
                    icon = f"   {cls_name}"

                log.info(f"{icon:<22}  conf={conf:.2f}  bbox=({x1},{y1},{w}×{h})")

    finally:
        cap.release()
        csv_fh.close()
        log.info("Done.")


if __name__ == "__main__":
    main()
