#!/usr/bin/env python3
"""
SquirrelBGone — detect.py
Phase 1: Detection only. Reads RTSP stream, runs YOLOv8 inference locally,
prints detections with confidence. No GPIO, no water — just eyes.

Requirements (conda env squirrelbgone, Python 3.12):
    pip install ultralytics opencv-python-headless python-dotenv

Model: Warren Wiens "Squirrel Detector 1.1" weights (.pt) downloaded once
from Roboflow Universe and committed to repo (or placed alongside this script).
Download: https://universe.roboflow.com/warren-wiens-d0d4p/squirrel-detector-1.1
  → Versions → v1 → Export → YOLOv8 PyTorch → download weights .pt file

Usage:
    cp .env.example .env && nano .env   # set RTSP_URL
    python detect.py
"""

import os
import sys
import time
import signal
import logging
import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# RTSP sub-stream: lower res, lower CPU, plenty of detail at 15-20ft
# Format: rtsp://<user>:<pass>@<camera_ip>:554/h264Preview_01_sub
RTSP_URL = os.environ.get("RTSP_URL", "")

# Path to downloaded .pt weights file
MODEL_PATH = os.environ.get("MODEL_PATH", "squirrel_detector.pt")

# Inference frame rate cap — Pi 5 handles 5fps comfortably on CPU
TARGET_FPS = int(os.environ.get("TARGET_FPS", "5"))

# Confidence threshold — start here, tune in Task 6 after reviewing logs
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))

# Classes that should trigger the sprayer (Task 2+)
TRIGGER_CLASSES = {"squirrel"}

# Classes we want to see logged but never trigger on
BENIGN_CLASSES = {"bird", "crow", "pigeon", "robin", "sparrow"}

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

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Validate config
    if not RTSP_URL:
        log.error("RTSP_URL is not set. Copy .env.example → .env and fill it in.")
        sys.exit(1)

    if not Path(MODEL_PATH).exists():
        log.error(
            f"Model weights not found at '{MODEL_PATH}'.\n"
            "  Download from Roboflow Universe:\n"
            "    https://universe.roboflow.com/warren-wiens-d0d4p/squirrel-detector-1.1\n"
            "  → Versions → v1 → Export → YOLOv8 PyTorch → download .pt\n"
            "  Then set MODEL_PATH in .env or place the file alongside detect.py."
        )
        sys.exit(1)

    # Load model (ultralytics caches after first load)
    log.info(f"Loading model: {MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    log.info("Model loaded.")

    # Open RTSP stream
    log.info(f"Connecting to stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        log.error(
            "Could not open RTSP stream. Check:\n"
            "  1. Camera IP and credentials in RTSP_URL\n"
            "  2. Camera is powered and on the same LAN as the Pi\n"
            "  3. ffprobe rtsp://... to verify the stream independently"
        )
        sys.exit(1)

    log.info(f"Stream open. Running at {TARGET_FPS}fps, threshold={CONFIDENCE_THRESHOLD}")
    log.info("Watching for movement… (Ctrl+C to stop)\n")

    frame_interval = 1.0 / TARGET_FPS
    last_frame_time = 0.0

    while _running:
        now = time.monotonic()

        # Throttle to TARGET_FPS — read and discard buffered frames between samples
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame read failed — stream may have dropped. Retrying in 3s…")
            time.sleep(3)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        if (now - last_frame_time) < frame_interval:
            continue  # too soon — discard frame, keep reading to drain buffer
        last_frame_time = now

        # Run inference (CPU, no GPU needed)
        results = model(frame, verbose=False)[0]

        # Parse detections
        any_hit = False
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            if cls_name in TRIGGER_CLASSES:
                icon = "🐿️  SQUIRREL"
            elif cls_name in BENIGN_CLASSES:
                icon = "🐦  bird"
            else:
                icon = f"   {cls_name}"

            log.info(f"{icon:<22}  conf={conf:.2f}  bbox=({x1},{y1},{w}×{h})")
            any_hit = True

    cap.release()
    log.info("Done.")


if __name__ == "__main__":
    main()