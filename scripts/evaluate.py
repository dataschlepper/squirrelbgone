"""
SquirrelBGone — Model Evaluation
Evaluates models/squirrelbgone_best.pt against the Warren Wiens test set.

Requirements:
    pip install ultralytics roboflow

Usage:
    export ROBOFLOW_API_KEY='rf_your_key_here'
    python3 evaluate.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/squirrelbgone_best.pt")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# ── Checks ────────────────────────────────────────────────────────────────────

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

if not ROBOFLOW_API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not set.")
    sys.exit(1)

# ── Download test set ─────────────────────────────────────────────────────────

print("Downloading Warren Wiens test set from Roboflow Universe...")
from roboflow import Roboflow

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
dataset = rf.workspace("warren-wiens-d0d4p") \
             .project("squirrel-detector-1.1") \
             .version(1) \
             .download("yolov8")

data_yaml = Path(dataset.location) / "data.yaml"

# ── Evaluate ──────────────────────────────────────────────────────────────────

print(f"\nLoading {MODEL_PATH}...")
from ultralytics import YOLO
import yaml

model = YOLO(str(MODEL_PATH))
metrics = model.val(data=str(data_yaml), split="test")

with open(data_yaml) as f:
    class_names = yaml.safe_load(f)["names"]

# ── Results ───────────────────────────────────────────────────────────────────

print("\n── Overall ───────────────────────────────────────")
print(f"  mAP50:     {metrics.box.map50:.3f}")
print(f"  mAP50-95:  {metrics.box.map:.3f}")
print(f"  Precision: {metrics.box.mp:.3f}")
print(f"  Recall:    {metrics.box.mr:.3f}")

print("\n── Per-class AP50 ────────────────────────────────")
for name, ap in zip(class_names, metrics.box.ap50):
    bar = "█" * int(ap * 20)
    print(f"  {name:<12} {ap:.3f}  {bar}")