#!/usr/bin/env bash
# =============================================================================
# Knee Osteoarthritis Classification — Evaluation (Ultralytics CLI)
# =============================================================================
# Run inside Docker:
#   docker compose run --rm yolo bash scripts/evaluate.sh
#   docker compose run --rm yolo bash scripts/evaluate.sh \
#     --weights runs/classify/knee_cls/weights/best.pt --split test --plot True
# =============================================================================

set -euo pipefail

# ── Default parameters ────────────────────────────────────────────────────────
DATA="/workspace/data/knee-osteoarthritis-dataset-with-severity"
WEIGHTS="/workspace/runs/classify/knee_cls/weights/best.pt"
SPLIT="test"
IMGSZ=224
BATCH=32
DEVICE=""
PLOT=False
PROJECT="/workspace/runs/evaluate"
NAME="knee_eval"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: bash scripts/evaluate.sh [OPTIONS]

Options:
  --data      Dataset root directory (default: $DATA)
  --weights   Model weights path (default: $WEIGHTS)
  --split     Dataset split: train / val / test (default: $SPLIT)
  --imgsz     Input image size (default: $IMGSZ)
  --batch     Batch size (default: $BATCH)
  --device    Device: '' auto, 'cpu', '0' (default: auto)
  --plot      Save confusion matrix plots: True/False (default: $PLOT)
  --project   Output directory (default: $PROJECT)
  --name      Experiment name (default: $NAME)
  -h, --help  Show this help

Examples:
  # Evaluate on test set (default)
  bash scripts/evaluate.sh

  # Evaluate on val set with plots
  bash scripts/evaluate.sh --split val --plot True

  # Evaluate a specific model
  bash scripts/evaluate.sh \
    --weights runs/classify/knee_cls2/weights/best.pt \
    --split test --plot True
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)     DATA="$2";     shift 2 ;;
    --weights)  WEIGHTS="$2";  shift 2 ;;
    --split)    SPLIT="$2";    shift 2 ;;
    --imgsz)    IMGSZ="$2";    shift 2 ;;
    --batch)    BATCH="$2";    shift 2 ;;
    --device)   DEVICE="$2";   shift 2 ;;
    --plot)     PLOT="$2";     shift 2 ;;
    --project)  PROJECT="$2";  shift 2 ;;
    --name)     NAME="$2";     shift 2 ;;
    -h|--help)  usage ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage ;;
  esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! -d "$DATA" ]]; then
  echo "[ERROR] Dataset directory not found: $DATA"
  exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[ERROR] Weights not found: $WEIGHTS"
  echo "        Train a model first: bash scripts/train.sh"
  exit 1
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " Knee OA KL-Grade Classification — Evaluation (CLI)"
echo "============================================================"
echo "  Data      : $DATA"
echo "  Weights   : $WEIGHTS"
echo "  Split     : $SPLIT"
echo "  Image sz  : $IMGSZ"
echo "  Batch     : $BATCH"
echo "  Device    : ${DEVICE:-auto}"
echo "  Plot      : $PLOT"
echo "  Output    : $PROJECT/$NAME"
echo "============================================================"

DEVICE_ARG=""
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARG="device=$DEVICE"
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
# Note: For classification val, 'data' is the dataset root directory.
#       Reference: https://docs.ultralytics.com/datasets/classify/
yolo classify val \
  model="$WEIGHTS" \
  data="$DATA" \
  split="$SPLIT" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  plots="$PLOT" \
  project="$PROJECT" \
  name="$NAME" \
  ${DEVICE_ARG}

echo ""
echo "[Done] Evaluation results saved to: $PROJECT/$NAME"
