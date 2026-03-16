#!/usr/bin/env bash
# =============================================================================
# Knee Osteoarthritis Classification — Inference (Ultralytics CLI)
# =============================================================================
# Run inside Docker:
#   docker compose run --rm yolo bash scripts/predict.sh \
#     --weights runs/classify/knee_cls/weights/best.pt \
#     --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/2/9001695L.png
#
#   # Predict entire test directory
#   docker compose run --rm yolo bash scripts/predict.sh \
#     --weights runs/classify/knee_cls/weights/best.pt \
#     --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/
# =============================================================================

set -euo pipefail

# ── Default parameters ────────────────────────────────────────────────────────
WEIGHTS="/workspace/runs/classify/knee_cls/weights/best.pt"
SOURCE=""
IMGSZ=224
DEVICE=""
CONF=0.25
SAVE=False
PROJECT="/workspace/runs/predict"
NAME="knee_predict"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: bash scripts/predict.sh --source <path> [OPTIONS]

Required:
  --source    Image path or directory to run inference on

Options:
  --weights   Model weights path (default: $WEIGHTS)
  --imgsz     Input image size (default: $IMGSZ)
  --device    Device: '' auto, 'cpu', '0' (default: auto)
  --conf      Confidence threshold (default: $CONF)
  --save      Save result images: True/False (default: $SAVE)
  --project   Output directory (default: $PROJECT)
  --name      Experiment name (default: $NAME)
  -h, --help  Show this help

Examples:
  # Single image
  bash scripts/predict.sh --source /workspace/data/.../test/2/9001695L.png

  # Entire test split
  bash scripts/predict.sh \
    --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/ \
    --save True

  # Custom weights
  bash scripts/predict.sh \
    --weights runs/classify/knee_cls2/weights/best.pt \
    --source /workspace/data/.../test/
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights)  WEIGHTS="$2";  shift 2 ;;
    --source)   SOURCE="$2";   shift 2 ;;
    --imgsz)    IMGSZ="$2";    shift 2 ;;
    --device)   DEVICE="$2";   shift 2 ;;
    --conf)     CONF="$2";     shift 2 ;;
    --save)     SAVE="$2";     shift 2 ;;
    --project)  PROJECT="$2";  shift 2 ;;
    --name)     NAME="$2";     shift 2 ;;
    -h|--help)  usage ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage ;;
  esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$SOURCE" ]]; then
  echo "[ERROR] --source is required."
  usage
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[ERROR] Weights not found: $WEIGHTS"
  echo "        Train a model first: bash scripts/train.sh"
  exit 1
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " Knee OA KL-Grade Classification — Inference (CLI)"
echo "============================================================"
echo "  Weights   : $WEIGHTS"
echo "  Source    : $SOURCE"
echo "  Image sz  : $IMGSZ"
echo "  Device    : ${DEVICE:-auto}"
echo "  Save      : $SAVE"
echo "============================================================"

DEVICE_ARG=""
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARG="device=$DEVICE"
fi

# ── Run inference ─────────────────────────────────────────────────────────────
yolo classify predict \
  model="$WEIGHTS" \
  source="$SOURCE" \
  imgsz="$IMGSZ" \
  conf="$CONF" \
  save="$SAVE" \
  project="$PROJECT" \
  name="$NAME" \
  ${DEVICE_ARG}

echo ""
echo "[Done] Results saved to: $PROJECT/$NAME"
