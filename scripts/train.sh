#!/usr/bin/env bash
# =============================================================================
# Knee Osteoarthritis Classification — Training (Ultralytics CLI)
# =============================================================================
# Run inside Docker:
#   docker compose run --rm yolo bash scripts/train.sh
#   docker compose run --rm yolo bash scripts/train.sh --model yolo11s-cls.pt --epochs 50
#
# Run on host (if ultralytics is installed locally):
#   bash scripts/train.sh
# =============================================================================

set -euo pipefail

# ── Default parameters ────────────────────────────────────────────────────────
DATA="/workspace/data/knee-osteoarthritis-dataset-with-severity"
MODEL="yolo11n-cls.pt"
EPOCHS=100
IMGSZ=224
BATCH=32
DEVICE=""          # auto-detect (set "cpu" or "0" to override)
WORKERS=8
LR0=0.01
PATIENCE=20
OPTIMIZER="AdamW"
DROPOUT=0.2
LABEL_SMOOTHING=0.1
PROJECT="/workspace/runs/classify"
NAME="knee_cls"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: bash scripts/train.sh [OPTIONS]

Options:
  --data            Dataset root directory (default: $DATA)
  --model           Pretrained model (default: $MODEL)
                    Options: yolo11n/s/m/l/x-cls.pt, yolov8n/s/m/l/x-cls.pt
  --epochs          Training epochs (default: $EPOCHS)
  --imgsz           Input image size (default: $IMGSZ)
  --batch           Batch size (default: $BATCH)
  --device          Device: '' auto, 'cpu', '0', '0,1' (default: auto)
  --workers         Dataloader workers (default: $WORKERS)
  --lr0             Initial learning rate (default: $LR0)
  --patience        Early stopping patience (default: $PATIENCE)
  --optimizer       Optimizer (default: $OPTIMIZER)
  --dropout         Dropout (default: $DROPOUT)
  --project         Output project dir (default: $PROJECT)
  --name            Experiment name (default: $NAME)
  -h, --help        Show this help

Examples:
  bash scripts/train.sh
  bash scripts/train.sh --model yolo11s-cls.pt --epochs 50 --batch 64
  bash scripts/train.sh --device cpu --epochs 5 --name knee_cls_test
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)           DATA="$2";            shift 2 ;;
    --model)          MODEL="$2";           shift 2 ;;
    --epochs)         EPOCHS="$2";          shift 2 ;;
    --imgsz)          IMGSZ="$2";           shift 2 ;;
    --batch)          BATCH="$2";           shift 2 ;;
    --device)         DEVICE="$2";          shift 2 ;;
    --workers)        WORKERS="$2";         shift 2 ;;
    --lr0)            LR0="$2";             shift 2 ;;
    --patience)       PATIENCE="$2";        shift 2 ;;
    --optimizer)      OPTIMIZER="$2";       shift 2 ;;
    --dropout)        DROPOUT="$2";         shift 2 ;;
    --project)        PROJECT="$2";         shift 2 ;;
    --name)           NAME="$2";            shift 2 ;;
    -h|--help)        usage ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage ;;
  esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! -d "$DATA" ]]; then
  echo "[ERROR] Dataset directory not found: $DATA"
  echo "        Mount your data volume or set --data to the correct path."
  exit 1
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " Knee OA KL-Grade Classification — Training (CLI)"
echo "============================================================"
echo "  Data      : $DATA"
echo "  Model     : $MODEL"
echo "  Epochs    : $EPOCHS"
echo "  Image sz  : $IMGSZ"
echo "  Batch     : $BATCH"
echo "  Device    : ${DEVICE:-auto}"
echo "  Optimizer : $OPTIMIZER"
echo "  LR0       : $LR0"
echo "  Patience  : $PATIENCE"
echo "  Project   : $PROJECT/$NAME"
echo "============================================================"

# ── Build device argument (omit if empty = auto) ──────────────────────────────
DEVICE_ARG=""
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARG="device=$DEVICE"
fi

# ── Run training ──────────────────────────────────────────────────────────────
# Note: For classification, 'data' must be the dataset root directory path,
#       NOT a YAML file. Folder names (0,1,2,3,4) become class labels.
#       Reference: https://docs.ultralytics.com/datasets/classify/
yolo classify train \
  data="$DATA" \
  model="$MODEL" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  workers="$WORKERS" \
  lr0="$LR0" \
  patience="$PATIENCE" \
  optimizer="$OPTIMIZER" \
  dropout="$DROPOUT" \
  label_smoothing="$LABEL_SMOOTHING" \
  project="$PROJECT" \
  name="$NAME" \
  pretrained=True \
  amp=True \
  exist_ok=False \
  ${DEVICE_ARG}

echo ""
echo "[Done] Weights saved to: $PROJECT/$NAME/weights/best.pt"
