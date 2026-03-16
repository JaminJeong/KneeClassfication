"""
Knee Osteoarthritis Classification Training Script
Uses Ultralytics YOLO classification (YOLOv8-cls / YOLO11-cls)

Usage:
    python scripts/train.py
    python scripts/train.py --model yolo11n-cls.pt --epochs 100 --imgsz 224
    python scripts/train.py --model yolo11s-cls.pt --epochs 50 --batch 32 --device cuda:0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO classifier for knee osteoarthritis grading")

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-cls.pt",
        help="Pretrained model to start from. Options: yolo11n-cls.pt, yolo11s-cls.pt, "
             "yolo11m-cls.pt, yolo11l-cls.pt, yolo11x-cls.pt, "
             "yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt, yolov8l-cls.pt, yolov8x-cls.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/workspace/data/knee-osteoarthritis-dataset-with-severity",
        help="Dataset root directory (must contain train/ val/ test/ subdirs with class folders)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default="", help="Device: '' for auto, 'cpu', '0', '0,1'")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="/workspace/runs/classify", help="Save directory")
    parser.add_argument("--name", type=str, default="knee_cls", help="Experiment name")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0 to disable)")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="Optimizer",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for classifier head")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--amp", action="store_true", default=True, help="Automatic mixed precision")
    parser.add_argument("--augment", action="store_true", default=True, help="Use TTA during val")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data path: absolute paths are used as-is, relative paths resolved from project root
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / args.data

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    if not data_path.is_dir():
        raise ValueError(
            f"'--data' must be the dataset root directory, not a YAML file.\n"
            f"  Got: {data_path}\n"
            f"  Expected a directory containing train/ val/ test/ subdirs."
        )

    # Load model (downloads pretrained weights automatically if not cached)
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)

    print(f"\n{'='*60}")
    print("Knee Osteoarthritis KL-Grade Classification Training")
    print(f"{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Data dir : {data_path}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Image sz : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device or 'auto'}")
    print(f"{'='*60}\n")

    # Train
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        patience=args.patience,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        amp=args.amp,
        augment=args.augment,
        exist_ok=False,
        verbose=True,
    )

    print(f"\nTraining complete. Results saved to: {results.save_dir}")

    # Run evaluation on test set
    print("\nEvaluating on test set...")
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
    )
    print(f"Test Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Test Top-5 Accuracy: {metrics.top5:.4f}")


if __name__ == "__main__":
    main()
