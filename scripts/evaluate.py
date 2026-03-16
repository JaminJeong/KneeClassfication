"""
Knee Osteoarthritis Classification Evaluation Script
Computes accuracy, confusion matrix, and per-class metrics on a dataset split.

Usage:
    python scripts/evaluate.py --weights runs/classify/knee_cls/weights/best.pt
    python scripts/evaluate.py --weights best.pt --split test --plot
    python scripts/evaluate.py --weights best.pt --split val --batch 64
"""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

KL_GRADES = ["Normal (KL-0)", "Doubtful (KL-1)", "Minimal (KL-2)", "Moderate (KL-3)", "Severe (KL-4)"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained knee classifier on a dataset split")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pt)")
    parser.add_argument(
        "--data",
        type=str,
        default="/workspace/data/knee-osteoarthritis-dataset-with-severity",
        help="Dataset root directory (must contain train/ val/ test/ subdirs with class folders)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device: '' auto, 'cpu', '0'")
    parser.add_argument("--plot", action="store_true", help="Save confusion matrix and metric plots")
    parser.add_argument("--project", type=str, default="/workspace/runs/evaluate", help="Output directory")
    parser.add_argument("--name", type=str, default="knee_eval", help="Experiment name")
    return parser.parse_args()


def print_metrics_table(metrics):
    """Print a formatted metrics summary."""
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy : {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
    print(f"  Top-5 Accuracy : {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")
    print(f"{'='*60}")


def compute_per_class_accuracy(model, data_path: str, split: str, imgsz: int, device):
    """Run predictions and compute per-class accuracy."""
    from pathlib import Path as P

    split_dir = P(data_path) / split

    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return

    per_class_correct = {}
    per_class_total = {}

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name))

    print(f"\nComputing per-class accuracy on '{split}' split...")
    for cls_dir in class_dirs:
        cls_id = int(cls_dir.name)
        images = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))

        if not images:
            continue

        results = model.predict(
            source=str(cls_dir),
            imgsz=imgsz,
            device=device,
            verbose=False,
            stream=True,
        )

        correct = 0
        total = 0
        for r in results:
            pred_cls = int(r.probs.top1)
            if pred_cls == cls_id:
                correct += 1
            total += 1

        per_class_correct[cls_id] = correct
        per_class_total[cls_id] = total

    print(f"\n{'Class':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 55)
    all_correct = 0
    all_total = 0
    for cls_id in sorted(per_class_correct.keys()):
        acc = per_class_correct[cls_id] / per_class_total[cls_id] if per_class_total[cls_id] > 0 else 0
        name = KL_GRADES[cls_id]
        print(f"  {name:<23} {per_class_correct[cls_id]:>8} {per_class_total[cls_id]:>8} {acc:>9.1%}")
        all_correct += per_class_correct[cls_id]
        all_total += per_class_total[cls_id]

    print("-" * 55)
    overall = all_correct / all_total if all_total > 0 else 0
    print(f"  {'Overall':<23} {all_correct:>8} {all_total:>8} {overall:>9.1%}")


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / args.data

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    if not data_path.is_dir():
        raise ValueError(
            f"'--data' must be the dataset root directory, not a YAML file.\n"
            f"  Got: {data_path}"
        )

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # Standard ultralytics val
    print(f"\nRunning ultralytics val on '{args.split}' split...")
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        plots=args.plot,
        project=args.project,
        name=args.name,
    )

    print_metrics_table(metrics)

    # Per-class breakdown
    compute_per_class_accuracy(
        model,
        data_path=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
    )


if __name__ == "__main__":
    main()
