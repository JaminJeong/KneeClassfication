"""
Knee Osteoarthritis Classification Inference Script
Runs YOLO classification on a single image, directory, or glob pattern.

Usage:
    # Single image
    python scripts/predict.py --weights runs/classify/knee_cls/weights/best.pt --source image.png

    # Directory of images
    python scripts/predict.py --weights best.pt --source /path/to/images/

    # Test split of dataset
    python scripts/predict.py --weights best.pt --source /home/jayden/data/knee-osteoarthritis-dataset-with-severity/test/

    # With confidence threshold and saving results
    python scripts/predict.py --weights best.pt --source images/ --save
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

KL_GRADES = {
    0: "Normal",
    1: "Doubtful",
    2: "Minimal",
    3: "Moderate",
    4: "Severe",
}

KL_DESCRIPTIONS = {
    0: "No signs of osteoarthritis",
    1: "Possible osteophytes, doubtful joint space narrowing",
    2: "Definite osteophytes, possible joint space narrowing",
    3: "Multiple osteophytes, definite joint space narrowing, sclerosis",
    4: "Large osteophytes, marked joint space narrowing, severe sclerosis",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained knee classifier")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, directory, or glob (e.g. images/*.png)",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size")
    parser.add_argument("--device", type=str, default="", help="Device: '' auto, 'cpu', '0'")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--save", action="store_true", help="Save annotated results")
    parser.add_argument("--save-json", action="store_true", help="Save predictions to JSON")
    parser.add_argument("--project", type=str, default="/workspace/runs/predict", help="Output directory")
    parser.add_argument("--name", type=str, default="knee_predict", help="Experiment name")
    parser.add_argument("--top-k", type=int, default=5, help="Show top-k class probabilities")
    return parser.parse_args()


def format_result(result, top_k: int = 5) -> dict:
    """Format a single prediction result into a readable dict."""
    probs = result.probs

    top_indices = probs.top5[:top_k]
    top_confs = probs.top5conf[:top_k].tolist()

    top_predictions = [
        {
            "rank": i + 1,
            "class_id": int(idx),
            "class_name": KL_GRADES[int(idx)],
            "grade": f"KL-{int(idx)}",
            "description": KL_DESCRIPTIONS[int(idx)],
            "confidence": round(float(conf), 4),
        }
        for i, (idx, conf) in enumerate(zip(top_indices, top_confs))
    ]

    best = top_predictions[0]
    return {
        "image": str(result.path),
        "predicted_class": best["class_id"],
        "predicted_grade": best["grade"],
        "predicted_label": best["class_name"],
        "confidence": best["confidence"],
        "top_predictions": top_predictions,
    }


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        conf=args.conf,
        batch=args.batch,
        save=args.save,
        project=args.project,
        name=args.name,
        verbose=False,
    )

    all_predictions = []
    for result in results:
        pred = format_result(result, top_k=args.top_k)
        all_predictions.append(pred)

        # Print summary per image
        img_name = Path(pred["image"]).name
        print(f"\n[{img_name}]")
        print(f"  Prediction : KL-{pred['predicted_class']} ({pred['predicted_label']}) — {pred['confidence']:.1%}")
        for p in pred["top_predictions"]:
            bar = "█" * int(p["confidence"] * 20)
            print(f"  KL-{p['class_id']} {p['class_name']:<12} {bar:<20} {p['confidence']:.1%}")

    if args.save_json:
        out_dir = Path(args.project) / args.name
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\nPredictions saved to: {json_path}")

    print(f"\nDone. Processed {len(all_predictions)} image(s).")
    return all_predictions


if __name__ == "__main__":
    main()
