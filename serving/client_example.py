"""
Serving API 클라이언트 예제
서버 실행 후 이 스크립트로 예측을 테스트할 수 있습니다.

Usage:
    python serving/client_example.py --image path/to/xray.png
    python serving/client_example.py --image path/to/xray.png --url http://localhost:8000
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def predict_with_file_upload(image_path: str, url: str) -> dict:
    """multipart/form-data로 이미지 업로드 후 예측"""
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/png")}
        response = requests.post(f"{url}/predict", files=files)
    response.raise_for_status()
    return response.json()


def predict_with_base64(image_path: str, url: str) -> dict:
    """base64 인코딩으로 이미지 전송 후 예측"""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    payload = {"image": encoded, "imgsz": 224}
    response = requests.post(f"{url}/predict/base64", json=payload)
    response.raise_for_status()
    return response.json()


def print_result(result: dict) -> None:
    top = result["top_prediction"]
    print(f"\n{'='*50}")
    print(f"  Top Prediction : {top['grade']} — {top['label']}")
    print(f"  Confidence     : {top['confidence']:.1%}")
    print(f"  Description    : {top['description']}")
    print(f"  Inference time : {result['inference_ms']:.1f} ms")
    print(f"{'='*50}")
    print("\nAll Probabilities:")
    for pred in result["all_predictions"]:
        bar = "█" * int(pred["confidence"] * 30)
        print(f"  {pred['grade']:5s} {pred['label']:10s}  {bar:<30s}  {pred['confidence']:.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Knee OA Severity API client")
    parser.add_argument("--image", required=True, help="Path to knee X-ray image")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--method",
        choices=["upload", "base64"],
        default="upload",
        help="Request method (default: upload)",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Health check
    try:
        health = requests.get(f"{args.url}/health", timeout=5).json()
        print(f"Server status : {health['status']}")
        print(f"Model loaded  : {health['model_loaded']}")
    except requests.exceptions.ConnectionError:
        print(f"Error: cannot connect to {args.url}", file=sys.stderr)
        sys.exit(1)

    # Predict
    if args.method == "base64":
        result = predict_with_base64(args.image, args.url)
    else:
        result = predict_with_file_upload(args.image, args.url)

    print_result(result)


if __name__ == "__main__":
    main()
