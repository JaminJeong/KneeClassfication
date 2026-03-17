"""
Knee OA YOLO Classification — FastAPI Serving
Ultralytics YOLO 모델을 REST API로 서빙합니다.

Run:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

    # With custom model path
    KNEE_MODEL_PATH=runs/classify/knee_cls_py_test/weights/best.pt \
        uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

KL_GRADES = {0: "Normal", 1: "Doubtful", 2: "Minimal", 3: "Moderate", 4: "Severe"}

KL_DESCRIPTIONS = {
    0: "No features of OA. Joint space appears normal.",
    1: "Doubtful narrowing of joint space with possible osteophytic lipping.",
    2: "Definite osteophytes and possible narrowing of the joint space.",
    3: "Multiple osteophytes, definite narrowing, some sclerosis, possible bony deformity.",
    4: "Large osteophytes, marked narrowing, severe sclerosis, and definite bony deformity.",
}

DEFAULT_MODEL = os.environ.get(
    "KNEE_MODEL_PATH",
    str(ROOT / "runs" / "classify" / "knee_cls_py_test" / "weights" / "best.pt"),
)
DEFAULT_IMGSZ = int(os.environ.get("KNEE_IMGSZ", "224"))
DEFAULT_DEVICE = os.environ.get("KNEE_DEVICE", "cpu")

# ── Global model holder ────────────────────────────────────────────────────────
_model = None


def get_model():
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model


# ── Lifespan: load model once at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    model_path = DEFAULT_MODEL
    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}. /predict will return 503.")
    else:
        from ultralytics import YOLO
        logger.info(f"Loading model from {model_path} ...")
        _model = YOLO(model_path)
        # warm-up
        dummy = Image.new("RGB", (DEFAULT_IMGSZ, DEFAULT_IMGSZ), color=128)
        _run_inference(_model, dummy)
        logger.info("Model loaded and warmed up.")
    yield
    _model = None


app = FastAPI(
    title="Knee OA Severity API",
    description="Kellgren-Lawrence grade classification from knee X-ray images using YOLO.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class ClassPrediction(BaseModel):
    class_id: int
    grade: str
    label: str
    confidence: float
    description: str


class PredictResponse(BaseModel):
    top_prediction: ClassPrediction
    all_predictions: list[ClassPrediction]
    inference_ms: float
    model_path: str


class Base64PredictRequest(BaseModel):
    image: str  # base64-encoded image (PNG/JPEG)
    imgsz: Optional[int] = None


# ── Inference helper ───────────────────────────────────────────────────────────
def _run_inference(model, image: Image.Image, imgsz: int = DEFAULT_IMGSZ) -> list[ClassPrediction]:
    """Run YOLO classification and return sorted predictions."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        results = model.predict(
            source=tmp_path,
            imgsz=imgsz,
            device=DEFAULT_DEVICE,
            verbose=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result = results[0]
    probs = result.probs

    # top5 sorted by confidence
    top5_idx = probs.top5
    top5_conf = probs.top5conf.tolist()

    predictions = [
        ClassPrediction(
            class_id=int(idx),
            grade=f"KL-{int(idx)}",
            label=KL_GRADES[int(idx)],
            confidence=round(float(conf), 6),
            description=KL_DESCRIPTIONS[int(idx)],
        )
        for idx, conf in zip(top5_idx, top5_conf)
    ]
    return predictions


def _load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health():
    """서버 및 모델 상태를 반환합니다."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": DEFAULT_MODEL,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict KL grade from uploaded image file",
)
async def predict_file(
    file: UploadFile = File(..., description="Knee X-ray image (PNG/JPEG/BMP)"),
    imgsz: int = DEFAULT_IMGSZ,
):
    """
    이미지 파일을 업로드해 Kellgren-Lawrence 등급을 예측합니다.

    - **file**: 무릎 X-ray 이미지
    - **imgsz**: 추론 시 사용할 이미지 크기 (default: 224)
    """
    model = get_model()
    raw = await file.read()
    image = _load_image_from_bytes(raw)

    t0 = time.perf_counter()
    predictions = _run_inference(model, image, imgsz=imgsz)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        top_prediction=predictions[0],
        all_predictions=predictions,
        inference_ms=round(elapsed_ms, 2),
        model_path=DEFAULT_MODEL,
    )


@app.post(
    "/predict/base64",
    response_model=PredictResponse,
    summary="Predict KL grade from base64-encoded image",
)
def predict_base64(request: Base64PredictRequest):
    """
    Base64로 인코딩된 이미지로 Kellgren-Lawrence 등급을 예측합니다.

    ```json
    {
      "image": "<base64_encoded_image>",
      "imgsz": 224
    }
    ```
    """
    model = get_model()
    try:
        raw = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    image = _load_image_from_bytes(raw)
    imgsz = request.imgsz or DEFAULT_IMGSZ

    t0 = time.perf_counter()
    predictions = _run_inference(model, image, imgsz=imgsz)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        top_prediction=predictions[0],
        all_predictions=predictions,
        inference_ms=round(elapsed_ms, 2),
        model_path=DEFAULT_MODEL,
    )


@app.get("/classes", summary="List KL grade classes")
def list_classes():
    """모델이 분류하는 KL 등급 목록을 반환합니다."""
    return {
        str(k): {"grade": f"KL-{k}", "label": v, "description": KL_DESCRIPTIONS[k]}
        for k, v in KL_GRADES.items()
    }
