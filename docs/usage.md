# 사용법 가이드

## 목차

1. [환경 설정](#환경-설정)
2. [학습 (Training)](#학습-training)
3. [추론 (Inference)](#추론-inference)
4. [평가 (Evaluation)](#평가-evaluation)
5. [Streamlit 데모](#streamlit-데모)
6. [Docker 사용법](#docker-사용법)

---

## 환경 설정

### 로컬 환경 (Python 가상환경)

```bash
# 저장소 클론
git clone https://github.com/your-username/KneeClassfication.git
cd KneeClassfication

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 데이터셋 경로 설정

`configs/knee_cls.yaml`에서 `path`를 본인 환경에 맞게 수정합니다:

```yaml
path: /home/jayden/data/knee-osteoarthritis-dataset-with-severity
```

---

## 학습 (Training)

### 기본 학습

```bash
# 기본 설정으로 학습 (YOLO11n-cls, 100 epochs, 224×224)
python scripts/train.py
```

### 주요 옵션

```bash
python scripts/train.py \
  --model yolo11s-cls.pt \   # 모델 크기: n < s < m < l < x
  --epochs 150 \             # 학습 에폭 수
  --imgsz 224 \              # 입력 이미지 크기
  --batch 32 \               # 배치 크기
  --device 0 \               # GPU ID ('' = 자동, 'cpu' = CPU)
  --lr0 0.001 \              # 초기 학습률
  --patience 30 \            # 조기 종료 patience
  --optimizer AdamW          # 옵티마이저
```

### 모델 크기 선택 가이드

| 모델 | 파라미터 | 속도 | 권장 용도 |
|------|---------|------|---------|
| `yolo11n-cls.pt` | 1.5M | 매우 빠름 | 빠른 실험, 엣지 기기 |
| `yolo11s-cls.pt` | 5.5M | 빠름 | 균형 잡힌 성능 |
| `yolo11m-cls.pt` | 11M | 보통 | 높은 정확도 |
| `yolo11l-cls.pt` | 25M | 느림 | 최고 성능 |
| `yolo11x-cls.pt` | 56M | 매우 느림 | 연구용 |

### 학습 재시작 (Resume)

```bash
python scripts/train.py --resume runs/classify/knee_cls/weights/last.pt
```

### 결과 저장 위치

```
runs/
└── classify/
    └── knee_cls/
        ├── weights/
        │   ├── best.pt   ← 최고 성능 가중치
        │   └── last.pt   ← 마지막 체크포인트
        ├── results.csv
        ├── confusion_matrix.png
        └── ...
```

---

## 추론 (Inference)

### 단일 이미지

```bash
python scripts/predict.py \
  --weights runs/classify/knee_cls/weights/best.pt \
  --source path/to/knee_xray.png
```

### 디렉터리 전체

```bash
python scripts/predict.py \
  --weights runs/classify/knee_cls/weights/best.pt \
  --source /home/jayden/data/knee-osteoarthritis-dataset-with-severity/test/
```

### 결과 저장

```bash
# 이미지 결과 저장
python scripts/predict.py \
  --weights best.pt \
  --source images/ \
  --save

# JSON으로 예측 결과 저장
python scripts/predict.py \
  --weights best.pt \
  --source images/ \
  --save-json
```

### 출력 예시

```
[9001695L.png]
  Prediction : KL-2 (Minimal) — 78.4%
  KL-0 Normal       ████                 18.3%
  KL-1 Doubtful     █                    3.1%
  KL-2 Minimal      ████████████████     78.4%
  KL-3 Moderate     ██                   8.2%
  KL-4 Severe                            2.0%
```

---

## 평가 (Evaluation)

### 테스트 셋 평가

```bash
python scripts/evaluate.py \
  --weights runs/classify/knee_cls/weights/best.pt \
  --split test
```

### 혼동 행렬(Confusion Matrix) 저장

```bash
python scripts/evaluate.py \
  --weights best.pt \
  --split test \
  --plot
```

### 출력 예시

```
============================================================
Evaluation Results
============================================================
  Top-1 Accuracy : 0.7234 (72.34%)
  Top-5 Accuracy : 1.0000 (100.00%)
============================================================

Class                    Correct    Total   Accuracy
-------------------------------------------------------
  Normal (KL-0)              591      639      92.5%
  Doubtful (KL-1)            178      296      60.1%
  Minimal (KL-2)             312      447      69.8%
  Moderate (KL-3)            149      223      66.8%
  Severe (KL-4)               28       51      54.9%
-------------------------------------------------------
  Overall                   1258     1656      76.0%
```

---

## Streamlit 데모

### 로컬 실행

```bash
# 기본 실행 (모델 경로는 runs/classify/knee_cls/weights/best.pt)
streamlit run example/streamlit_app.py

# 커스텀 모델 경로 지정
KNEE_MODEL_PATH=/path/to/best.pt streamlit run example/streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속

### 기능

- 드래그 앤 드롭 또는 파일 선택으로 X-ray 업로드
- KL Grade 예측 및 신뢰도 표시
- 전체 클래스 확률 바 차트 시각화
- 데이터셋 샘플 이미지로 빠른 테스트

---

## Docker 사용법

### GPU 환경 (학습/추론)

```bash
# GPU 컨테이너 실행 (대화형 셸)
docker compose run --rm yolo bash

# 컨테이너 내에서 학습
python scripts/train.py --model yolo11s-cls.pt --epochs 100
```

### CPU 환경

```bash
docker compose --profile cpu run --rm yolo-cpu bash
```

### Streamlit 데모 (Docker)

```bash
docker compose --profile demo up streamlit
```

브라우저에서 `http://localhost:8501` 접속

### Jupyter Lab

```bash
docker compose --profile jupyter up jupyter
```

브라우저에서 `http://localhost:8888` 접속

---

## 자주 묻는 질문

**Q: CUDA out of memory 오류가 발생합니다.**

A: 배치 크기를 줄이거나 더 작은 모델을 사용하세요:
```bash
python scripts/train.py --batch 16 --model yolo11n-cls.pt
```

**Q: 학습이 느립니다.**

A: `--device` 옵션으로 GPU를 명시적으로 지정하세요:
```bash
python scripts/train.py --device 0
```

**Q: 클래스 불균형 문제를 어떻게 처리하나요?**

A: 기본 설정에 `label_smoothing=0.1`, `dropout=0.2`가 포함되어 있습니다.
추가로 Ultralytics의 자동 augmentation이 적용됩니다.
