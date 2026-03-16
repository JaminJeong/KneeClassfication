# 사용법 가이드

## 목차

1. [환경 설정](#환경-설정)
2. [실행 방식 비교](#실행-방식-비교)
3. [학습 (Training)](#학습-training)
4. [추론 (Inference)](#추론-inference)
5. [평가 (Evaluation)](#평가-evaluation)
6. [Streamlit 데모](#streamlit-데모)
7. [Docker 사용법](#docker-사용법)
8. [자주 묻는 질문](#자주-묻는-질문)

---

## 환경 설정

### 로컬 환경 (Python 가상환경)

```bash
git clone https://github.com/your-username/KneeClassfication.git
cd KneeClassfication

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Docker 환경 (권장)

```bash
# GPU 컨테이너 진입
docker compose run --rm yolo bash

# CPU 전용
docker compose --profile cpu run --rm yolo-cpu bash
```

Docker 컨테이너 내 경로 구조:

| 호스트 경로 | 컨테이너 경로 |
|------------|--------------|
| `./scripts` | `/workspace/scripts` |
| `./runs` | `/workspace/runs` |
| `/home/jayden/data` | `/workspace/data` |

---

## 실행 방식 비교

본 프로젝트는 **Python 스크립트**와 **Shell 스크립트(Ultralytics CLI)** 두 가지 방식을 지원합니다.

| 항목 | Python 스크립트 | Shell 스크립트 (CLI) |
|------|----------------|-------------------|
| 파일 | `scripts/*.py` | `scripts/*.sh` |
| 실행 방법 | `python scripts/train.py` | `bash scripts/train.sh` |
| `--data` 인자 | 데이터셋 루트 디렉터리 | 데이터셋 루트 디렉터리 |
| 결과 저장 위치 | `/workspace/runs/classify/` | `/workspace/runs/classify/` |
| 특징 | 세밀한 제어, 코드 확장 용이 | 빠른 실험, 별도 설치 불필요 |

> **주의:** `--data`는 두 방식 모두 **YAML 파일이 아닌 데이터셋 루트 디렉터리 경로**를 사용합니다.
> Ultralytics classification은 `train/` `val/` `test/` 하위 폴더 구조를 직접 읽습니다.

---

## 학습 (Training)

### Python 스크립트

```bash
# 기본 학습 (Docker 내부, 데이터 경로 자동 설정)
python scripts/train.py

# 주요 옵션 지정
python scripts/train.py \
  --model yolo11s-cls.pt \
  --epochs 100 \
  --imgsz 224 \
  --batch 32 \
  --device 0 \
  --lr0 0.001 \
  --patience 30 \
  --optimizer AdamW \
  --name my_experiment

# 로컬 환경에서 데이터 경로 지정
python scripts/train.py \
  --data /path/to/knee-osteoarthritis-dataset-with-severity \
  --project /path/to/output/runs/classify
```

### Shell 스크립트 (Ultralytics CLI)

```bash
# 기본 학습
bash scripts/train.sh

# 주요 옵션 지정
bash scripts/train.sh \
  --model yolo11s-cls.pt \
  --epochs 100 \
  --batch 32 \
  --device 0 \
  --name my_experiment

# 도움말
bash scripts/train.sh --help
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
# Python
python scripts/train.py --resume /workspace/runs/classify/knee_cls/weights/last.pt

# Shell
bash scripts/train.sh  # 같은 name이면 자동으로 이어받지 않음
                       # resume은 Python 스크립트를 사용하세요
```

### 결과 저장 위치

```
/workspace/runs/classify/
└── knee_cls/                  ← --name 으로 지정한 실험명
    ├── weights/
    │   ├── best.pt            ← 최고 성능 가중치
    │   └── last.pt            ← 마지막 체크포인트
    ├── results.csv
    ├── confusion_matrix.png
    └── args.yaml
```

---

## 추론 (Inference)

### Python 스크립트

```bash
# 단일 이미지
python scripts/predict.py \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/0/9003175L.png

# 디렉터리 전체
python scripts/predict.py \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/

# 결과 이미지 저장
python scripts/predict.py \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --source /workspace/data/.../test/ \
  --save

# JSON으로 예측 결과 저장
python scripts/predict.py \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --source /workspace/data/.../test/ \
  --save-json
```

### Shell 스크립트 (Ultralytics CLI)

```bash
# 단일 이미지
bash scripts/predict.sh \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/0/9003175L.png

# 디렉터리 전체 + 결과 저장
bash scripts/predict.sh \
  --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/ \
  --save True

# 도움말
bash scripts/predict.sh --help
```

### 출력 예시

```
[9003175L.png]
  Prediction : KL-0 (Normal) — 39.9%
  KL-0 Normal       ███████              39.9%
  KL-2 Minimal      █████                29.9%
  KL-1 Doubtful     ███                  17.0%
  KL-3 Moderate     ██                   11.7%
  KL-4 Severe                            1.4%
```

---

## 평가 (Evaluation)

### Python 스크립트

```bash
# 테스트 셋 평가 (기본)
python scripts/evaluate.py \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt

# val 셋 + 혼동 행렬 저장
python scripts/evaluate.py \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --split val \
  --plot
```

### Shell 스크립트 (Ultralytics CLI)

```bash
# 테스트 셋 평가 (기본)
bash scripts/evaluate.sh \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt

# val 셋 + 혼동 행렬 저장
bash scripts/evaluate.sh \
  --split val \
  --plot True

# 도움말
bash scripts/evaluate.sh --help
```

### 출력 예시

```
============================================================
Evaluation Results
============================================================
  Top-1 Accuracy : 0.7234 (72.34%)
  Top-5 Accuracy : 1.0000 (100.00%)
============================================================

Computing per-class accuracy on 'test' split...

Class                      Correct    Total   Accuracy
-------------------------------------------------------
  Normal (KL-0)                591      639     92.5%
  Doubtful (KL-1)              178      296     60.1%
  Minimal (KL-2)               312      447     69.8%
  Moderate (KL-3)              149      223     66.8%
  Severe (KL-4)                 28       51     54.9%
-------------------------------------------------------
  Overall                     1258     1656     76.0%
```

---

## Streamlit 데모

### 로컬 실행

```bash
# 기본 실행 (모델 경로: /workspace/runs/classify/knee_cls/weights/best.pt)
streamlit run example/streamlit_app.py

# 커스텀 모델 경로 지정
KNEE_MODEL_PATH=/path/to/best.pt streamlit run example/streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속

### Docker 실행

```bash
docker compose --profile demo up streamlit
```

브라우저에서 `http://localhost:8501` 접속

### 기능

- 드래그 앤 드롭 또는 파일 선택으로 X-ray 업로드
- KL Grade 예측 및 신뢰도 표시
- 전체 클래스 확률 바 차트 시각화
- 데이터셋 샘플 이미지로 빠른 테스트

---

## Docker 사용법

### 컨테이너 진입 후 직접 실행

```bash
# 컨테이너 진입
docker compose run --rm yolo bash

# 컨테이너 내부에서 실행
cd /workspace

# Python 스크립트
python scripts/train.py --epochs 100
python scripts/predict.py --weights /workspace/runs/classify/knee_cls/weights/best.pt --source ...
python scripts/evaluate.py --weights /workspace/runs/classify/knee_cls/weights/best.pt

# Shell 스크립트 (Ultralytics CLI)
bash scripts/train.sh --epochs 100
bash scripts/predict.sh --weights /workspace/runs/classify/knee_cls/weights/best.pt --source ...
bash scripts/evaluate.sh
```

### 호스트에서 한 번에 실행

```bash
# Python 스크립트
docker compose run --rm yolo python scripts/train.py --epochs 100 --device 0

# Shell 스크립트
docker compose run --rm yolo bash scripts/train.sh --epochs 100

# 평가
docker compose run --rm yolo bash scripts/evaluate.sh --split test --plot True
```

### CPU 환경

```bash
docker compose --profile cpu run --rm yolo-cpu bash scripts/train.sh --device cpu
```

### Jupyter Lab

```bash
docker compose --profile jupyter up jupyter
```

브라우저에서 `http://localhost:8888` 접속

---

## 자주 묻는 질문

**Q: `--data`에 YAML 파일을 넣으면 에러가 납니다.**

A: Ultralytics classification은 YAML을 지원하지 않습니다. 데이터셋 루트 디렉터리를 전달하세요.

```bash
# 잘못된 예
python scripts/train.py --data configs/knee_cls.yaml

# 올바른 예
python scripts/train.py --data /workspace/data/knee-osteoarthritis-dataset-with-severity
```

**Q: 결과가 `/ultralytics/runs/...`에 저장됩니다.**

A: `--project`에 절대 경로를 명시하세요. 기본값은 이미 `/workspace/runs/...`로 설정되어 있습니다.

```bash
python scripts/train.py --project /workspace/runs/classify
```

**Q: CUDA out of memory 오류가 발생합니다.**

A: 배치 크기를 줄이거나 더 작은 모델을 사용하세요.

```bash
python scripts/train.py --batch 16 --model yolo11n-cls.pt
bash scripts/train.sh --batch 16 --model yolo11n-cls.pt
```

**Q: 학습이 느립니다.**

A: `--device` 옵션으로 GPU를 명시적으로 지정하세요.

```bash
python scripts/train.py --device 0
bash scripts/train.sh --device 0
```

**Q: 클래스 불균형 문제를 어떻게 처리하나요?**

A: 기본 설정에 `dropout=0.2`, `label_smoothing=0.1`이 포함되어 있으며, Ultralytics의 자동 augmentation이 적용됩니다.
