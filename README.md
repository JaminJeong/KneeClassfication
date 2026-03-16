# KneeClassification

**YOLO 기반 무릎 골관절염 중증도 분류 (Knee Osteoarthritis KL Grade Classification)**

Ultralytics YOLO 프레임워크를 사용하여 무릎 X-ray 이미지에서 **Kellgren-Lawrence (KL) Grade**를 자동으로 분류하는 딥러닝 프로젝트입니다.

---

## 개요

| 항목 | 내용 |
|------|------|
| 프레임워크 | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) |
| 태스크 | 이미지 분류 (Image Classification) |
| 클래스 수 | 5 (KL-0 ~ KL-4) |
| 데이터셋 | Knee OA Dataset with Severity (9,786 images) |
| 입력 | 무릎 X-ray 이미지 (PNG/JPG) |
| 출력 | KL Grade (0~4) + 신뢰도 |

## KL Grade 분류 기준

| Grade | 명칭 | 설명 |
|-------|------|------|
| **KL-0** | Normal | 정상, 골관절염 소견 없음 |
| **KL-1** | Doubtful | 의심 단계, 경미한 골극 가능 |
| **KL-2** | Minimal | 명확한 골극, 관절 간격 협소 가능 |
| **KL-3** | Moderate | 중등도, 명확한 관절 간격 협소 및 경화 |
| **KL-4** | Severe | 중증, 심한 골변형 및 관절 간격 협소 |

---

## 빠른 시작

### 1. Docker 환경 진입

```bash
# GPU 환경 (권장)
docker compose run --rm yolo bash

# CPU 환경
docker compose --profile cpu run --rm yolo-cpu bash
```

### 2. 데이터셋 준비

```bash
# Kaggle API로 다운로드
kaggle datasets download -d tommyngx/kneeoa -p /home/jayden/data/
cd /home/jayden/data && unzip kneeoa.zip -d knee-osteoarthritis-dataset-with-severity
```

데이터셋 상세 안내 → [docs/dataset.md](docs/dataset.md)

### 3. 학습

두 가지 방식을 모두 지원합니다. Docker 컨테이너 내부에서 실행합니다.

```bash
# Python 스크립트
python scripts/train.py --model yolo11s-cls.pt --epochs 100

# Shell 스크립트 (Ultralytics CLI)
bash scripts/train.sh --model yolo11s-cls.pt --epochs 100
```

### 4. 추론

```bash
# Python 스크립트
python scripts/predict.py \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/0/9003175L.png

# Shell 스크립트 (Ultralytics CLI)
bash scripts/predict.sh \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --source /workspace/data/knee-osteoarthritis-dataset-with-severity/test/
```

### 5. 평가

```bash
# Python 스크립트
python scripts/evaluate.py \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt

# Shell 스크립트 (Ultralytics CLI)
bash scripts/evaluate.sh \
  --weights /workspace/runs/classify/knee_cls/weights/best.pt \
  --split test --plot True
```

### 6. Streamlit 데모

```bash
docker compose --profile demo up streamlit  # → http://localhost:8501
```

---

## 프로젝트 구조

```
KneeClassfication/
├── configs/
│   └── knee_cls.yaml          # 데이터셋 메타 정보 (참고용)
├── scripts/
│   ├── train.py               # 학습 — Python API
│   ├── train.sh               # 학습 — Ultralytics CLI
│   ├── predict.py             # 추론 — Python API
│   ├── predict.sh             # 추론 — Ultralytics CLI
│   ├── evaluate.py            # 평가 — Python API (per-class 정확도 포함)
│   └── evaluate.sh            # 평가 — Ultralytics CLI
├── example/
│   └── streamlit_app.py       # Streamlit 데모 앱
├── docs/
│   ├── usage.md               # 상세 사용법
│   └── dataset.md             # 데이터셋 가이드 및 다운로드
├── runs/                      # 학습/추론 결과 (gitignore)
├── models/                    # 모델 가중치 (gitignore)
├── docker-compose.yml         # Docker 환경 설정
├── requirements.txt
└── README.md
```

---

## 실행 방식 비교

| 항목 | Python 스크립트 | Shell 스크립트 (CLI) |
|------|----------------|-------------------|
| 파일 | `scripts/*.py` | `scripts/*.sh` |
| `--data` | 데이터셋 루트 디렉터리 | 데이터셋 루트 디렉터리 |
| 결과 저장 | `/workspace/runs/classify/` | `/workspace/runs/classify/` |
| 특징 | 세밀한 제어, 코드 확장 | 빠른 실험, CLI 직접 사용 |

> `--data`는 두 방식 모두 **YAML 파일이 아닌 데이터셋 루트 디렉터리 경로**를 사용합니다.

---

## Docker 전체 서비스

```bash
# GPU 학습/추론 (대화형 셸)
docker compose run --rm yolo bash

# CPU 전용
docker compose --profile cpu run --rm yolo-cpu bash

# Streamlit 데모
docker compose --profile demo up streamlit   # → http://localhost:8501

# Jupyter Lab
docker compose --profile jupyter up jupyter  # → http://localhost:8888
```

---

## 스크립트 옵션 요약

### train.py / train.sh

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `yolo11n-cls.pt` | 모델 크기 (n/s/m/l/x) |
| `--data` | `/workspace/data/...` | 데이터셋 루트 디렉터리 |
| `--epochs` | `100` | 학습 에폭 수 |
| `--imgsz` | `224` | 입력 이미지 크기 |
| `--batch` | `32` | 배치 크기 |
| `--device` | 자동 | GPU 번호 또는 `cpu` |
| `--project` | `/workspace/runs/classify` | 결과 저장 디렉터리 |
| `--name` | `knee_cls` | 실험 이름 |
| `--lr0` | `0.01` | 초기 학습률 |
| `--patience` | `20` | 조기 종료 patience |

### predict.py / predict.sh

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--weights` | *(필수)* | 학습된 가중치 경로 |
| `--source` | *(필수)* | 이미지 또는 디렉터리 경로 |
| `--save` | `False` | 결과 이미지 저장 |
| `--save-json` | `False` | 예측 결과 JSON 저장 (py only) |
| `--top-k` | `5` | 상위 K개 확률 출력 (py only) |
| `--project` | `/workspace/runs/predict` | 결과 저장 디렉터리 |

### evaluate.py / evaluate.sh

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--weights` | *(필수)* | 학습된 가중치 경로 |
| `--data` | `/workspace/data/...` | 데이터셋 루트 디렉터리 |
| `--split` | `test` | 평가 분할 (train/val/test) |
| `--plot` | `False` | 혼동 행렬 시각화 저장 |
| `--project` | `/workspace/runs/evaluate` | 결과 저장 디렉터리 |

---

## 참고 자료

- [Ultralytics YOLO 문서](https://docs.ultralytics.com/)
- [Ultralytics Classification Dataset 형식](https://docs.ultralytics.com/datasets/classify/)
- [UltralyticsDocker 참고 저장소](https://github.com/JaminJeong/UltralyticsDocker)
- [Kaggle 데이터셋](https://www.kaggle.com/datasets/tommyngx/kneeoa)
- [KL Grade 논문](https://doi.org/10.1136/ard.16.4.494)
- 상세 사용법 → [docs/usage.md](docs/usage.md)
- 데이터셋 가이드 → [docs/dataset.md](docs/dataset.md)

---

## 라이선스

본 프로젝트는 연구 및 교육 목적으로 작성되었습니다.
임상 진단 목적으로 사용할 수 없습니다.
