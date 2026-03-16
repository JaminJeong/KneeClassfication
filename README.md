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

### 1. 설치

```bash
git clone https://github.com/your-username/KneeClassfication.git
cd KneeClassfication

pip install -r requirements.txt
```

### 2. 데이터셋 준비

```bash
# Kaggle API로 다운로드
kaggle datasets download -d tommyngx/kneeoa -p /home/jayden/data/
cd /home/jayden/data && unzip kneeoa.zip -d knee-osteoarthritis-dataset-with-severity
```

데이터셋 상세 안내 → [docs/dataset.md](docs/dataset.md)

### 3. 학습

```bash
python scripts/train.py --model yolo11s-cls.pt --epochs 100
```

### 4. 추론

```bash
python scripts/predict.py \
  --weights runs/classify/knee_cls/weights/best.pt \
  --source path/to/xray.png
```

### 5. Streamlit 데모 실행

```bash
streamlit run example/streamlit_app.py
```

---

## 프로젝트 구조

```
KneeClassfication/
├── configs/
│   └── knee_cls.yaml          # 데이터셋 설정
├── scripts/
│   ├── train.py               # 학습 스크립트
│   ├── predict.py             # 추론 스크립트
│   └── evaluate.py            # 평가 스크립트
├── example/
│   └── streamlit_app.py       # Streamlit 데모 앱
├── docs/
│   ├── usage.md               # 상세 사용법
│   └── dataset.md             # 데이터셋 가이드
├── runs/                      # 학습/추론 결과 (gitignore)
├── models/                    # 모델 가중치 (gitignore)
├── docker-compose.yml         # Docker 환경 설정
├── requirements.txt
└── README.md
```

---

## Docker 사용

```bash
# GPU 환경
docker compose run --rm yolo bash

# CPU 환경
docker compose --profile cpu run --rm yolo-cpu bash

# Streamlit 데모
docker compose --profile demo up streamlit   # → http://localhost:8501

# Jupyter Lab
docker compose --profile jupyter up jupyter  # → http://localhost:8888
```

---

## 스크립트 옵션 요약

### train.py

```
--model      모델 크기 (yolo11n/s/m/l/x-cls.pt)
--epochs     학습 에폭 수 (기본: 100)
--imgsz      입력 이미지 크기 (기본: 224)
--batch      배치 크기 (기본: 32)
--device     GPU 번호 또는 'cpu' (기본: 자동)
--lr0        초기 학습률 (기본: 0.01)
--patience   조기 종료 patience (기본: 20)
```

### predict.py

```
--weights    학습된 가중치 경로 (필수)
--source     입력 이미지 또는 디렉터리 (필수)
--save       결과 이미지 저장
--save-json  예측 결과를 JSON으로 저장
--top-k      상위 K개 클래스 확률 출력 (기본: 5)
```

### evaluate.py

```
--weights    학습된 가중치 경로 (필수)
--split      평가할 데이터 분할 (train/val/test, 기본: test)
--plot       혼동 행렬 등 시각화 저장
```

---

## 참고 자료

- [Ultralytics YOLO 문서](https://docs.ultralytics.com/)
- [UltralyticsDocker 참고 저장소](https://github.com/JaminJeong/UltralyticsDocker)
- [Kaggle 데이터셋](https://www.kaggle.com/datasets/tommyngx/kneeoa)
- [KL Grade 논문](https://doi.org/10.1136/ard.16.4.494)
- 상세 사용법 → [docs/usage.md](docs/usage.md)
- 데이터셋 가이드 → [docs/dataset.md](docs/dataset.md)

---

## 라이선스

본 프로젝트는 연구 및 교육 목적으로 작성되었습니다.
임상 진단 목적으로 사용할 수 없습니다.
