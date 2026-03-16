# Dataset Guide

## Knee Osteoarthritis Dataset with Severity (KL Grade)

본 프로젝트는 무릎 골관절염(Knee Osteoarthritis) X-ray 이미지를 **Kellgren-Lawrence (KL) Grade** 기준으로 분류합니다.

---

## Kellgren-Lawrence Grade 기준

| Grade | 명칭 | 설명 |
|-------|------|------|
| KL-0 | Normal (정상) | 골관절염 소견 없음 |
| KL-1 | Doubtful (의심) | 의심스러운 골극, 관절 간격 경미한 협소 가능 |
| KL-2 | Minimal (경증) | 명확한 골극, 관절 간격 가능한 협소 |
| KL-3 | Moderate (중등도) | 다수의 골극, 명확한 관절 간격 협소, 경화 |
| KL-4 | Severe (중증) | 대형 골극, 현저한 관절 간격 협소, 심한 경화, 골변형 |

---

## 데이터셋 다운로드

### 방법 1: Kaggle 웹사이트에서 직접 다운로드

1. [Kaggle 데이터셋 페이지](https://www.kaggle.com/datasets/tommyngx/kneeoa) 접속
2. 오른쪽 상단 **Download** 버튼 클릭
3. 압축 해제 후 사용

### 방법 2: Kaggle API를 이용한 다운로드 (권장)

#### Kaggle API 설치 및 인증 설정

```bash
pip install kaggle

# Kaggle API 토큰 설정
# https://www.kaggle.com/settings → API → Create New Token → kaggle.json 다운로드
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 데이터셋 다운로드

```bash
# 다운로드 대상 디렉터리 생성
mkdir -p /home/jayden/data

# Kaggle CLI로 다운로드
kaggle datasets download -d tommyngx/kneeoa -p /home/jayden/data/

# 압축 해제
cd /home/jayden/data
unzip kneeoa.zip -d knee-osteoarthritis-dataset-with-severity
```

### 방법 3: 직접 링크 (로그인 필요)

- **Kaggle**: https://www.kaggle.com/datasets/tommyngx/kneeoa
- **대안 미러**: https://data.mendeley.com/datasets/56rmx5bjcr/1

---

## 데이터셋 구조

다운로드 후 아래 구조가 되어야 합니다:

```
knee-osteoarthritis-dataset-with-severity/
├── train/
│   ├── 0/          # KL-0 Normal    (2,286 images)
│   ├── 1/          # KL-1 Doubtful  (1,046 images)
│   ├── 2/          # KL-2 Minimal   (1,516 images)
│   ├── 3/          # KL-3 Moderate  (  757 images)
│   └── 4/          # KL-4 Severe    (  173 images)
├── val/
│   ├── 0/          (328 images)
│   ├── 1/          (153 images)
│   ├── 2/          (212 images)
│   ├── 3/          (106 images)
│   └── 4/          ( 27 images)
├── test/
│   ├── 0/          (639 images)
│   ├── 1/          (296 images)
│   ├── 2/          (447 images)
│   ├── 3/          (223 images)
│   └── 4/          ( 51 images)
└── auto_test/
    ├── 0/          (604 images)
    ├── 1/          (275 images)
    ├── 2/          (403 images)
    ├── 3/          (200 images)
    └── 4/          ( 44 images)
```

### 데이터셋 통계

| Split | KL-0 | KL-1 | KL-2 | KL-3 | KL-4 | Total |
|-------|------|------|------|------|------|-------|
| train | 2,286 | 1,046 | 1,516 | 757 | 173 | **5,778** |
| val | 328 | 153 | 212 | 106 | 27 | **826** |
| test | 639 | 296 | 447 | 223 | 51 | **1,656** |
| auto_test | 604 | 275 | 403 | 200 | 44 | **1,526** |
| **Total** | **3,857** | **1,770** | **2,578** | **1,286** | **295** | **9,786** |

### 파일명 규칙

- 형식: `<환자ID><방향>.png`
- 예시: `9001695L.png` (L = 왼쪽), `9003126R.png` (R = 오른쪽)

### 클래스 불균형 주의

KL-0(정상)이 KL-4(중증) 대비 약 13배 많아 심각한 클래스 불균형이 존재합니다.
학습 시 `label_smoothing`, `dropout`, 데이터 증강 등의 기법을 활용합니다.

---

## 데이터셋 Config 설정

다운로드 후 `configs/knee_cls.yaml`에서 경로를 수정하세요:

```yaml
path: /home/jayden/data/knee-osteoarthritis-dataset-with-severity
train: train
val: val
test: test
```

경로를 본인 환경에 맞게 수정합니다.
