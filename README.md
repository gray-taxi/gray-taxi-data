# NYC Yellow Taxi 분석 및 ML 예측 통합 파이프라인 (2026-05)

본 프로젝트는 2026년 5월 NYC Yellow Taxi 데이터를 정제하고, 기술통계 분석 및 시각화, 그리고 팀원들의 다양한 기계학습(ML) 모델링 결과물을 통합하여 관리 및 자동 분석 보고서(`report.md`) 생성을 일괄 수행할 수 있도록 구성한 통합 파이프라인 프레임워크입니다.

---

## 📂 최종 디렉토리 구조

```text
gray-taxi-data/
├── data/                                 # 정제 데이터 저장 폴더
│   └── yellow_tripdata_2026-05_clean.parquet  # 정제된 데이터셋 (parquet 형식)
├── notebooks/                            # 분석 및 EDA 노트북 폴더
├── outputs/                              # 시각화 이미지 및 모델 평가 데이터
│   ├── avg_tip_by_period.png             # 시간대별 평균 팁 차트 (정적)
│   ├── avg_tip_by_period_interactive.html # 시간대별 평균 팁 차트 (인터랙티브)
│   ├── taxi_demand_by_hour.png           # 시간대별 이용량 차트 (정적)
│   ├── taxi_demand_by_hour_interactive.html # 시간대별 이용량 차트 (인터랙티브)
│   ├── model_metrics.json                # 4가지 모델의 통합 평가 지표 (JSON)
│   └── stat_summary.md                   # 기술통계 및 상관계수 요약 마크다운 캐시
├── saved_models/                         # 학습이 완료된 4가지 통합 모델 저장소
│   ├── taxi_airport_pipeline.pkl         # 공항 요금제 판별 이진 분류 모델 (Logistic Regression)
│   ├── taxi_congestion_pipeline.pkl      # 운행 정체 이진 분류 모델 (Logistic Regression)
│   ├── taxi_fare_pipeline.pkl            # 요금 예측 회귀 모델 (Ridge)
│   └── taxi_high_tip_pipeline.pkl        # 높은 팁 여부 예측 분류 모델 (RandomForest)
├── src/                                  # 공통 실행 모듈 (통합 파이프라인 소스코드)
│   ├── preprocess.py                     # 데이터 전처리 모듈
│   ├── pipeline.py                       # ML 모델 파이프라인 학습 및 평가 모듈
│   └── report.py                         # 최종 마크다운 보고서 자동 생성 모듈
├── templates/                            # 보고서 생성을 위한 Jinja2 템플릿 폴더
│   └── report_template.md.j2             # Jinja2 보고서 템플릿 파일
├── pipeline/                             # (팀원별 레거시/개별 파이프라인 작업 폴더)
├── visualize/                            # (팀원별 레거시/개별 시각화 작업 폴더)
├── requirements.txt                      # 통합 패키지 의존성 파일
└── report.md                             # 자동 렌더링된 최종 분석 및 모델 성능 결과 보고서
```

---

## 💻 개발 환경 구축 가이드

동일한 환경을 재현하여 모델을 다시 학습시키거나 스크립트를 실행하려면 아래 절차를 따르십시오.

### 1. 가상환경 생성 및 활성화

프로젝트 루트 디렉토리에서 터미널(PowerShell 등)을 열고 가상환경을 생성 및 활성화합니다.

*   **Windows**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
*   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 2. 필수 의존성 패키지 설치

`requirements.txt`에 명시된 필수 패키지 버전을 일괄 설치합니다.
```bash
pip install -r requirements.txt
```

---

## 🚀 파이프라인 실행 가이드

전체 단계를 개별적으로 또는 일괄적으로 실행할 수 있습니다. (가상환경이 활성화된 상태에서 실행해 주세요.)

### 1단계: 데이터 전처리 실행 (선택 사항)
원본 raw 데이터가 `data/raw/` 하위에 위치하는 경우, 데이터 정제 가이드에 따라 클린 데이터셋을 새로 구축합니다. (이미 `data/yellow_tripdata_2026-05_clean.parquet`이 준비되어 있다면 이 단계는 건너뛰어도 좋습니다.)
```bash
python src/preprocess.py
```

### 2단계: 기계학습 모델 통합 학습
공통 모듈인 `src/pipeline.py`를 사용하여 4가지 예측 타스크 모델을 선택적으로 또는 일괄적으로 학습할 수 있습니다. 학습이 완료되면 평가지표가 `outputs/model_metrics.json`에 업데이트되고 학습 완료된 피클 파일(`.pkl`)이 `saved_models/`에 저장됩니다.

*   **요금 예측 회귀 모델만 학습**:
    ```bash
    python src/pipeline.py --task fare
    ```
*   **운행 정체 이진 분류 모델만 학습**:
    ```bash
    python src/pipeline.py --task congestion
    ```
*   **공항요금제 판별 이진 분류 모델만 학습**:
    ```bash
    python src/pipeline.py --task airport
    ```
*   **높은 팁 여부 예측 이진 분류 모델만 학습**:
    ```bash
    python src/pipeline.py --task tip
    ```
*   **전체 모델 일괄 학습 (기본 설정)**:
    ```bash
    python src/pipeline.py --task all
    ```

### 3단계: 최종 통합 보고서(`report.md`) 자동 생성
Jinja2 템플릿 엔진을 사용하여 요약 기술통계 테이블(`outputs/stat_summary.md`), 차트 이미지 경로, 학습 완료된 모델 평가지표(`outputs/model_metrics.json`)를 결합하여 종합 보고서를 프로젝트 루트에 빌드합니다.
```bash
python src/report.py
```

*   **결과**: 프로젝트 루트에 최종 [report.md](file:///C:/Users/JMG/Desktop/skala/gray-taxi-data/report.md) 문서가 생성 및 업데이트됩니다.
