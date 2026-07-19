# NYC Yellow Taxi 분석 및 요금 예측 파이프라인 (2026-05)

2026년 5월 NYC Yellow Taxi 데이터를 정제하고, Pandas/Polars 로딩 비교, 시각화(Seaborn·Plotly), 기술통계·상관분석·t-검정, `scikit-learn Pipeline` 기반 요금(`fare_amount`) 예측 모델 학습까지 수행한 뒤 결과를 `report.md`로 자동 생성하는 End-to-End 분석 파이프라인입니다.

---

## 📂 디렉토리 구조

```text
gray-taxi-data/
├── data/
│   ├── raw/yellow_tripdata_2026-05.parquet        # 원본 데이터 (gitignore)
│   └── processed/yellow_tripdata_2026-05_clean.parquet  # 정제된 데이터셋
├── notebooks/
│   ├── 01_visualization.ipynb                     # Seaborn 정적 + Plotly 인터랙티브 시각화
│   └── 02_statistical_analysis.ipynb              # 기술통계·상관분석·t-검정, 리포트용 요약 export
├── outputs/                                       # 차트, 통계 요약, 모델 지표 등 산출물
├── saved_models/                                  # 학습된 sklearn Pipeline (.pkl, gitignore)
├── src/
│   ├── preprocess.py                              # 원본 raw 데이터 전처리(결측치·중복·이상치 처리)
│   ├── compare_loading.py                         # Pandas vs Polars 로딩 성능 비교
│   ├── pipeline.py                                # ML Pipeline 학습·평가·저장 (요금 예측 Ridge 회귀)
│   └── report.py                                  # Jinja2 기반 report.md 자동 생성
├── templates/
│   └── report_template.md.j2                      # 보고서 자동 생성용 Jinja2 템플릿
├── tests/                                         # src/ 모듈에 대한 pytest 단위 테스트
├── requirements.txt
└── report.md                                      # 자동 렌더링된 최종 분석 및 모델 성능 보고서

## 💻 개발 환경 구축

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 파이프라인 실행 순서

```bash
# 1. 원본 데이터 전처리 (결측치·중복·이상치 처리 → data/processed/*.parquet, outputs/preprocess_summary.md)
python src/preprocess.py

# 2. Pandas vs Polars 로딩 성능 비교 (→ outputs/loading_comparison.md)
python src/compare_loading.py

# 3. 시각화 + 통계분석 노트북 실행 (Jupyter/VS Code에서 직접 실행하거나 아래처럼 일괄 실행)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_visualization.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_statistical_analysis.ipynb

# 4. ML Pipeline 학습 (→ outputs/model_metrics.json, saved_models/taxi_fare_pipeline.pkl)
python src/pipeline.py

# 5. 최종 보고서 자동 생성 (→ report.md)
python src/report.py
```

---

## 🧪 테스트

`src/` 각 모듈에 대한 pytest 단위 테스트가 `tests/`에 있습니다. 실제 대용량 parquet 대신
작은 합성 데이터를 사용해 빠르게 실행되며, 핵심 기능(정상 경로 + 파일 없음 등
기본 예외 상황)이 정상 동작하는지 확인합니다.

```bash
pytest tests/ -v
```

| 파일 | 확인 내용 |
| :--- | :--- |
| `tests/test_preprocess.py` | 중복·구조적결측·이상치 행 제거, 파생변수 생성, 요약 파일 저장, 원본파일 없을 때 처리 |
| `tests/test_compare_loading.py` | Pandas/Polars 비교 결과 저장, 원본파일 없을 때 처리 |
| `tests/test_pipeline.py` | Pipeline 구성, 학습·평가 지표(r2/mse/rmse/mae), joblib·JSON 저장/재로딩 |
| `tests/test_report.py` | 파일 읽기 헬퍼(`_read_text`/`_read_json`), Jinja2 템플릿 렌더링 |
