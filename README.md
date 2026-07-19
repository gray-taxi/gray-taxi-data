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
├── trash/                                         # 팀원별 개별 초안 작업물 보관 (더 이상 유지보수하지 않음)
├── requirements.txt
└── report.md                                      # 자동 렌더링된 최종 분석 및 모델 성능 보고서
```

`trash/`에는 팀원 5명(bonjoon, sangyoon, jeongmungi, heewon, heeyoon)이 각자 진행했던 개별 스크립트/노트북 원본이 보존되어 있습니다. 최종 제출물은 이를 하나의 파이프라인으로 통합·정리한 위 구조이며, ML Pipeline 대표 주제는 **요금(fare_amount) 예측**으로 수렴했습니다.

---

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

## 👥 팀 구성 및 역할

| 담당 | 원래 작업 | 최종 반영 |
| :--- | :--- | :--- |
| sangyoon | 데이터 전처리, Pandas/Polars 로딩 비교, 정체예측·공항요금제 판별 모델 | 전처리·로딩 비교 로직을 `src/preprocess.py`, `src/compare_loading.py`로 통합 반영 |
| jeongmungi | 시각화 노트북, 통계분석 노트북, 요금예측(Ridge) ML Pipeline, report.py 초안 | **대표 주제로 채택** — `notebooks/`, `src/pipeline.py`, `src/report.py`, `templates/`의 뼈대로 반영 |
| bonjoon | 높은 팁 여부 예측 (RandomForest) | 최종 리포트에서는 생략 (`trash/pipeline/bonjoon/`에 원본 보존) |
| heewon | 기술통계·상관계수·t-test | 최종 리포트에서는 생략 (`trash/visualize/heewon/`에 원본 보존) |
| heeyoon | Seaborn/Plotly 시각화 | 최종 리포트에서는 생략 (`trash/visualize/heeyoon/`에 원본 보존) |

팀 최종 산출물은 **요금 예측 하나의 주제로 수렴**하도록 결정되어, 나머지 개별 분석(높은 팁 예측, 정체예측, 공항요금제 판별)은 `report.md`에 포함하지 않았습니다. 해당 코드는 `trash/`에 원형 그대로 보존되어 있습니다.
