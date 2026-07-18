# NYC Yellow Taxi 분석 및 요금 예측 파이프라인 (2026-05)

본 프로젝트는 2026년 5월 NYC Yellow Taxi 데이터를 정제 및 기술통계 분석, 독립표본 t-검정, 데이터 시각화(Seaborn & Plotly), 그리고 `scikit-learn Pipeline` 기반 요금 예측 모델 학습까지 일련의 데이터 수집-처리-분석-예측 사이클을 일괄 수행하고 보고서 작성을 자동화한 분석 파이프라인입니다.

---

## 디렉토리 구조

```text
gray-taxi-data/
├── data/
│   └── yellow_tripdata_2026-05_clean.parquet  # 전처리 완료된 학습 데이터셋
├── notebooks/
│   ├── 01_visualization.ipynb                 # 정적(Seaborn) & 동적(Plotly) 시각화 노트북
│   └── 02_statistical_analysis.ipynb          # 왜도/첨도 기술통계 및 3대 t-test 검정 노트북
├── outputs/
│   ├── avg_tip_by_period.png                  # 시간대별 평균 팁 차트 (정적)
│   ├── correlation_matrix.png                 # 상관계수 히트맵 (정적)
│   ├── taxi_demand_by_hour.png                # 시간대별 이용량 차트 (정적)
│   ├── tip_by_passenger_count.png             # 승객수별 평균 팁 차트 (정적)
│   ├── avg_tip_by_period_interactive.html     # 시간대별 평균 팁 차트 (인터랙티브)
│   ├── taxi_demand_by_hour_interactive.html   # 시간대별 이용량 차트 (인터랙티브)
│   ├── trip_distance_vs_fare_interactive.html # 주행거리 vs 요금 산점도 (인터랙티브)
│   ├── model_metrics.json                     # ML 파이프라인 평가지표 백업
│   └── stat_summary.md                        # 기술통계 및 상관행렬 테이블 캐시
├── saved_models/
│   └── taxi_fare_pipeline.pkl                 # 학습된 sklearn 전처리+Ridge 예측 파이프라인
├── src/
│   ├── __init__.py
│   ├── preprocess.py                          # 원본 raw 데이터 전처리 파이프라인 스크립트
│   ├── pipeline.py                            # ML 파이프라인 빌드, 학습 및 저장 스크립트
│   └── report.py                              # Jinja2 기반 report.md 생성 스케줄/자동화 스크립트
├── templates/
│   └── report_template.md.j2                  # 보고서 자동 생성을 위한 Jinja2 마크다운 템플릿
├── .env.template                              # 환경 변수 설정 템플릿
├── .gitignore                                 # Git 제외 항목 파일
├── requirements.txt                           # 재현 가능한 개발 환경용 패키지 목록
└── report.md                                  # 자동화 렌더링된 최종 분석 및 모델 성능 결과 보고서
```

---

## 개발 환경 구축 가이드

동일한 환경을 재현하여 모델을 다시 학습시키거나 노트북을 실행하려면 아래 절차를 따르십시오.

### 1. 가상환경 생성 및 활성화
프로젝트 루트 디렉토리에서 터미널을 열고 가상환경을 생성 및 활성화합니다.

* **Windows**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 2. 필수 의존성 패키지 설치
`requirements.txt`에 명시된 필수 패키지 버전을 일괄 설치합니다.
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 복사 (선택 사항)
필요시 `.env.template` 파일을 `.env`로 복사하여 프로젝트 경로 변수들을 재지정할 수 있습니다.
```bash
copy .env.template .env
```

---

## 파이프라인 실행 가이드

전체 단계를 개별적으로 또는 일괄적으로 실행할 수 있습니다.

### 1단계: 분석 및 시각화 노트북 실행
VS Code 혹은 Jupyter Notebook 환경에서 [notebooks/01_visualization.ipynb](file:///C:/Users/JMG/Desktop/skala/gray-taxi-data/notebooks/01_visualization.ipynb)와 [notebooks/02_statistical_analysis.ipynb](file:///C:/Users/JMG/Desktop/skala/gray-taxi-data/notebooks/02_statistical_analysis.ipynb)를 직접 실행하여 정적/동적 시각화 이미지와 독립표본 t-검정 결과를 갱신하고 `outputs/stat_summary.md`를 내보냅니다.

### 2단계: 기계학습 모델 학습 및 모델 저장
수치 피처 표준화(`StandardScaler`), 카테고리 원핫 인코딩(`OneHotEncoder`) 및 L2 규제 회귀 모델(`Ridge`)을 하나의 파이프라인 객체로 묶어 학습을 시작합니다.
```bash
python src/pipeline.py
```
* **결과**: `outputs/model_metrics.json`과 `saved_models/taxi_fare_pipeline.pkl`이 생성 및 업데이트됩니다.

### 3단계: 최종 보고서(report.md) 자동 생성
Jinja2 템플릿 엔진을 사용하여 위 단계들에서 추출된 가설 검정 결론, 요약 통계량 테이블, 차트 이미지 경로, 모델 평가지표 등을 결합하여 종합 마크다운 문서인 `report.md`를 렌더링합니다.
```bash
python src/report.py
```
* **결과**: 프로젝트 루트에 최종 [report.md](file:///C:/Users/JMG/Desktop/skala/gray-taxi-data/report.md) 문서가 업데이트됩니다.