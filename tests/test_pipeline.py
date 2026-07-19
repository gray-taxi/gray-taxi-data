"""
test_pipeline.py -- src/pipeline.py 간단 단위 테스트

기능: 소규모 합성 정제 데이터로 Pipeline 구성·학습·평가·저장이 정상 동작하는지 확인한다.

변경내역
  2026-07-19  최초 작성
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.pipeline import build_pipeline, train_and_evaluate, save_pipeline, save_metrics


# ML Pipeline이 요구하는 컬럼 스키마를 갖춘, 거리에 비례하는 요금(fare_amount)을 가진
# 합성 데이터를 생성한다 (R2가 유의미하게 나오도록 약한 노이즈만 추가).
def _make_clean_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    trip_distance = rng.uniform(0.5, 10.0, n)
    trip_duration_min = rng.uniform(5, 40, n)
    return pd.DataFrame({
        "trip_distance": trip_distance,
        "trip_duration_min": trip_duration_min,
        "passenger_count": rng.integers(1, 5, n),
        "average_speed_mph": trip_distance / (trip_duration_min / 60),
        "VendorID": rng.choice([1, 2], n),
        "RatecodeID": rng.choice([1, 2], n),
        "pickup_hour": rng.integers(0, 24, n),
        "pickup_day_of_week": rng.integers(0, 7, n),
        "time_period": rng.choice(["daytime", "night"], n),
        "is_weekend": rng.integers(0, 2, n),
        "fare_amount": trip_distance * 3.0 + rng.normal(0, 0.5, n),
    })


# build_pipeline()이 (preprocessor, regressor) 2단계로 구성된 sklearn Pipeline을 반환하는지 확인한다.
def test_build_pipeline_returns_sklearn_pipeline():
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "regressor"]


# 학습 후 반환된 metrics가 r2/mse/rmse/mae 키를 모두 갖고, 선형관계가 뚜렷한 합성데이터에서 R2가 0.5를 넘는지 확인한다.
def test_train_and_evaluate_returns_expected_metrics():
    df = _make_clean_df()
    pipeline, metrics = train_and_evaluate(df)

    assert isinstance(pipeline, Pipeline)
    assert set(metrics.keys()) == {"r2", "mse", "rmse", "mae"}
    # 거리와 강한 선형관계를 준 합성데이터이므로 R2는 준수한 값이 나와야 한다
    assert metrics["r2"] > 0.5


# save_pipeline/save_metrics가 joblib·JSON 파일을 실제로 생성하고, 재로딩한 값이 원본과 일치하는지 확인한다.
def test_save_pipeline_and_metrics(tmp_path):
    df = _make_clean_df()
    pipeline, metrics = train_and_evaluate(df)

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    save_pipeline(pipeline, str(model_path))
    save_metrics(metrics, str(metrics_path))

    assert model_path.exists()
    assert metrics_path.exists()

    reloaded = joblib.load(model_path)
    assert isinstance(reloaded, Pipeline)

    with open(metrics_path, encoding="utf-8") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["r2"] == metrics["r2"]
