"""
NYC Yellow Taxi 2026-05 — ML Pipeline 학습 및 평가 모듈
=========================================================
sklearn Pipeline(ColumnTransformer 전처리 + Ridge 회귀)으로 요금(fare_amount)을
예측하는 모델을 학습, 평가, joblib으로 저장합니다.
"""

import os
import json
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


NUMERIC_FEATURES = ["trip_distance", "trip_duration_min", "passenger_count", "average_speed_mph"]
CATEGORICAL_FEATURES = ["VendorID", "RatecodeID", "pickup_hour", "pickup_day_of_week", "time_period", "is_weekend"]
TARGET_COL = "fare_amount"


def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일이 경로에 존재하지 않습니다: {data_path}")
    return pd.read_parquet(data_path)


def build_pipeline() -> Pipeline:
    """전처리(ColumnTransformer) + Ridge 회귀 추정기를 하나의 Pipeline으로 구성합니다."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0)),
    ])


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    print("[Pipeline] 모델 학습을 시작합니다...")
    pipeline.fit(X_train, y_train)
    print("[Pipeline] 모델 학습 완료.")

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }

    print("평가 지표:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    return pipeline, metrics


def save_pipeline(pipeline: Pipeline, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"모델 저장 완료: {save_path}")


def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"평가 지표 저장 완료: {metrics_path}")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "yellow_tripdata_2026-05_clean.parquet")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "taxi_fare_pipeline.pkl")
    METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_metrics.json")

    print(f"데이터 파일 경로: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"데이터 로드 성공 (행: {len(df):,}, 열: {df.shape[1]})")

    trained_pipeline, eval_metrics = train_and_evaluate(df)

    save_pipeline(trained_pipeline, MODEL_PATH)
    save_metrics(eval_metrics, METRICS_PATH)

    # 재로딩 검증 — 저장한 파이프라인이 그대로 예측되는지 확인
    reloaded = joblib.load(MODEL_PATH)
    X_check = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X_check[col] = X_check[col].astype(str)
    _ = reloaded.predict(X_check.head(5))
    print("[검증] 재로딩된 파이프라인으로 예측 성공")
