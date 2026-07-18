"""
NYC Yellow Taxi 2026-05 — ML 파이프라인 학습 모듈
================================================
이 모듈은 전처리(StandardScaler, OneHotEncoder)와 회귀 모델(Ridge)을
하나의 sklearn Pipeline으로 통합하여 모델 학습, 평가 및 저장을 수행합니다.
"""

import os
import json
from typing import Dict, Tuple
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_clean_data(data_path: str) -> pd.DataFrame:
    """
    전처리가 완료된 Parquet 데이터를 안전하게 로드합니다.
    (일부 환경의 pandas-pyarrow 연동 버그 우회를 위해 pyarrow 직접 변환 적용)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일이 경로에 존재하지 않습니다: {data_path}")
    table = pq.read_table(data_path)
    return table.to_pandas()


def build_ml_pipeline() -> Pipeline:
    """
    전처리 레이어와 모델 추정기가 결합된 sklearn Pipeline 객체를 구성합니다.
    - 수치형 피처: StandardScaler 적용
    - 범주형 피처: OneHotEncoder 적용 (알 수 없는 범주는 무시)
    """
    # 전처리에 사용할 피처 설정
    numeric_features = [
        "trip_distance",
        "trip_duration_min",
        "passenger_count",
        "average_speed_mph",
    ]
    categorical_features = [
        "VendorID",
        "RatecodeID",
        "pickup_hour",
        "pickup_day_of_week",
        "time_period",
        "is_weekend",
    ]

    # 전처리기 구성
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # ColumnTransformer 통합
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 파이프라인 결합 (전처리 + Ridge 회귀 모델)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    return pipeline


def train_and_evaluate(
    data_path: str,
    target_col: str = "fare_amount",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    파이프라인을 학습시키고 평가하여 모델 객체와 성능 메트릭을 반환합니다.
    """
    # 데이터 로드
    df = load_clean_data(data_path)

    # 피처 및 타겟 분리
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 학습 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 파이프라인 생성 및 학습
    pipeline = build_ml_pipeline()
    print("[Pipeline] 모델 학습을 시작합니다...")
    pipeline.fit(X_train, y_train)
    print("[Pipeline] 모델 학습 완료.")

    # 예측 및 평가
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }

    return pipeline, metrics


def save_pipeline_model(model: Pipeline, save_path: str) -> None:
    """
    학습된 파이프라인 모델을 지정된 경로에 joblib을 사용해 저장합니다.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[Pipeline] 모델 저장 성공: {save_path}")


if __name__ == "__main__":
    # 스크립트 단독 실행 시 학습 수행
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE = os.path.join(PROJECT_ROOT, "data", "yellow_tripdata_2026-05_clean.parquet")
    MODEL_FILE = os.path.join(PROJECT_ROOT, "saved_models", "taxi_fare_pipeline.pkl")
    METRICS_FILE = os.path.join(PROJECT_ROOT, "outputs", "model_metrics.json")

    print(f"데이터 파일 경로: {DATA_FILE}")
    
    # 모델 학습 및 평가
    trained_model, eval_metrics = train_and_evaluate(DATA_FILE)

    print("\n" + "=" * 45)
    print("       ML 파이프라인 모델 평가 결과")
    print("=" * 45)
    for metric_name, val in eval_metrics.items():
        print(f"  {metric_name.upper():<5} : {val:.4f}")
    print("=" * 45)

    # 모델 및 평가 메트릭 저장
    save_pipeline_model(trained_model, MODEL_FILE)
    
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=4)
    print(f"[Pipeline] 성능 지표 저장 성공: {METRICS_FILE}")
