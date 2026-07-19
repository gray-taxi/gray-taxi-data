"""
NYC Yellow Taxi 2026-05 — ML 통합 파이프라인 학습 및 평가 모듈
============================================================
이 모듈은 조원들의 서로 다른 ML 학습 로직(요금 회귀, 정체 분류, 공항요금 분류, 고액팁 분류)을
통합하여 명령어 인자(--task)에 따라 학습, 평가 및 모델 저장을 일괄 관리합니다.
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score
)

def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일이 경로에 존재하지 않습니다: {data_path}")
    return pd.read_parquet(data_path)

# ==========================================
# 1. 정문기 - 요금 예측 회귀 (fare)
# ==========================================
def train_fare(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    print("\n--- [Task: fare] 요금 예측 회귀 모델 학습 시작 ---")
    
    # 피처 정의
    numeric_features = ["trip_distance", "trip_duration_min", "passenger_count", "average_speed_mph"]
    categorical_features = ["VendorID", "RatecodeID", "pickup_hour", "pickup_day_of_week", "time_period", "is_weekend"]
    
    # 결측치 방지 및 전처리
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("num_scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ])
    
    X = df[numeric_features + categorical_features].copy()
    for col in categorical_features:
        X[col] = X[col].astype(str)
    y = df["fare_amount"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred))
    }
    
    print("평가 지표:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
    return pipeline, metrics

# ==========================================
# 2. 상윤 - 운행 정체 이진 분류 (congestion)
# ==========================================
def train_congestion(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    print("\n--- [Task: congestion] 운행 정체 이진 분류 모델 학습 시작 ---")
    
    # 피처 정의
    numeric_features = ["trip_distance", "passenger_count", "pickup_hour", "pickup_day_of_week"]
    categorical_features = ["time_period", "is_weekend", "store_and_fwd_flag"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )
    
    pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    X = df[numeric_features + categorical_features].copy()
    for col in categorical_features:
        X[col] = X[col].astype(str)
    
    # 평균 속도 8mph 미만 = 정체
    y = (df["average_speed_mph"] < 8.0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob))
    }
    
    print("평가 지표:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
    return pipeline, metrics

# ==========================================
# 3. 상윤 - 공항요금제 판별 이진 분류 (airport)
# ==========================================
def train_airport(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    print("\n--- [Task: airport] 공항요금제 판별 이진 분류 모델 학습 시작 ---")
    
    numeric_features = ["tolls_amount", "congestion_surcharge", "Airport_fee", "cbd_congestion_fee"]
    categorical_features = ["PULocationID", "DOLocationID", "VendorID"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01))
            ]), categorical_features),
        ]
    )
    
    pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])
    
    X = df[numeric_features + categorical_features].copy()
    for col in categorical_features:
        X[col] = X[col].astype(str)
        
    # JFK=2, Newark=3 정액요금제 판별
    # category 타입(실수값)을 문자열/정수로 정상 판별하도록 처리
    y = df["RatecodeID"].astype(float).isin([2.0, 3.0]).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob))
    }
    
    print("평가 지표:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
    return pipeline, metrics

# ==========================================
# 4. 본준 - 높은 팁 여부 예측 이진 분류 (tip)
# ==========================================
def train_tip(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    print("\n--- [Task: tip] 높은 팁 이진 분류 모델 학습 시작 ---")
    
    # 팁은 신용카드 결제(payment_type == 1)에만 기록됨
    # category 타입이므로 실수형태(1.0) 또는 1로 필터링
    df_credit = df[df["payment_type"].astype(float) == 1.0].copy()
    df_credit["tip_pct"] = df_credit["tip_amount"] / df_credit["fare_amount"]
    df_credit["high_tip"] = (df_credit["tip_pct"] >= 0.15).astype(int)
    
    numeric_features = ["trip_distance", "trip_duration_min", "passenger_count", "fare_amount", "pickup_hour"]
    categorical_features = ["VendorID", "RatecodeID", "time_period", "is_weekend"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        # 속도 조절을 위해 n_estimators=100 설정
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    X = df_credit[numeric_features + categorical_features].copy()
    for col in categorical_features:
        X[col] = X[col].astype(str)
    y = df_credit["high_tip"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    print("평가 지표:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
    return pipeline, metrics

def save_model(pipeline: Pipeline, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"모델 저장 완료: {save_path}")

def update_metrics(metrics_file: str, task: str, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}
    else:
        all_metrics = {}
        
    all_metrics[task] = metrics
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"평가 지표가 '{metrics_file}'에 통합 기록되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="NYC Yellow Taxi 2026-05 ML 파이프라인 통합 학습 모듈")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["fare", "congestion", "airport", "tip", "all"],
        help="학습시킬 예측 타스크 선택 (fare, congestion, airport, tip, all)"
    )
    args = parser.parse_args()
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "yellow_tripdata_2026-05_clean.parquet")
    METRICS_FILE = os.path.join(PROJECT_ROOT, "outputs", "model_metrics.json")
    
    print(f"데이터 파일 경로: {DATA_PATH}")
    try:
        df = load_data(DATA_PATH)
        print(f"데이터 로드 성공 (행: {len(df):,}, 열: {df.shape[1]})")
    except Exception as e:
        print(f"데이터 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)
        
    tasks_to_run = [args.task] if args.task != "all" else ["fare", "congestion", "airport", "tip"]
    
    for task in tasks_to_run:
        if task == "fare":
            pipeline, metrics = train_fare(df)
            save_path = os.path.join(PROJECT_ROOT, "saved_models", "taxi_fare_pipeline.pkl")
            save_model(pipeline, save_path)
            update_metrics(METRICS_FILE, "fare", metrics)
        elif task == "congestion":
            pipeline, metrics = train_congestion(df)
            save_path = os.path.join(PROJECT_ROOT, "saved_models", "taxi_congestion_pipeline.pkl")
            save_model(pipeline, save_path)
            update_metrics(METRICS_FILE, "congestion", metrics)
        elif task == "airport":
            pipeline, metrics = train_airport(df)
            save_path = os.path.join(PROJECT_ROOT, "saved_models", "taxi_airport_pipeline.pkl")
            save_model(pipeline, save_path)
            update_metrics(METRICS_FILE, "airport", metrics)
        elif task == "tip":
            pipeline, metrics = train_tip(df)
            save_path = os.path.join(PROJECT_ROOT, "saved_models", "taxi_high_tip_pipeline.pkl")
            save_model(pipeline, save_path)
            update_metrics(METRICS_FILE, "tip", metrics)

if __name__ == "__main__":
    main()
