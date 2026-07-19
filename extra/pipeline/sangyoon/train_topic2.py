"""ML Pipeline — 주제2(운행 정체 예측) 전처리 + 모델 구성·학습·평가·저장.

입력: 전처리 결과 parquet (yellow_tripdata_2026-05_clean.parquet)
타깃: 평균속도 8mph 미만 = 정체(1), 그 외 = 원활(0)
승차 시점에 알 수 있는 정보(거리·시각·요일)만으로 정체를 예측한다.
실행: python -m src.train_topic2
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEAN_PATH = "../../data/processed/yellow_tripdata_2026-05_clean.parquet"

# 수치 피처: 승차 시점에 이미 아는 값만.
NUM_COLS = ["trip_distance", "passenger_count", "pickup_hour", "pickup_day_of_week"]

# 범주 피처: 시간대 구간·주말 여부·저장전송 플래그.
CAT_COLS = ["time_period", "is_weekend", "store_and_fwd_flag"]

# ★ 누수 차단(중요): trip_duration_min, average_speed_mph는 피처에서 반드시 제외한다.
#   speed = distance / duration 이므로 이 둘을 넣으면 타깃(속도<8)을 정의상 컨닝한다.
#   운행이 끝나야 알 수 있는 값이라 '승차 시점 예측'이라는 목적에도 어긋난다.


def prepare_data(filepath: str = CLEAN_PATH):
    """데이터를 feature/target으로 나누고 train/test로 분리한다."""
    df = pd.read_parquet(filepath)

    # target: 평균속도 8mph 미만 = 정체
    y = (df["average_speed_mph"] < 8.0).astype(int)

    X = df[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str)   # 범주형으로 통일 (OneHot 안전 처리)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"정체 비율(train): {y_train.mean():.2%}")
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    """ColumnTransformer(수치=Scaler, 범주=OneHot) + LogisticRegression을 하나로 묶는다."""
    preproc = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUM_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_COLS),
    ])
    return Pipeline([
        ("prep", preproc),
        # 정체 44% vs 원활 56%로 균형에 가까워 class_weight 없이 학습한다.
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def evaluate_pipeline(pipeline: Pipeline, X_test, y_test):
    """정확도·F1 등 평가 지표를 출력한다."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)

    print(f"\n=== 평가 지표 ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"PR-AUC   : {pr:.4f}")
    print(f"(참고: 다수클래스 baseline accuracy = {max(y_test.mean(), 1-y_test.mean()):.4f})")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["원활", "정체"]))
    return acc, f1


def save_accuracy(acc: float, f1: float, filepath: str = "models/accuracy_topic2.txt") -> None:
    """평가 지표를 텍스트로 저장한다."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"- Accuracy: {acc:.4f}\n- F1-score: {f1:.4f}\n")
    print(f"평가 지표 저장: {filepath}")


def save_pipeline(pipeline: Pipeline, filepath: str = "models/pipeline_topic2.joblib") -> None:
    """학습된 Pipeline 전체를 joblib으로 직렬화한다."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"파이프라인 저장: {filepath}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("파이프라인 학습 완료!")

    acc, f1 = evaluate_pipeline(pipeline, X_test, y_test)
    save_accuracy(acc, f1)
    save_pipeline(pipeline)

    loaded = joblib.load("models/pipeline_topic2.joblib")
    print(f"재로딩 검증 예측 정확도: {accuracy_score(y_test, loaded.predict(X_test)):.4f}")