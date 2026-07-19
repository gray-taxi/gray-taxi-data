"""ML Pipeline — 전처리 + 모델을 하나의 객체로 구성·학습·평가·저장.

입력: 전처리 결과 parquet (yellow_tripdata_2026-05_clean.parquet)
주제: 공항 정액요금제(JFK=2, Newark=3) 판별 이진 분류
강의 방식(ColumnTransformer)으로 수치·범주 컬럼을 분리 처리한다.
실행: python -m src.train
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

# 수치 피처: 부가요금 4종. 공항 운행은 통행료·Airport_fee 패턴이 뚜렷하다.
NUM_COLS = ["tolls_amount", "congestion_surcharge", "Airport_fee", "cbd_congestion_fee"]

# 범주 피처: 승·하차 지역과 벤더. 지역만으로는 요금제가 결정되지 않으므로
# (JFK 출발이어도 상당수는 일반요금제) 누수가 아니라 정상적인 강한 신호다.
CAT_COLS = ["PULocationID", "DOLocationID", "VendorID"]


def prepare_data(filepath: str = CLEAN_PATH):
    """데이터를 feature/target으로 나누고 train/test로 분리한다."""
    df = pd.read_parquet(filepath)

    # target: 공항 정액요금제(2,3) -> 1, 그 외 -> 0
    # RatecodeID는 전처리에서 category(실수값)로 변환됨 -> 정수 리스트로 비교.
    y = df["RatecodeID"].isin([2, 3]).astype(int)

    # 누수 차단: RatecodeID(타깃 원본), fare_amount·trip_distance 등
    # 요금제로 결정되는 사후 값은 피처에서 제외한다.
    X = df[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        # category dtype이라도 OneHotEncoder가 안전하게 처리하도록 문자열로 통일.
        X[c] = X[c].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"공항요금제 비율(train): {y_train.mean():.2%}")
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
            # 지역 카디널리티가 크므로 드문 범주는 묶고, 미지의 범주도 무시한다.
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=0.01)),
        ]), CAT_COLS),
    ])
    return Pipeline([
        ("prep", preproc),
        # 3% 소수 클래스라 class_weight로 불균형을 보정한다.
        ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight="balanced")),
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
    # 불균형(양성 약 3.5%)이라 accuracy는 전부 음성만 찍어도 0.96이 나온다.
    # 따라서 F1·PR-AUC가 실질 성능 지표다.
    print(f"(참고: 다수클래스 baseline accuracy = {max(y_test.mean(), 1-y_test.mean()):.4f})")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["일반", "공항"]))
    return acc, f1


def save_accuracy(acc: float, f1: float, filepath: str = "models/accuracy.txt") -> None:
    """평가 지표를 텍스트로 저장한다 (report.md에서 읽음)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"- Accuracy: {acc:.4f}\n- F1-score: {f1:.4f}\n")
    print(f"평가 지표 저장: {filepath}")


def save_pipeline(pipeline: Pipeline, filepath: str = "models/pipeline.joblib") -> None:
    """학습된 Pipeline 전체를 joblib으로 직렬화한다 (전처리+모델 함께)."""
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

    # 재로딩 검증 — 저장한 파이프라인이 그대로 예측되는지 확인
    loaded = joblib.load("models/pipeline.joblib")
    print(f"재로딩 검증 예측 정확도: {accuracy_score(y_test, loaded.predict(X_test)):.4f}")