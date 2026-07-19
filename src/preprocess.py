"""
NYC Yellow Taxi 2026-05 — 전처리 파이프라인
============================================
이 모듈은 raw 데이터를 읽어 이상치 제거, 결측치 처리, 파생변수 생성 등의 데이터 정제를 수행합니다.
"""

import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s %(message)s]",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("taxi_preprocess")

def preprocess_data(raw_path: str, output_path: str) -> None:
    """
    NYC Yellow Taxi 데이터를 정제하여 지정된 경로에 parquet 포맷으로 저장합니다.
    """
    if not os.path.exists(raw_path):
        logger.error(f"원본 데이터 파일이 존재하지 않습니다: {raw_path}")
        logger.info("이미 정제된 데이터가 존재한다면 이 단계는 생략해도 좋습니다.")
        return

    logger.info(f"[로드 시작] {raw_path}")
    df = pd.read_parquet(raw_path)
    logger.info(f"[로드 완료] 행: {len(df):,}  열: {df.shape[1]}")

    # STEP 1. 파생 변수 생성 — trip_duration_min
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    logger.info(f"[STEP 1] trip_duration_min 파생 변수 생성 완료")

    # STEP 2. 구조적 결측 레코드 제거
    structural_missing_cols = [
        "passenger_count",
        "RatecodeID",
        "store_and_fwd_flag",
        "congestion_surcharge",
        "Airport_fee",
    ]
    before = len(df)
    df = df.dropna(subset=structural_missing_cols)
    after = len(df)
    logger.info(f"[STEP 2] 구조적 결측 행 제거: {before - after:,}건 ({(before - after) / before * 100:.2f}%)  잔여 행: {after:,}")

    # STEP 3. 날짜/시간 이상치 제거 (2026-05만 유지)
    before = len(df)
    date_mask = (
        (df["tpep_pickup_datetime"].dt.year == 2026) &
        (df["tpep_pickup_datetime"].dt.month == 5)
    )
    df = df[date_mask].copy()
    after = len(df)
    logger.info(f"[STEP 3] 날짜 범위 필터 (2026-05만 유지) 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 4. 운행 시간(trip_duration_min) 이상치 제거 (0 < duration <= 120분)
    DURATION_LOWER = 0
    DURATION_UPPER = 120
    before = len(df)
    df = df[
        (df["trip_duration_min"] > DURATION_LOWER) &
        (df["trip_duration_min"] <= DURATION_UPPER)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 4] 운행 시간 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 5. 운행 거리(trip_distance) 이상치 제거 (0 < distance <= 100마일)
    DISTANCE_UPPER = 100
    before = len(df)
    df = df[
        (df["trip_distance"] > 0) &
        (df["trip_distance"] <= DISTANCE_UPPER)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 5] 운행 거리 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 6. 요금 및 금액 이상치 제거
    before = len(df)
    df = df[
        (df["fare_amount"] > 0) &
        (df["total_amount"] > 0) &
        (df["tip_amount"] >= 0) &
        (df["tolls_amount"] >= 0) &
        (df["congestion_surcharge"] >= 0)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 6] 요금·금액 이상치 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 7. 승객 수(passenger_count) 이상치 제거
    before = len(df)
    df = df[
        (df["passenger_count"] >= 1) &
        (df["passenger_count"] <= 6)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 7] 승객 수 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 8. 비표준 코드값(RatecodeID, payment_type) 제거
    VALID_RATECODES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    VALID_PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]
    before = len(df)
    df = df[
        df["RatecodeID"].isin(VALID_RATECODES) &
        df["payment_type"].isin(VALID_PAYMENT_TYPES)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 8] 비표준 코드값 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 9. 파생 변수 추가 생성
    df["average_speed_mph"] = df["trip_distance"] / (df["trip_duration_min"] / 60.0)
    
    SPEED_LOWER = 0.5
    SPEED_UPPER = 100.0
    before = len(df)
    df = df[
        (df["average_speed_mph"] >= SPEED_LOWER) &
        (df["average_speed_mph"] <= SPEED_UPPER)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 9A] 속도 기반 필터 제거: {before - after:,}건  잔여: {after:,}")

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["pickup_day_of_week"].isin([5, 6]).astype(int)

    def classify_time_period(hour):
        if 7 <= hour < 10:
            return "morning_rush"
        elif 17 <= hour < 21:
            return "evening_rush"
        elif 22 <= hour or hour < 6:
            return "night"
        else:
            return "daytime"

    df["time_period"] = df["pickup_hour"].apply(classify_time_period)
    logger.info(f"[STEP 9B] 파생 변수 생성 완료 (average_speed_mph, pickup_hour, pickup_day_of_week, is_weekend, time_period)")

    # STEP 10. store_and_fwd_flag 이진 정수형 인코딩
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    logger.info(f"[STEP 10] store_and_fwd_flag 인코딩 완료 (Y->1, N->0)")

    # STEP 11. 데이터 타입 최적화
    categorical_cols = ["VendorID", "RatecodeID", "payment_type", "PULocationID", "DOLocationID", "time_period"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    logger.info(f"[STEP 11] 범주형 타입 변환 완료: {categorical_cols}")

    # 최종 Parquet 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"[저장 완료] {output_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "yellow_tripdata_2026-05.parquet")
    PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "yellow_tripdata_2026-05_clean.parquet")
    
    preprocess_data(RAW_DATA, PROCESSED_DATA)
