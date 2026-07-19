"""
preprocess.py -- NYC Yellow Taxi 2026-05 원본 데이터 정제 모듈

기능: raw parquet를 읽어 파생변수를 생성하고, 완전중복·구조적결측을 제거한 뒤
날짜/운행시간/거리/요금/승객수/코드값/속도 기준 이상치를 필터링한다. 정제된
parquet와 결측치·중복 처리 요약(preprocess_summary.md)을 저장한다.

구성
  preprocess_data      -- raw parquet 로드부터 정제·저장·요약(md) 기록까지 전체 파이프라인 실행
  classify_time_period -- 시각(hour)을 출근/낮/퇴근/심야 4구간으로 분류 (preprocess_data 내부 함수)

변경내역
  2026-07-19  최초 작성
  2026-07-19  통합 파이프라인용으로 재작성 (완전중복 제거, 전처리 요약 저장 기능 추가)
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

# raw parquet를 읽어 이상치 제거·파생변수 생성 후 정제된 parquet와 요약파일을 생성한다.
def preprocess_data(raw_path: str, output_path: str, summary_path: str = None) -> None:
    """
    NYC Yellow Taxi 데이터를 정제하여 지정된 경로에 parquet 포맷으로 저장합니다.
    """
    if not os.path.exists(raw_path):
        logger.error(f"원본 데이터 파일이 존재하지 않습니다: {raw_path}")
        logger.info("이미 정제된 데이터가 존재한다면 이 단계는 생략해도 좋습니다.")
        return

    logger.info(f"[로드 시작] {raw_path}")
    df = pd.read_parquet(raw_path)
    original_count = len(df)
    logger.info(f"[로드 완료] 행: {len(df):,}  열: {df.shape[1]}")

    # STEP 1. 파생 변수 생성 — trip_duration_min
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    logger.info(f"[STEP 1] trip_duration_min 파생 변수 생성 완료")

    # STEP 2. 중복 레코드 제거
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    dup_removed = before - after
    logger.info(f"[STEP 2] 완전 중복 행 제거: {dup_removed:,}건 ({dup_removed / before * 100:.4f}%)  잔여 행: {after:,}")

    # STEP 3. 구조적 결측 레코드 제거
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
    missing_removed = before - after
    logger.info(f"[STEP 3] 구조적 결측 행 제거: {missing_removed:,}건 ({missing_removed / before * 100:.2f}%)  잔여 행: {after:,}")

    # STEP 4. 날짜/시간 이상치 제거 (2026-05만 유지)
    before = len(df)
    date_mask = (
        (df["tpep_pickup_datetime"].dt.year == 2026) &
        (df["tpep_pickup_datetime"].dt.month == 5)
    )
    df = df[date_mask].copy()
    after = len(df)
    logger.info(f"[STEP 4] 날짜 범위 필터 (2026-05만 유지) 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 5. 운행 시간(trip_duration_min) 이상치 제거 (0 < duration <= 120분)
    DURATION_LOWER = 0
    DURATION_UPPER = 120
    before = len(df)
    df = df[
        (df["trip_duration_min"] > DURATION_LOWER) &
        (df["trip_duration_min"] <= DURATION_UPPER)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 5] 운행 시간 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 6. 운행 거리(trip_distance) 이상치 제거 (0 < distance <= 100마일)
    DISTANCE_UPPER = 100
    before = len(df)
    df = df[
        (df["trip_distance"] > 0) &
        (df["trip_distance"] <= DISTANCE_UPPER)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 6] 운행 거리 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 7. 요금 및 금액 이상치 제거
    before = len(df)
    df = df[
        (df["fare_amount"] > 0) &
        (df["total_amount"] > 0) &
        (df["tip_amount"] >= 0) &
        (df["tolls_amount"] >= 0) &
        (df["congestion_surcharge"] >= 0)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 7] 요금·금액 이상치 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 8. 승객 수(passenger_count) 이상치 제거
    before = len(df)
    df = df[
        (df["passenger_count"] >= 1) &
        (df["passenger_count"] <= 6)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 8] 승객 수 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 9. 비표준 코드값(RatecodeID, payment_type) 제거
    VALID_RATECODES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    VALID_PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]
    before = len(df)
    df = df[
        df["RatecodeID"].isin(VALID_RATECODES) &
        df["payment_type"].isin(VALID_PAYMENT_TYPES)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 9] 비표준 코드값 필터 제거: {before - after:,}건  잔여: {after:,}")

    # STEP 10. 파생 변수 추가 생성
    df["average_speed_mph"] = df["trip_distance"] / (df["trip_duration_min"] / 60.0)

    SPEED_LOWER = 0.5
    SPEED_UPPER = 100.0
    before = len(df)
    df = df[
        (df["average_speed_mph"] >= SPEED_LOWER) &
        (df["average_speed_mph"] <= SPEED_UPPER)
    ].copy()
    after = len(df)
    logger.info(f"[STEP 10A] 속도 기반 필터 제거: {before - after:,}건  잔여: {after:,}")

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["pickup_day_of_week"].isin([5, 6]).astype(int)

    # 시각(hour)을 출근/낮/퇴근/심야 4개 구간 라벨로 분류한다.
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
    logger.info(f"[STEP 10B] 파생 변수 생성 완료 (average_speed_mph, pickup_hour, pickup_day_of_week, is_weekend, time_period)")

    # STEP 11. store_and_fwd_flag 이진 정수형 인코딩
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    logger.info(f"[STEP 11] store_and_fwd_flag 인코딩 완료 (Y->1, N->0)")

    # STEP 12. 데이터 타입 최적화
    categorical_cols = ["VendorID", "RatecodeID", "payment_type", "PULocationID", "DOLocationID", "time_period"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    logger.info(f"[STEP 12] 범주형 타입 변환 완료: {categorical_cols}")

    final_count = len(df)

    # 최종 Parquet 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"[저장 완료] {output_path}")

    # 데이터 준비 요약 저장 (report.md 생성)
    if summary_path:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        lines = [
            "### 결측치·중복 처리 및 EDA 요약",
            "",
            f"- 원본 행 수: **{original_count:,}**",
            f"- 완전 중복 행 제거: **{dup_removed:,}건**",
            f"- 구조적 결측 행 제거 (passenger_count/RatecodeID/store_and_fwd_flag/congestion_surcharge/Airport_fee 동시 결측): **{missing_removed:,}건**",
            f"- 이상치 필터링(날짜/운행시간/거리/요금/승객수/코드값/속도) 후 최종 행 수: **{final_count:,}**",
            f"- 데이터 잔존율: **{final_count / original_count * 100:.2f}%**",
            f"- 최종 컬럼 수: **{df.shape[1]}**",
        ]
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"[요약 저장 완료] {summary_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "yellow_tripdata_2026-05.parquet")
    PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "yellow_tripdata_2026-05_clean.parquet")
    SUMMARY_PATH = os.path.join(PROJECT_ROOT, "outputs", "preprocess_summary.md")

    preprocess_data(RAW_DATA, PROCESSED_DATA, SUMMARY_PATH)
