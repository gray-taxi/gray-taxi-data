"""
NYC Yellow Taxi 2026-05 — 전처리 파이프라인
============================================
각 전처리 단계의 근거는 EDA_Report.md의 해당 섹션을 참조합니다.
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# 0. 데이터 로드
# ──────────────────────────────────────────────
PARQUET_PATH = "./data/raw/yellow_tripdata_2026-05.parquet"

df = pd.read_parquet(PARQUET_PATH)
print(f"[로드 완료] 행: {len(df):,}  열: {df.shape[1]}")


# ══════════════════════════════════════════════
# STEP 1. 파생 변수 생성 — trip_duration_min
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §7-3 파생 변수 생성 추천
#   "trip_duration_min: 분석 및 요금 예측에 핵심적인 변수"
#   이후 필터링(STEP 4)에서도 사용하므로 가장 먼저 생성한다.
df["trip_duration_min"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60.0

print(f"\n[STEP 1] trip_duration_min 파생 변수 생성 완료")


# ══════════════════════════════════════════════
# STEP 2. 구조적 결측 레코드 제거
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §1 결측치 현황, §7-1 결측치 처리
#   passenger_count, RatecodeID, store_and_fwd_flag,
#   congestion_surcharge, Airport_fee 5개 컬럼이 정확히
#   동일한 955,371건(23.35%)에서 함께 결측 → 구조적 결측(Structural Missing).
#
#   §7-1: "`payment_type=0`인 레코드와 결측이 일치한다면,
#          해당 레코드 전체를 제거(dropna)하는 것이 가장 안전합니다."
#
#   먼저 교차 검증으로 payment_type=0과 결측의 관계를 확인한다.

structural_missing_cols = [
    "passenger_count",
    "RatecodeID",
    "store_and_fwd_flag",
    "congestion_surcharge",
    "Airport_fee",
]

# 교차 검증: payment_type=0 행 중 결측이 몇 %를 차지하는지 확인
missing_mask = df[structural_missing_cols[0]].isna()
payment0_mask = df["payment_type"] == 0
overlap = (missing_mask & payment0_mask).sum()
print(f"\n[STEP 2] 구조적 결측 교차검증")
print(f"  - 결측 행 수        : {missing_mask.sum():,}")
print(f"  - payment_type=0 행 : {payment0_mask.sum():,}")
print(f"  - 두 조건 동시 충족  : {overlap:,}")
print(f"  - 결측 중 payment=0 비율: {overlap / missing_mask.sum() * 100:.2f}%")

# 결측이 payment_type=0과 완전히 일치하면 해당 레코드 전체 제거
# (두 비율이 100%에 가까울수록 단순 dropna가 안전)
before = len(df)
df = df.dropna(subset=structural_missing_cols)
after = len(df)
print(f"  → 구조적 결측 행 제거: {before - after:,}건 ({(before - after) / before * 100:.2f}%)")
print(f"  → 잔여 행: {after:,}")


# ══════════════════════════════════════════════
# STEP 3. 날짜/시간 이상치 제거
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §4 이상치 진단, §7-2 시간 데이터 정제
#   "tpep_pickup_datetime이 2026년 5월이 아닌 데이터 14건(0.0003%) 존재"
#   "필터: pickup_year == 2026 이고 pickup_month == 5"

before = len(df)
date_mask = (
    (df["tpep_pickup_datetime"].dt.year == 2026) &
    (df["tpep_pickup_datetime"].dt.month == 5)
)
df = df[date_mask].copy()
after = len(df)
print(f"\n[STEP 3] 날짜 범위 필터 (2026-05만 유지)")
print(f"  → 제거: {before - after:,}건  잔여: {after:,}")


# ══════════════════════════════════════════════
# STEP 4. 운행 시간(trip_duration_min) 이상치 제거
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §4 이상치 진단, §7-2 운행 시간 필터링
#   "Trip Duration = 0 min: 52,063건(1.27%) — 시스템 에러"
#   "Trip Duration < 0 min: 0건 (없음)"
#   "Trip Duration > 3 hours: 1,430건(0.035%) — 미터기 미종료 등 노이즈"
#   "Trip Duration > 12 hours: 806건(0.02%) — 극단적 이상치"
#   "일반 범위: 0초 초과 ~ 120분 이하를 권장"
#
#   상한을 120분(2시간)으로 설정.
#   95th percentile이 48.35분임(§3 기초통계)을 감안하면 충분히 보수적인 상한.

DURATION_LOWER = 0       # 0분 이하는 시스템 오류
DURATION_UPPER = 120     # 120분 초과는 비정상 운행으로 판단

before = len(df)
df = df[
    (df["trip_duration_min"] > DURATION_LOWER) &
    (df["trip_duration_min"] <= DURATION_UPPER)
].copy()
after = len(df)
print(f"\n[STEP 4] 운행 시간 필터 (0 < duration <= {DURATION_UPPER}분)")
print(f"  → 제거: {before - after:,}건  잔여: {after:,}")


# ══════════════════════════════════════════════
# STEP 5. 운행 거리(trip_distance) 이상치 제거
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §4 이상치 진단, §7-2 운행 거리 필터링
#   "Trip Distance = 0: 113,031건(2.76%) — 주행 없이 요금 발생 등 비정상"
#   "Trip Distance < 0: 0건 (없음)"
#   "Trip Distance > 100 miles: 136건(0.003%) — 극단적 이상치"
#   "trip_distance의 std가 483.6으로 매우 크고, max=307,491마일로
#    명백한 오염 데이터가 포함 → fare_amount와 교차 확인 권장" (§7-2)

DISTANCE_UPPER = 100     # 100마일 초과는 NYC 일반 운행 범위를 벗어남

before = len(df)
df = df[
    (df["trip_distance"] > 0) &
    (df["trip_distance"] <= DISTANCE_UPPER)
].copy()
after = len(df)
print(f"\n[STEP 5] 운행 거리 필터 (0 < distance <= {DISTANCE_UPPER}마일)")
print(f"  → 제거: {before - after:,}건  잔여: {after:,}")


# ══════════════════════════════════════════════
# STEP 6. 요금 및 금액 이상치 제거
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §4 이상치 진단, §7-2 요금 및 금액 데이터 정제
#   "Fare Amount < 0: 14,231건(0.35%) — 취소/환불 건"
#   "Fare Amount = 0: 2,951건(0.07%) — 비정상 요금"
#   "Total Amount < 0: 14,877건(0.36%)"
#   "Total Amount = 0: 668건(0.02%)"
#   "Tip Amount < 0: 44건(0.001%) — 환불 처리 의심"
#   "필터: fare_amount > 0, total_amount > 0"
#   "tip_amount, tolls_amount는 >= 0 조건 적용"
#   ※ congestion_surcharge min=-2.5 확인됨 (§3 기초통계) → 음수 제거

before = len(df)
df = df[
    (df["fare_amount"] > 0) &          # 기본 요금 양수만 유효 (§7-2)
    (df["total_amount"] > 0) &         # 총 금액 양수만 유효 (§7-2)
    (df["tip_amount"] >= 0) &          # 팁은 0 이상 (환불 음수 제거) (§4)
    (df["tolls_amount"] >= 0) &        # 통행료는 0 이상 (§7-2)
    (df["congestion_surcharge"] >= 0)  # 혼잡 통행료 음수 제거 (§3 min=-2.5)
].copy()
after = len(df)
print(f"\n[STEP 6] 요금·금액 이상치 필터")
print(f"  → 제거: {before - after:,}건  잔여: {after:,}")


# ══════════════════════════════════════════════
# STEP 7. 승객 수(passenger_count) 이상치 제거
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §4 이상치 진단
#   "Passenger Count = 0: 12,533건(0.31%) — 미터기 조작 오류 또는 미입력"
#   "Passenger Count > 6: 4건(0.0001%) — NYC 택시 최대 정원(6명) 초과"
#   ※ §2 passenger_count 정의: "운전사가 미터기에 수동 입력한 값"
#     → 오입력 가능성이 있으며 0명은 의미 없는 레코드로 처리

before = len(df)
df = df[
    (df["passenger_count"] >= 1) &   # 0명 제거 (§4: 0명 = 오입력)
    (df["passenger_count"] <= 6)     # 7명 이상 제거 (§4: NYC 최대 정원 초과)
].copy()
after = len(df)
print(f"\n[STEP 7] 승객 수 필터 (1 <= passenger_count <= 6)")
print(f"  → 제거: {before - after:,}건  잔여: {after:,}")


# ══════════════════════════════════════════════
# STEP 8. 비표준 코드값(RatecodeID, payment_type) 제거
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §5 범주형 변수 분포, §7-3 범주형 데이터 인코딩
#   "RatecodeID = 99 (비표준 코드): 140,897건(3.44%)" (§4)
#   §2 컬럼 정의: "RatecodeID 유효 범위: 1~6"
#   "Payment Type Invalid(!= 1-6): 955,371건(23.35%) — payment_type=0 포함" (§4)
#   "정의되지 않은 코드값을 가지는 행은 이상치로 취급하여 제거하거나
#    '기타(Unknown)' 범주로 통합" (§7-3)
#
#   ※ payment_type=0 행은 STEP 2에서 이미 구조적 결측으로 대부분 제거됨.
#     이 단계는 혹시 남아 있는 케이스를 명시적으로 처리하는 안전장치.

VALID_RATECODES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # §2 RatecodeID 코드 정의
VALID_PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]            # §2 payment_type 코드 정의

before = len(df)
df = df[
    df["RatecodeID"].isin(VALID_RATECODES) &     # 99.0 등 비표준 요금제 제거 (§4)
    df["payment_type"].isin(VALID_PAYMENT_TYPES)  # payment_type=0 재차 제거 (§4)
].copy()
after = len(df)
print(f"\n[STEP 8] 비표준 코드값 필터 (RatecodeID, payment_type)")
print(f"  → 제거: {before - after:,}건  잔여: {after:,}")


# ══════════════════════════════════════════════
# STEP 9. 파생 변수 추가 생성
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §7-3 파생 변수 생성 추천

# (A) 평균 속도 (mph)
# 근거: §7-3 "average_speed = trip_distance / (trip_duration_min / 60.0)
#        너무 빠르거나(시속 100마일 초과) 너무 느린 데이터 탐지에 활용"
df["average_speed_mph"] = df["trip_distance"] / (df["trip_duration_min"] / 60.0)

# (B) 속도 기반 2차 이상치 필터
# 근거: §7-3 "이상치 탐지 기준으로 활용 가능"
# 시속 0.5 mph 미만: 사실상 정지 상태, GPS 오류 의심
# 시속 100 mph 초과: NYC 시내 주행 불가 수준
SPEED_LOWER = 0.5
SPEED_UPPER = 100.0

before = len(df)
df = df[
    (df["average_speed_mph"] >= SPEED_LOWER) &
    (df["average_speed_mph"] <= SPEED_UPPER)
].copy()
after = len(df)
print(f"\n[STEP 9A] 속도 기반 2차 필터 ({SPEED_LOWER} <= speed <= {SPEED_UPPER} mph)")
print(f"  → 제거: {before - after:,}건  잔여: {after:,}")

# (C) 시간대 변수 생성
# 근거: §7-3 "pickup_hour, pickup_day_of_week 추출하여
#        요일별/시간대별 운행 패턴 분석에 반영"
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek  # 0=월요일, 6=일요일
df["is_weekend"] = df["pickup_day_of_week"].isin([5, 6]).astype(int)

# (D) 시간대 구간 분류 (러시아워 식별용)
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

print(f"\n[STEP 9B] 파생 변수 생성 완료")
print(f"  추가 컬럼: average_speed_mph, pickup_hour, pickup_day_of_week, is_weekend, time_period")


# ══════════════════════════════════════════════
# STEP 10. store_and_fwd_flag 처리
# ══════════════════════════════════════════════
# 근거: EDA_Report.md §7-1 결측치 처리 (store_and_fwd_flag)
#   "최빈값 'N'으로 단순 대체하는 것은 데이터 왜곡 위험이 있으므로 지양"
#   "구조적 결측 레코드를 제거하는 방향을 우선 고려하거나,
#    모델 피처에서 제외하는 것을 권장"
#
#   → STEP 2에서 구조적 결측 행이 이미 제거됨.
#     이 시점에서 남아 있는 store_and_fwd_flag는 모두 Y/N이어야 함.
#     잔여 결측이 있을 경우 대체 없이 제거.

remaining_flag_missing = df["store_and_fwd_flag"].isna().sum()
print(f"\n[STEP 10] store_and_fwd_flag 잔여 결측: {remaining_flag_missing:,}건")
if remaining_flag_missing > 0:
    # 구조적 결측 제거 후에도 남은 결측 → 대체 없이 제거 (§7-1 지침)
    df = df.dropna(subset=["store_and_fwd_flag"]).copy()
    print(f"  → {remaining_flag_missing:,}건 추가 제거 (대체 없이 삭제)")
else:
    print(f"  → 잔여 결측 없음, 추가 처리 불필요")

# 문자열 Y/N을 이진 정수로 인코딩 (모델 학습 편의)
# Y(오프라인 저장 후 전송)=1, N(실시간 전송)=0
df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
print(f"  → store_and_fwd_flag: Y→1, N→0 인코딩 완료")


# ══════════════════════════════════════════════
# STEP 11. 데이터 타입 최적화
# ══════════════════════════════════════════════
# 범주형 변수를 category dtype으로 변환하여 메모리 절약
categorical_cols = [
    "VendorID",          # §5 VendorID 분포: 4개 고유값
    "RatecodeID",        # §5 RatecodeID 분포: 유효 6개 코드
    "payment_type",      # §5 Payment Type 분포: 유효 5개 코드
    "PULocationID",      # §6 Top 10 Location: 265개 존 ID
    "DOLocationID",      # §6 Top 10 Location: 265개 존 ID
    "time_period",       # §7-3 파생 변수: 4개 구간
]
for col in categorical_cols:
    df[col] = df[col].astype("category")

print(f"\n[STEP 11] 범주형 타입 변환 완료: {categorical_cols}")


# ══════════════════════════════════════════════
# 최종 요약 출력
# ══════════════════════════════════════════════
original_count = 4_090_836  # EDA_Report.md §1: 총 행 수 4,090,836

print("\n" + "=" * 55)
print("전처리 완료 요약")
print("=" * 55)
print(f"원본 행 수    : {original_count:>12,}")
print(f"최종 행 수    : {len(df):>12,}")
print(f"제거 행 수    : {original_count - len(df):>12,}")
print(f"데이터 잔존율 : {len(df) / original_count * 100:>11.2f}%")
print(f"최종 컬럼 수  : {df.shape[1]:>12}")
print("=" * 55)
print("\n최종 컬럼 목록:")
print(df.dtypes.to_string())

# ──────────────────────────────────────────────
# 전처리 결과 저장
# ──────────────────────────────────────────────
OUTPUT_PATH = "./data/processed/yellow_tripdata_2026-05_clean.parquet"
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\n[저장 완료] {OUTPUT_PATH}")
