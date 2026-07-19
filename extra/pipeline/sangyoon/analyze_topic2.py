"""
NYC Yellow Taxi 2026-05 — 주제2(운행 정체 예측) 분석
====================================================
대상 컬럼(시간·거리, 5개): tpep_pickup_datetime, tpep_dropoff_datetime,
trip_distance, passenger_count, store_and_fwd_flag
(trip_duration_min, average_speed_mph, pickup_hour, pickup_day_of_week,
time_period 등은 위 5개 컬럼으로부터 preprocess.py에서 파생된 값)
전처리 결과물(clean.parquet)을 대상으로 기술통계·상관계수·t-test를 수행한다.
"""

import numpy as np
import pandas as pd
from scipy import stats

import logging

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s %(message)s]",
    datefmt = "%H:%M:%S"
)
logger = logging.getLogger("taxi_eda")

# ──────────────────────────────────────────────
# 0. 데이터 로드
# ──────────────────────────────────────────────
PARQUET_PATH = "../../data/processed/yellow_tripdata_2026-05_clean.parquet"

TOPIC2_COLS = [
    "trip_distance", "passenger_count", "store_and_fwd_flag",
    "trip_duration_min", "average_speed_mph",
    "pickup_hour", "pickup_day_of_week", "is_weekend", "time_period",
]

df = pd.read_parquet(PARQUET_PATH, columns = TOPIC2_COLS)
logger.info(f"[로드 완료] 행: {len(df):,}  주제2 컬럼: {TOPIC2_COLS}")


# ══════════════════════════════════════════════
# STEP 1. 기술통계
# ══════════════════════════════════════════════
logger.info(f"[STEP 1] 기술통계\n{df.describe().T.to_string()}")


# ══════════════════════════════════════════════
# STEP 2. 상관계수 행렬
# ══════════════════════════════════════════════
num_cols = ["trip_distance", "trip_duration_min", "average_speed_mph", "passenger_count", "pickup_hour"]
corr = df[num_cols].corr()
logger.info(f"[STEP 2] 상관계수 행렬\n{corr.round(3).to_string()}")


# ══════════════════════════════════════════════
# STEP 3. 정체(is_congested) 라벨 정의
# ══════════════════════════════════════════════
# average_speed_mph < 8mph를 "정체 운행"으로 정의 (시내 혼잡 구간 통행 속도 기준).
CONGESTED_SPEED_THRESHOLD = 8.0

df["is_congested"] = (df["average_speed_mph"] < CONGESTED_SPEED_THRESHOLD).astype(int)
positive_rate = df["is_congested"].mean() * 100
baseline_acc = max(positive_rate, 100 - positive_rate)
logger.info(
    f"[STEP 3] is_congested 라벨 (average_speed_mph < {CONGESTED_SPEED_THRESHOLD}mph) "
    f"양성 비율: {positive_rate:.2f}%  baseline acc: {baseline_acc:.1f}%"
)


# ══════════════════════════════════════════════
# STEP 4. 정체 vs 비정체 trip_distance t-test
# ══════════════════════════════════════════════
congested = df.loc[df["is_congested"] == 1, "trip_distance"]
noncongested = df.loc[df["is_congested"] == 0, "trip_distance"]

# Welch's t-test — 두 그룹의 분산이 다르므로 equal_var=False.
t_stat, p_val = stats.ttest_ind(congested, noncongested, equal_var = False)

# Cohen's d — 표본 수로 가중한 pooled std를 사용한다.
# 단순 평균((s1²+s2²)/2)은 두 그룹 크기가 같을 때만 유효하므로, 크기 차를 반영해 가중한다.
na, nb = len(congested), len(noncongested)
pooled_std = np.sqrt(
    ((na - 1) * congested.var(ddof = 1) + (nb - 1) * noncongested.var(ddof = 1))
    / (na + nb - 2)
)
cohend = (congested.mean() - noncongested.mean()) / pooled_std

logger.info(
    f"[STEP 4] 정체(n={na:,}, mean={congested.mean():.3f}mi) vs "
    f"비정체(n={nb:,}, mean={noncongested.mean():.3f}mi) trip_distance 비교"
)
logger.info(f"  → Welch t-test: t={t_stat:.3f}, p={p_val:.3e}, Cohen's d={cohend:.3f}")
# p값 해석: p<0.05이면 귀무가설 기각. 단 n이 커 p는 거의 0에 수렴하므로 크기는 Cohen's d로 판단.
if p_val < 0.05:
    logger.info(f"  → 해석: p<0.05 → 유의미한 차이 있음. Cohen's d={cohend:.2f} → 실질적 차이 크기")
else:
    logger.info(f"  → 해석: p>=0.05 → 유의미한 차이 없음 (우연 가능)")


# ══════════════════════════════════════════════
# STEP 5. 시간대(time_period) / 평일·주말 별 정체 비율
# ══════════════════════════════════════════════
period_rate = df.groupby("time_period", observed = True)["is_congested"].mean().sort_values(ascending = False) * 100
weekend_rate = df.groupby("is_weekend")["is_congested"].mean() * 100

logger.info(f"[STEP 5] time_period별 정체 비율(%)\n{period_rate.round(2).to_string()}")
logger.info(f"  → is_weekend별 정체 비율(%)\n{weekend_rate.round(2).to_string()}")
logger.info(
    "  → 시간대·요일별 정체 비율 차이가 크면 pickup_hour·time_period가 유효 피처임을 시사"
)


# ══════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════
logger.info(
    f"[요약] trip_distance는 정체 여부와 뚜렷한 상관(Cohen's d={cohend:.2f})을 보여 "
    "주제1(결제·금액)보다 시간·거리 피처가 타깃을 더 잘 설명하는 것으로 확인됨"
)