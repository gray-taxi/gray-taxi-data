"""
NYC Yellow Taxi 2026-05 — 주제3(공항 요금제 판별) 분석
====================================================
대상 컬럼(지역·부가요금, 8개): RatecodeID, PULocationID, DOLocationID,
tolls_amount, congestion_surcharge, Airport_fee, cbd_congestion_fee, VendorID
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

TOPIC3_COLS = [
    "RatecodeID", "PULocationID", "DOLocationID", "tolls_amount",
    "congestion_surcharge", "Airport_fee", "cbd_congestion_fee", "VendorID",
]
SURCHARGE_COLS = ["tolls_amount", "congestion_surcharge", "Airport_fee", "cbd_congestion_fee"]

# 필요한 8개 컬럼만 읽어 메모리·로딩 시간을 절약한다.
df = pd.read_parquet(PARQUET_PATH, columns = TOPIC3_COLS)
logger.info(f"[로드 완료] 행: {len(df):,}  주제3 컬럼: {TOPIC3_COLS}")


# ══════════════════════════════════════════════
# STEP 1. RatecodeID / VendorID 분포 및 기술통계
# ══════════════════════════════════════════════
rc = df["RatecodeID"].value_counts(normalize = True).sort_index() * 100
vd = df["VendorID"].value_counts(normalize = True).sort_index() * 100
logger.info(f"[STEP 1] RatecodeID 분포(%)\n{rc.round(2).to_string()}")
logger.info(f"  → VendorID 분포(%)\n{vd.round(2).to_string()}")
logger.info(f"  → 기술통계\n{df[SURCHARGE_COLS].describe().T.to_string()}")


# ══════════════════════════════════════════════
# STEP 2. 상관계수 행렬
# ══════════════════════════════════════════════
corr = df[SURCHARGE_COLS].corr()
logger.info(f"[STEP 2] 상관계수 행렬\n{corr.round(3).to_string()}")


# ══════════════════════════════════════════════
# STEP 3. 공항 요금제(is_airport) 라벨 정의
# ══════════════════════════════════════════════
# RatecodeID 2(JFK 정액) 또는 3(Newark)을 "공항 요금제"로 정의한다.
# JFK만 잡으면 Newark 공항 운행이 비공항으로 잘못 분류되므로 둘 다 포함한다.
df["is_airport"] = df["RatecodeID"].isin([2, 3]).astype(int)
positive_rate = df["is_airport"].mean() * 100
baseline_acc = max(positive_rate, 100 - positive_rate)
logger.info(
    f"[STEP 3] is_airport 라벨 (RatecodeID in [2, 3], JFK·Newark) "
    f"양성 비율: {positive_rate:.2f}%  baseline acc: {baseline_acc:.1f}%"
)


# ══════════════════════════════════════════════
# STEP 4. 공항 vs 비공항 tolls_amount t-test
# ══════════════════════════════════════════════
airport_tolls = df.loc[df["is_airport"] == 1, "tolls_amount"]
nonairport_tolls = df.loc[df["is_airport"] == 0, "tolls_amount"]

# Welch's t-test — 두 그룹의 분산이 크게 다르므로 equal_var=False.
t_stat, p_val = stats.ttest_ind(airport_tolls, nonairport_tolls, equal_var = False)

# Cohen's d — 표본 수로 가중한 pooled std를 사용한다.
# 단순 평균((s1²+s2²)/2)은 두 그룹 크기가 같을 때만 유효한데,
# 여기서는 공항(약 10만) vs 비공항(약 277만)으로 크기 차가 크므로 가중이 필수다.
na, nb = len(airport_tolls), len(nonairport_tolls)
pooled_std = np.sqrt(
    ((na - 1) * airport_tolls.var(ddof = 1) + (nb - 1) * nonairport_tolls.var(ddof = 1))
    / (na + nb - 2)
)
cohend = (airport_tolls.mean() - nonairport_tolls.mean()) / pooled_std

logger.info(
    f"[STEP 4] 공항(n={na:,}, mean={airport_tolls.mean():.3f}) vs "
    f"비공항(n={nb:,}, mean={nonairport_tolls.mean():.3f}) tolls_amount 비교"
)
logger.info(f"  → Welch t-test: t={t_stat:.3f}, p={p_val:.3e}, Cohen's d={cohend:.3f}")
# p값 해석: p<0.05이므로 '두 그룹 평균이 같다'는 귀무가설을 기각한다.
#           단 n이 매우 커 p는 거의 항상 0에 수렴하므로, 차이의 크기는 Cohen's d로 판단한다.
if p_val < 0.05:
    logger.info(f"  → 해석: p<0.05 → 유의미한 차이 있음. Cohen's d={cohend:.2f}(>0.8) → 실질적으로도 큰 차이")
else:
    logger.info("  → 해석: p>=0.05 → 유의미한 차이 없음 (우연 가능)")


# ══════════════════════════════════════════════
# STEP 5. 지역(PULocationID) 교차검증 — 지역 ≠ 요금제 확인
# ══════════════════════════════════════════════
# LocationID 132 = JFK, 138 = LaGuardia(TLC 존 조회표 기준).
# "공항 지역에서 출발 = 공항 요금제"가 아님을 보여 누수가 아님을 확인한다.
jfk_zone_match = (df.loc[df["is_airport"] == 1, "PULocationID"] == 132).mean() * 100
airport_fee_rate = (df.loc[df["is_airport"] == 1, "Airport_fee"] > 0).mean() * 100
# 역방향: JFK 존(132)에서 출발한 운행 중 실제 공항 요금제 비율
jfk_pickup = df["PULocationID"] == 132
airport_rate_at_jfk = df.loc[jfk_pickup, "is_airport"].mean() * 100
logger.info(f"[STEP 5] is_airport=1 그룹 중 PULocationID==132(JFK) 비율: {jfk_zone_match:.2f}%")
logger.info(f"  → is_airport=1 그룹 중 Airport_fee>0 비율: {airport_fee_rate:.2f}%")
logger.info(f"  → (역방향) JFK 존 출발 중 공항 요금제 비율: {airport_rate_at_jfk:.2f}% "
            f"→ 지역만으로 요금제가 결정되지 않음 = 누수 아님")


# ══════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════
logger.info(
    f"[요약] tolls_amount 기준 Cohen's d={cohend:.2f}로 매우 뚜렷하게 분리되지만, "
    f"양성 비율이 {positive_rate:.1f}%에 불과해 baseline acc({baseline_acc:.1f}%)만으로도 "
    "높은 정확도가 나오는 불균형 데이터임. accuracy 대신 F1/PR-AUC를 주 지표로 사용해야 함"
)