import pandas as pd

# ==================================================
# 1. 데이터 불러오기
# ==================================================

DATA_PATH = "../yellow_tripdata_2026-05_clean.parquet"

df = pd.read_parquet(DATA_PATH)


# ==================================================
# 2. 전체 데이터 기술통계 출력
# ==================================================

print("=" * 50)
print("전체 데이터 기술통계")
print("=" * 50)

print(df.describe())


# ==================================================
# 3. 주요 분석 변수 기술통계
# ==================================================

print("\n")
print("=" * 50)
print("주요 변수 기술통계")
print("=" * 50)

summary_df = df[
    [
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "tip_amount"
    ]
].describe()

print(summary_df)


# ==================================================
# 4. 상관계수 계산
# ==================================================

corr_df = df[
    [
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "tip_amount"
    ]
]

corr_matrix = corr_df.corr()

print("\n")
print("=" * 50)
print("상관계수 Matrix")
print("=" * 50)

print(corr_matrix.round(3))