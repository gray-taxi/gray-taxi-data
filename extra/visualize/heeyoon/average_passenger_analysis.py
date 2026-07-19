# ==================================================
# 분석 목적
# ==================================================
# NYC Yellow Taxi 데이터를 활용하여 시간대별 택시 이용량을 분석한다.
#
# 분석 목표:
# - 시간대별 운행 건수를 비교하여 택시 수요 패턴 파악
# - 승객 이용이 집중되는 피크 시간을 확인
#
# 분석 방법:
# 1. pickup_hour 컬럼 기준으로 운행 건수 집계
# 2. 시간대별 택시 이용량을 막대그래프로 시각화
#
# 결과:
# - 택시 수요가 높은 시간대 확인
# - 효율적인 배차 및 운영 전략 수립에 활용 가능
# ==================================================


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



# 시각화 설정
sns.set_theme(style="whitegrid")

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False



# 데이터 불러오기
DATA_PATH = "../data/proceed/yellow_tripdata_2026-05_clean.parquet"

df = pd.read_parquet(DATA_PATH)



# 출력 폴더 생성
os.makedirs("../../outputs", exist_ok=True)



# ==================================================
# 시간대별 택시 이용량 계산
# ==================================================

hour_df = (
    df.groupby("pickup_hour")
    .size()
    .reset_index(name="trip_count")
)


print("시간대별 택시 이용량")
print(hour_df)



# ==================================================
# 시각화
# ==================================================

plt.figure(figsize=(10,5))


ax = sns.barplot(
    data=hour_df,
    x="pickup_hour",
    y="trip_count",
    color="#A2D2FF"
)



# 막대 위 값 표시
for i, value in enumerate(hour_df["trip_count"]):

    ax.text(
        i,
        value + 10000,
        f"{value:,}",
        ha="center",
        fontsize=8
    )



plt.title(
    "시간대별 택시 이용량",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel("시간대")
plt.ylabel("운행 건수")



plt.tight_layout()
plt.savefig(
    "outputs/taxi_demand_by_hour.png",
    dpi=200,
    bbox_inches="tight"
)


plt.show()

plt.close()