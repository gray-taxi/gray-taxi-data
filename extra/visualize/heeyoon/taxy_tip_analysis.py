#########작성자: 김희윤 ##########
#########작성일: 2026-07-17 ###########
# ==================================================
# 분석 목적
# ==================================================
# Taxi 데이터를 활용하여 시간대별 평균 요금을 분석한다
#
# 분석 목표:
# - 승객의 이동 시간대(출근, 낮, 퇴근, 심야)에 따라
#   평균 요금이 어떻게 차이가 나는지 확인
#
# 분석 방법:
# 1. time_period 컬럼을 기준으로 시간대를 구분
# 2. 각 시간대별 tip_amount 평균 계산
# 3. 막대그래프를 통해 시간대별 평균 요금 차이를 시각화


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os

# ==================================================
# 1. Seaborn 설정 + Mac 한글 폰트
# ==================================================

sns.set_theme(style="whitegrid")

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ==================================================
# 2. 데이터 불러오기
# ==================================================

DATA_PATH = "/Users/kimheeyun/skala-intro/07_16_practice/gray-taxi-data/yellow_tripdata_2026-05_clean.parquet"

df = pd.read_parquet(DATA_PATH)


print(df.head())
print(df.info())

# ==================================================
# 3. 출력 폴더 생성
# ==================================================

os.makedirs("../../outputs", exist_ok=True)

# ==================================================
# 4. 시간대 한글 변환
# ==================================================

period_labels = {
    "morning_rush": "출근",
    "daytime": "낮",
    "evening_rush": "퇴근",
    "night": "심야"
}


df["period_label"] = df["time_period"].map(period_labels)
# ==================================================
# 5. 시간대별 평균 팁 계산
# ==================================================

tip_df = (
    df.groupby("period_label", observed=True)
    ["tip_amount"]
    .mean()
    .reindex(["출근", "낮", "퇴근", "심야"])
    .reset_index()
)

tip_df.columns = ["시간대", "평균팁"]

tip_df["평균팁"] = tip_df["평균팁"].round(2)

print("\n시간대별 평균 팁")
print(tip_df)

# ==================================================
# 6. 그래프
# ==================================================

colors = [
    "#CDB4DB",
    "#A2D2FF",
    "#BDE0FE",
    "#FFC8DD"
]

plt.figure(figsize=(8,5))


ax = sns.barplot(
    data=tip_df,
    x="시간대",
    y="평균팁",
    hue="시간대",
    palette=colors,
    legend=False
)


# 값 표시
for i, value in enumerate(tip_df["평균팁"]):

    ax.text(
        i,
        value + 0.05,
        f"${value:.2f}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.title(
    "시간대별 평균 팁",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("시간대")
plt.ylabel("평균 팁($)")

ax.yaxis.set_major_formatter(
    StrMethodFormatter("${x:.1f}")
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/avg_tip_by_period.png",
    dpi=200
)

plt.show()
plt.close()