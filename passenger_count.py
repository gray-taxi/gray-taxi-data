# ==================================================
# 분석 목적
# ==================================================
# NYC Yellow Taxi 데이터를 활용하여 승객 수에 따른
# 평균 팁 금액 차이를 분석한다.
#
# 분석 목표:
# - 탑승 승객 수별 평균 팁 금액 비교
# - 승객 구성에 따른 팁 지급 패턴 확인
#
# 분석 방법:
# 1. passenger_count 기준 그룹화
# 2. 그룹별 평균 tip_amount 계산
# 3. 막대그래프로 승객 수별 평균 팁 금액 시각화
#
# 결과:
# - 승객 수에 따른 팁 금액 차이 확인
# - 고객 행동 패턴 분석
# ==================================================


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



# ==================================================
# 1. 시각화 환경 설정
# ==================================================

sns.set_theme(style="whitegrid")

# Mac 한글 폰트
plt.rcParams["font.family"] = "AppleGothic"

# 음수 표시 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False



# ==================================================
# 2. 데이터 불러오기
# ==================================================

DATA_PATH = "/Users/kimheeyun/skala-intro/07_16_practice/gray-taxi-data/yellow_tripdata_2026-05_clean.parquet"

df = pd.read_parquet(DATA_PATH)



# ==================================================
# 3. 비정상 승객 수 제거
# ==================================================

# 실제 택시 탑승 인원 기준 (1~6명)
df = df[
    (df["passenger_count"] >= 1) &
    (df["passenger_count"] <= 6)
]



# ==================================================
# 4. 결과 저장 폴더 생성
# ==================================================

os.makedirs("outputs", exist_ok=True)



# ==================================================
# 5. 승객 수별 평균 팁 금액 계산
# ==================================================

tip_passenger_df = (
    df.groupby("passenger_count")
    .agg(
        avg_tip=("tip_amount", "mean")
    )
    .reset_index()
)


# 소수점 둘째 자리 표시
tip_passenger_df["avg_tip"] = (
    tip_passenger_df["avg_tip"]
    .round(2)
)


print("승객 수별 평균 팁 금액")
print(tip_passenger_df)



# ==================================================
# 6. 시각화
# ==================================================

plt.figure(figsize=(8,5))


# 승객 수별 색상 지정
colors = [
    "#CDB4DB",  # 1명 - 라벤더
    "#A2D2FF",  # 2명 - 하늘색
    "#BDE0FE",  # 3명 - 연하늘
    "#FFC8DD",  # 4명 - 연핑크
    "#FFAFCC",  # 5명 - 핑크
    "#FFD6A5"   # 6명 - 베이지
]


ax = sns.barplot(
    data=tip_passenger_df,
    x="passenger_count",
    y="avg_tip",
    palette=colors
)



# 막대 위 금액 표시
for i, value in enumerate(tip_passenger_df["avg_tip"]):

    ax.text(
        i,
        value + 0.05,
        f"${value:.2f}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )



plt.title(
    "승객 수에 따른 평균 팁 금액",
    fontsize=16,
    fontweight="bold",
    pad=15
)


plt.xlabel(
    "탑승 승객 수(명)"
)


plt.ylabel(
    "평균 팁 금액($)"
)



plt.grid(
    axis="y",
    alpha=0.3
)



plt.tight_layout()



# 그래프 저장
plt.savefig(
    "outputs/tip_by_passenger_count.png",
    dpi=200,
    bbox_inches="tight"
)



plt.show()

plt.close()