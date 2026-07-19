
###시간대에 따라 평균 팁 금액 차이가 있나??## 
import pandas as pd
from scipy.stats import ttest_ind


# ==================================================
# 1. 데이터 불러오기
# ==================================================

DATA_PATH = "yellow_tripdata_2026-05_clean.parquet"

df = pd.read_parquet(DATA_PATH)



# ==================================================
# 2. 비교할 그룹 생성
# ==================================================

day_tip = df[
    df["time_period"] == "daytime"
]["tip_amount"]


night_tip = df[
    df["time_period"] == "night"
]["tip_amount"]



print("=" * 50)
print("그룹별 평균 팁 금액")
print("=" * 50)


print("낮 시간 평균 팁:", round(day_tip.mean(), 2))
print("심야 시간 평균 팁:", round(night_tip.mean(), 2))



# ==================================================
# 3. 독립표본 t-test 수행
# ==================================================

t_stat, p_value = ttest_ind(
    day_tip,
    night_tip,
    equal_var=False
)



print("\n")
print("=" * 50)
print("T-test 결과")
print("=" * 50)


print("t-statistic :", round(t_stat, 3))
print("p-value :", p_value)



# ==================================================
# 4. p-value 해석
# ==================================================

alpha = 0.05


print("\n")
print("=" * 50)
print("결과 해석")
print("=" * 50)


if p_value < alpha:
    print(
        "p-value가 0.05보다 작으므로 "
        "귀무가설을 기각한다."
    )
    print(
        "→ 낮과 심야 시간대의 평균 팁 금액에는 "
        "통계적으로 유의미한 차이가 있다."
    )

else:
    print(
        "p-value가 0.05보다 크므로 "
        "귀무가설을 기각할 수 없다."
    )
    print(
        "→ 낮과 심야 시간대의 평균 팁 금액 차이는 "
        "통계적으로 유의미하지 않다."
    )