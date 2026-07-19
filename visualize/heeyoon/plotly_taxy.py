import pandas as pd
import plotly.express as px
import os


# ==================================================
# 1. 데이터 불러오기
# ==================================================

DATA_PATH = "../yellow_tripdata_2026-05_clean.parquet"

df = pd.read_parquet(DATA_PATH)


# ====================================s==============
# 2. 시간대별 택시 이용량 계산
# ==================================================

hour_df = (
    df.groupby("pickup_hour")
    .size()
    .reset_index(name="trip_count")
)


print("시간대별 택시 이용량")
print(hour_df)


# ==================================================
# 3. Plotly 인터랙티브 차트
# ==================================================

fig = px.bar(
    hour_df,
    x="pickup_hour",
    y="trip_count",
    title="시간대별 택시 이용량 (Interactive)",
    labels={
        "pickup_hour": "시간대",
        "trip_count": "운행 건수"
    },
    text="trip_count"
)


# 막대 위 숫자 표시
fig.update_traces(
    texttemplate="%{text:,}",
    textposition="outside"
)


# 레이아웃 설정
fig.update_layout(
    width=900,
    height=500,
    xaxis_title="시간대",
    yaxis_title="운행 건수"
)


# 출력
fig.show()


# ==================================================
# 4. HTML 저장
# ==================================================

os.makedirs("../outputs", exist_ok=True)

fig.write_html(
    "outputs/taxi_demand_by_hour_interactive.html"
)