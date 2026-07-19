import pandas as pd
import plotly.express as px
import os


# ==================================================
# 1. 데이터 불러오기
# ==================================================

DATA_PATH = "yellow_tripdata_2026-05_clean.parquet"

df = pd.read_parquet(DATA_PATH)



# ==================================================
# 2. 시간대 한글 변환
# ==================================================

period_labels = {
    "morning_rush": "출근",
    "daytime": "낮",
    "evening_rush": "퇴근",
    "night": "심야"
}


df["period_label"] = df["time_period"].map(period_labels)



# ==================================================
# 3. 시간대별 평균 팁 계산
# ==================================================

tip_df = (
    df.groupby("period_label")
    ["tip_amount"]
    .mean()
    .reindex(["출근", "낮", "퇴근", "심야"])
    .reset_index()
)


tip_df.columns = [
    "시간대",
    "평균팁"
]


tip_df["평균팁"] = tip_df["평균팁"].round(2)



print(tip_df)



# ==================================================
# 4. Plotly 인터랙티브 차트
# ==================================================

fig = px.bar(
    tip_df,
    x="시간대",
    y="평균팁",
    title="시간대별 평균 팁 금액 (Interactive)",
    labels={
        "시간대": "시간대",
        "평균팁": "평균 팁($)"
    },
    text="평균팁"
)



# 막대 위 값 표시
fig.update_traces(
    texttemplate="$%{text}",
    textposition="outside"
)



# 레이아웃 설정
fig.update_layout(
    width=900,
    height=500,
    xaxis_title="시간대",
    yaxis_title="평균 팁($)"
)



# 화면 출력
fig.show()



# ==================================================
# 5. HTML 저장
# ==================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


fig.write_html(
    "outputs/avg_tip_by_period_interactive.html"
)