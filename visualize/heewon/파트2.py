"""
[Day 2] 통계분석 파트
- 기술통계 출력
- 상관계수 출력
- t-test 결과 및 p-value 해석
데이터: 전처리된 뉴욕 택시 parquet 파일 기준
"""

import pandas as pd
from scipy import stats

# ── 0. 데이터 불러오기 ─────────────────────────────

df = pd.read_parquet("/home/chwn926/종합실습2/파트2/yellow_tripdata_2026-05_clean.parquet")

print("데이터 shape:", df.shape)
print(df.head())


# ── 1. 기술통계 출력 ───────────────────────────────
print("\n" + "=" * 60)
print("1. 기술통계 (Descriptive Statistics)")
print("=" * 60)

# 수치형 컬럼만 대상으로 기술통계 (평균, 표준편차, 사분위수 등)
desc_stats = df.describe().T
desc_stats["skew"] = df.select_dtypes(include="number").skew()
desc_stats["kurtosis"] = df.select_dtypes(include="number").kurt()
print(desc_stats)

# 결측치 확인 (전처리 검증용으로 같이 출력)
print("\n결측치 개수:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# ── 2. 상관계수 출력 ───────────────────────────────
print("\n" + "=" * 60)
print("2. 상관계수 (Correlation Matrix)")
print("=" * 60)

numeric_df = df.select_dtypes(include="number")
corr_matrix = numeric_df.corr(method="pearson")
print(corr_matrix.round(3))

# total_amount와 상관관계가 높은 변수 상위 5개 (자기 자신 제외)
print("\ntotal_amount와 상관관계 높은 변수 Top 5:")
top_corr = corr_matrix["total_amount"].drop("total_amount").abs().sort_values(ascending=False).head(5)
print(top_corr)


# ── 3. t-test 및 p-value 해석 ──────────────────────
print("\n" + "=" * 60)
print("3. t-test (독립표본 t-검정)")
print("=" * 60)

def run_ttest(group_col, value_col, group_a, group_b, alpha=0.05):
    """
    group_col 기준으로 두 그룹(group_a vs group_b)의 value_col 평균 차이를 검정
    """
    a = df[df[group_col] == group_a][value_col].dropna()
    b = df[df[group_col] == group_b][value_col].dropna()

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test

    print(f"\n[{value_col}] {group_col} = {group_a} vs {group_b}")
    print(f"  {group_a} 평균: {a.mean():.3f} (n={len(a)})")
    print(f"  {group_b} 평균: {b.mean():.3f} (n={len(b)})")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value    : {p_value:.4f}")

    if p_value < alpha:
        print(f"  → p-value({p_value:.4f}) < 유의수준({alpha}) → 귀무가설 기각")
        print(f"  → 두 그룹의 '{value_col}' 평균은 통계적으로 유의미한 차이가 있음")
    else:
        print(f"  → p-value({p_value:.4f}) >= 유의수준({alpha}) → 귀무가설 채택")
        print(f"  → 두 그룹의 '{value_col}' 평균은 통계적으로 유의미한 차이가 없음")

    return t_stat, p_value


# 예시 1: 주말 vs 평일 요금(total_amount) 비교
run_ttest(group_col="is_weekend", value_col="total_amount", group_a=0, group_b=1)

# 예시 2: 주말 vs 평일 평균 속도(average_speed_mph) 비교
run_ttest(group_col="is_weekend", value_col="average_speed_mph", group_a=0, group_b=1)


# ── 4. 결과 요약 저장 (선택) ───────────────────────
# report.md에 넣을 요약 텍스트를 별도 파일로 저장하고 싶다면 아래처럼 활용 가능
summary_lines = [
    "## 통계분석 결과 요약",
    "",
    "### 기술통계",
    desc_stats.to_markdown(),
    "",
    "### 상관계수",
    corr_matrix.round(3).to_markdown(),
]

with open("stat_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\n요약 결과가 stat_summary.md에 저장되었습니다.")