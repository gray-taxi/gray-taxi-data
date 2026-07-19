"""
NYC Yellow Taxi 2026-05 — Pandas vs Polars 로딩 성능 비교
==========================================================
raw parquet 파일을 pandas와 polars 양쪽으로 로딩하여 속도·메모리·dtype·결측치
집계 결과를 비교하고, 그 결과를 outputs/loading_comparison.md로 저장합니다.
"""

import os
import timeit
import logging

import pandas as pd
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s %(message)s]",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("compare_loading")


def compare_loading(parquet_path: str, summary_path: str = None) -> None:
    if not os.path.exists(parquet_path):
        logger.error(f"원본 데이터 파일이 존재하지 않습니다: {parquet_path}")
        return

    # --- Pandas 로드 및 성능 측정 ---
    _loaded = {}
    pandas_load_times = timeit.repeat(
        lambda: _loaded.__setitem__("df", pd.read_parquet(parquet_path)),
        number=1, repeat=3
    )
    df_pd = _loaded["df"]
    pandas_load_sec = min(pandas_load_times)

    logger.info(f"[Pandas] 로딩 시간(3회 측정, timeit): {[f'{t:.3f}' for t in pandas_load_times]}초")
    logger.info(f"[Pandas] 최소 로딩 시간: {pandas_load_sec:.3f}초, shape: {df_pd.shape}")

    # --- Polars 로드 및 성능 측정 ---
    _loaded = {}
    polars_load_times = timeit.repeat(
        lambda: _loaded.__setitem__("df", pl.read_parquet(parquet_path)),
        number=1, repeat=3
    )
    df_pl = _loaded["df"]
    polars_load_sec = min(polars_load_times)

    logger.info(f"[Polars] 로딩 시간(3회 측정, timeit): {[f'{t:.3f}' for t in polars_load_times]}초")
    logger.info(f"[Polars] 최소 로딩 시간: {polars_load_sec:.3f}초, shape: {df_pl.shape}")

    speed_ratio = pandas_load_sec / polars_load_sec
    logger.info(f"Polars가 Pandas보다 {speed_ratio:.2f}배 빠르다.")

    # --- dtype 비교 ---
    dtype_comparison = pd.DataFrame({
        "column": df_pd.columns,
        "pandas_dtype": df_pd.dtypes.astype(str).values,
        "polars_dtype": [str(dt) for dt in df_pl.dtypes],
    })

    # --- 결측치 비교 ---
    missing_pd = df_pd.isna().sum()
    missing_pd = missing_pd[missing_pd > 0].sort_values(ascending=False)
    missing_pd_pct = (missing_pd / len(df_pd) * 100).round(2)
    missing_summary_pd = pd.DataFrame({"missing_count": missing_pd, "missing_pct": missing_pd_pct})

    null_counts = df_pl.null_count().row(0, named=True)
    missing_summary_pl = (
        pd.Series(null_counts)
        .loc[lambda s: s > 0]
        .sort_values(ascending=False)
        .to_frame("missing_count")
    )
    missing_summary_pl["missing_pct"] = (missing_summary_pl["missing_count"] / df_pl.height * 100).round(2)

    match = missing_summary_pd["missing_count"].sort_index().equals(
        missing_summary_pl["missing_count"].sort_index()
    )
    logger.info(f"Pandas / Polars 결측치 집계 일치 여부: {match}")

    if summary_path:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        lines = [
            "### Pandas vs Polars 로딩 성능 비교",
            "",
            "| 항목 | Pandas | Polars |",
            "| :--- | ---: | ---: |",
            f"| 최소 로딩 시간(초, timeit x3) | {pandas_load_sec:.3f} | {polars_load_sec:.3f} |",
            f"| 행 수 | {df_pd.shape[0]:,} | {df_pl.shape[0]:,} |",
            f"| 열 수 | {df_pd.shape[1]} | {df_pl.shape[1]} |",
            f"| 메모리 사용량(MB) | {df_pd.memory_usage(deep=True).sum() / 1024**2:.1f} | {df_pl.estimated_size('mb'):.1f} |",
            "",
            f"**Polars가 Pandas보다 약 {speed_ratio:.2f}배 빠르게 로딩되었습니다.**",
            "",
            "#### dtype 비교",
            "",
            dtype_comparison.to_markdown(index=False),
            "",
            "#### 결측치 집계 일치 여부",
            "",
            f"- Pandas / Polars 결측치 집계 일치: **{match}**",
            "",
            "Pandas 결측치 현황:",
            "",
            missing_summary_pd.to_markdown() if len(missing_summary_pd) else "*(결측치 없음)*",
        ]
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"[요약 저장 완료] {summary_path}")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "yellow_tripdata_2026-05.parquet")
    SUMMARY_PATH = os.path.join(PROJECT_ROOT, "outputs", "loading_comparison.md")

    compare_loading(RAW_DATA, SUMMARY_PATH)
