import timeit

import pandas as pd
import polars as pl
import numpy as np

import logging

PARQUET_PATH = "../data/raw/yellow_tripdata_2026-05.parquet"

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s %(message)s]",
    datefmt = "%H:%M:%S"
)
logger = logging.getLogger("taxi_eda")

# 0-1. Pandas 로드 및 성능 출력
_loaded = {}

pandas_load_times = timeit.repeat(
    lambda: _loaded.__setitem__("df", pd.read_parquet(PARQUET_PATH)),
    number = 1, repeat = 3
)

df_pd = _loaded["df"]
pandas_load_sec = min(pandas_load_times)

logger.info(f"[Pandas] 로딩 시간(3회 측정, timeit): {[f'{t:.3f}' for t in pandas_load_times]}초")
logger.info(f"[Pandas] 최소 로딩 시간: {pandas_load_sec:.3f}초")
logger.info(f"[Pandas] shape: {df_pd.shape}")
logger.info(f"[Pandas] 메모리 사용량: {df_pd.memory_usage(deep = True).sum() / 1024**2:.1f} MB")

# 0-2. Polas 로드 및 성능 출력
_loaded = {}

polars_load_times = timeit.repeat(
    lambda: _loaded.__setitem__("df", pl.read_parquet(PARQUET_PATH)),
    number = 1, repeat = 3
)

df_pl = _loaded["df"]
polars_load_sec = min(polars_load_times)

logger.info(f"[Polars] 로딩 시간(3회 측정, timeit): {[f'{t:.3f}' for t in polars_load_times]}초")
logger.info(f"[Polars] 최소 로딩 시간: {polars_load_sec:.3f}초")
logger.info(f"[Polars] shape: {df_pl.shape}")
logger.info(f"[Polars] 메모리 사용량: {df_pl.estimated_size('mb'):.1f} MB")

# --- 결과 비교 ---
comparison = pd.DataFrame({
    "항목": ["최소 로딩 시간(초, timeit x3)", "행 수", "열 수"],
    "Pandas": [round(pandas_load_sec, 3), df_pd.shape[0], df_pd.shape[1]],
    "Polars": [round(polars_load_sec, 3), df_pl.shape[0], df_pl.shape[1]]
})
logger.info(f"Polars가 Pandas보다 {pandas_load_sec / polars_load_sec:.2f}배 빠르다.")

# --- dtype 비교 ---
dtype_comparison = pd.DataFrame({
    "column": df_pd.columns,
    "pandas_dtype": df_pd.dtypes.astype(str).values,
    "polars_dtype": [str(dt) for dt in df_pl.dtypes]
})
logger.info(f"dtype 비교\n{dtype_comparison.to_string(index = False)}")

# --- Pandas 결측치 현황 ---
missing_pd = df_pd.isna().sum()
missing_pd = missing_pd[missing_pd > 0].sort_values(ascending = False)
missing_pd_pct = (missing_pd / len(df_pd) * 100).round(2)
missing_summary_pd = pd.DataFrame({"missing_count": missing_pd, "missing_pct": missing_pd_pct})
logger.info(f"[Pandas] 결측치 현황\n{missing_summary_pd.to_string()}")

# --- Polars 결측치 현황 ---
null_counts = df_pl.null_count().row(0, named = True)
missing_summary_pl = (
    pd.Series(null_counts)
    .loc[lambda s: s > 0]
    .sort_values(ascending = False)
    .to_frame("missing_count")
)
missing_summary_pl["missing_pct"] = (missing_summary_pl["missing_count"] / df_pl.height * 100).round(2)
logger.info(f"[Polars] 결측치 현황\n{missing_summary_pl.to_string()}")

# --- 두 라이브러리 결측치 집계 일치 여부 확인 ---
assert missing_summary_pd["missing_count"].sort_index().equals(
    missing_summary_pl["missing_count"].sort_index()
), "Pandas와 Polars의 결측치 집계가 일치하지 않습니다."
logger.info("Pandas / Polars 결측치 집계 일치 확인 완료")
