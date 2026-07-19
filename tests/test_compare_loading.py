"""
test_compare_loading.py -- src/compare_loading.py 간단 단위 테스트

기능: 소규모 parquet로 compare_loading()을 실행해 오류 없이 동작하고
비교 결과 요약 마크다운이 저장되는지 확인한다.

변경내역
  2026-07-19  최초 작성
"""

import pandas as pd

from src.compare_loading import compare_loading


# pandas/polars 로딩 비교 결과가 outputs/loading_comparison.md 형식(비교표+dtype 비교)으로 저장되는지 확인한다.
def test_compare_loading_writes_summary(tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    summary_path = tmp_path / "loading_comparison.md"

    df = pd.DataFrame({
        "a": range(20),
        "b": [f"row-{i}" for i in range(20)],
    })
    df.to_parquet(parquet_path, index=False)

    compare_loading(str(parquet_path), str(summary_path))

    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "Pandas vs Polars" in content
    assert "dtype 비교" in content


# 원본 parquet 파일이 없을 때 예외를 던지지 않고 조용히 종료하며, 요약 파일도 생성하지 않는지 확인한다.
def test_compare_loading_skips_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.parquet"
    summary_path = tmp_path / "loading_comparison.md"

    compare_loading(str(missing_path), str(summary_path))

    assert not summary_path.exists()
