"""
test_preprocess.py -- src/preprocess.py 간단 단위 테스트

기능: 소규모 합성 raw 데이터로 preprocess_data()를 실행해 중복·구조적결측·이상치
행이 걸러지고 파생변수가 생성되는지, 요약 파일이 저장되는지 확인한다.

변경내역
  2026-07-19  최초 작성
"""

import pandas as pd
import pyarrow.parquet as pq

from src.preprocess import preprocess_data


# 정상 행 6개(서로 다른 시각/거리) + 완전중복 1개 + 비표준코드 1개 + 음수요금 1개 +
# 구조적결측 1개, 총 10행짜리 raw 스키마 합성 DataFrame을 만든다.
def _make_raw_df() -> pd.DataFrame:
    base = {
        "VendorID": 1,
        "passenger_count": 1.0,
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": 100,
        "DOLocationID": 200,
        "payment_type": 1,
        "fare_amount": 12.0,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": 2.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "total_amount": 16.0,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
        "cbd_congestion_fee": 0.75,
    }

    rows = []
    for i in range(6):
        row = dict(base)
        row["tpep_pickup_datetime"] = pd.Timestamp("2026-05-10 08:00:00") + pd.Timedelta(minutes=i)
        row["tpep_dropoff_datetime"] = row["tpep_pickup_datetime"] + pd.Timedelta(minutes=15)
        row["trip_distance"] = 2.0 + i * 0.5
        rows.append(row)

    rows.append(dict(rows[0]))  # 완전 중복 행 (dedup 검증용)

    invalid_ratecode = dict(rows[0])
    invalid_ratecode["RatecodeID"] = 99.0  # 비표준 코드값 (제거 대상)
    rows.append(invalid_ratecode)

    negative_fare = dict(rows[0])
    negative_fare["fare_amount"] = -5.0  # 음수 요금 이상치 (제거 대상)
    rows.append(negative_fare)

    structural_missing = dict(rows[0])
    structural_missing["passenger_count"] = None
    structural_missing["RatecodeID"] = None
    structural_missing["store_and_fwd_flag"] = None
    structural_missing["congestion_surcharge"] = None
    structural_missing["Airport_fee"] = None
    rows.append(structural_missing)  # 구조적 결측 행 (제거 대상)

    return pd.DataFrame(rows)


# 완전중복·비표준코드·음수요금·구조적결측 행이 모두 제거되고, 파생변수 컬럼이 생성되는지 확인한다.
def test_preprocess_removes_duplicates_and_invalid_rows(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    output_path = tmp_path / "clean.parquet"
    summary_path = tmp_path / "summary.md"

    _make_raw_df().to_parquet(raw_path, index=False)
    preprocess_data(str(raw_path), str(output_path), str(summary_path))

    result = pq.read_table(output_path).to_pandas()

    # 정상 행 6개만 남아야 한다 (중복/비표준코드/음수요금/구조적결측 행 모두 제거)
    assert len(result) == 6
    for col in ["trip_duration_min", "average_speed_mph", "pickup_hour", "time_period", "is_weekend"]:
        assert col in result.columns


# outputs/preprocess_summary.md에 원본 행수·중복제거 건수 등 요약 문구가 기록되는지 확인한다.
def test_preprocess_writes_summary_file(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    output_path = tmp_path / "clean.parquet"
    summary_path = tmp_path / "summary.md"

    _make_raw_df().to_parquet(raw_path, index=False)
    preprocess_data(str(raw_path), str(output_path), str(summary_path))

    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "원본 행 수" in content
    assert "완전 중복 행 제거" in content


# 원본 raw 파일이 없을 때 예외를 던지지 않고 조용히 종료하며, 출력 파일도 생성하지 않는지 확인한다.
def test_preprocess_skips_when_raw_file_missing(tmp_path):
    missing_raw = tmp_path / "does_not_exist.parquet"
    output_path = tmp_path / "clean.parquet"

    preprocess_data(str(missing_raw), str(output_path))

    assert not output_path.exists()
