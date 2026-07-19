### Pandas vs Polars 로딩 성능 비교

| 항목 | Pandas | Polars |
| :--- | ---: | ---: |
| 최소 로딩 시간(초, timeit x3) | 0.062 | 0.103 |
| 행 수 | 4,090,836 | 4,090,836 |
| 열 수 | 20 | 20 |
| 메모리 사용량(MB) | 580.6 | 550.1 |

**Polars가 Pandas보다 약 0.61배 빠르게 로딩되었습니다.**

#### dtype 비교

| column                | pandas_dtype   | polars_dtype                             |
|:----------------------|:---------------|:-----------------------------------------|
| VendorID              | int32          | Int32                                    |
| tpep_pickup_datetime  | datetime64[us] | Datetime(time_unit='us', time_zone=None) |
| tpep_dropoff_datetime | datetime64[us] | Datetime(time_unit='us', time_zone=None) |
| passenger_count       | float64        | Int64                                    |
| trip_distance         | float64        | Float64                                  |
| RatecodeID            | float64        | Int64                                    |
| store_and_fwd_flag    | str            | String                                   |
| PULocationID          | int32          | Int32                                    |
| DOLocationID          | int32          | Int32                                    |
| payment_type          | int64          | Int64                                    |
| fare_amount           | float64        | Float64                                  |
| extra                 | float64        | Float64                                  |
| mta_tax               | float64        | Float64                                  |
| tip_amount            | float64        | Float64                                  |
| tolls_amount          | float64        | Float64                                  |
| improvement_surcharge | float64        | Float64                                  |
| total_amount          | float64        | Float64                                  |
| congestion_surcharge  | float64        | Float64                                  |
| Airport_fee           | float64        | Float64                                  |
| cbd_congestion_fee    | float64        | Float64                                  |

#### 결측치 집계 일치 여부

- Pandas / Polars 결측치 집계 일치: **True**

Pandas 결측치 현황:

|                      |   missing_count |   missing_pct |
|:---------------------|----------------:|--------------:|
| passenger_count      |          955371 |         23.35 |
| RatecodeID           |          955371 |         23.35 |
| store_and_fwd_flag   |          955371 |         23.35 |
| congestion_surcharge |          955371 |         23.35 |
| Airport_fee          |          955371 |         23.35 |