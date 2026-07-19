"""
ML Pipeline 파트 - 신용카드 결제 운행의 "높은 팁 여부" 예측
------------------------------------------------------------
- 팁(tip_amount)은 신용카드 결제일 때만 기록되고, 현금 결제는 항상 0으로
  기록되기 때문에(EDA_Report.md 참고) payment_type == 1 (신용카드)인
  데이터만 사용한다.
- 타겟: tip_amount / fare_amount 가 15% 이상이면 1(높은 팁), 아니면 0
- sklearn Pipeline으로 전처리(스케일링/원핫인코딩) + 모델 학습을 한번에 처리
"""
 
import pandas as pd
import joblib
 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
 
# 1. 데이터 불러오기
df = pd.read_parquet("./yellow_tripdata_2026-05_clean.parquet")
print("원본 행 수:", len(df))
 
# 2. 타겟 변수 만들기
#    신용카드 결제(payment_type == 1)만 사용 (현금은 팁이 기록 안 됨)
df = df[df["payment_type"] == 1].copy()
df["tip_pct"] = df["tip_amount"] / df["fare_amount"]
df["high_tip"] = (df["tip_pct"] >= 0.15).astype(int)
 
print("분석 대상 행 수:", len(df))
print("high_tip 비율:\n", df["high_tip"].value_counts(normalize=True))
 
# 3. 피처 선택
#    tip_amount, total_amount는 타겟을 만드는 데 쓴 값이라 피처에서 제외
#    (넣으면 답을 미리 알려주는 셈이 되어 모델 성능이 뻥튀기됨)
numeric_features = [
    "trip_distance",
    "trip_duration_min",
    "passenger_count",
    "fare_amount",
    "pickup_hour",
]
categorical_features = [
    "VendorID",
    "RatecodeID",
    "time_period",
    "is_weekend",
]
 
X = df[numeric_features + categorical_features]
y = df["high_tip"]
 
# 4. train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# 5. 전처리 + 모델 Pipeline 구성
#    - 수치형: StandardScaler로 스케일 맞춰줌
#    - 범주형: OneHotEncoder로 0/1 벡터로 변환
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
 
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)
 
# 6. 학습
model.fit(X_train, y_train)
 
# 7. 평가
y_pred = model.predict(X_test)
 
print("\n===== 모델 평가 결과 =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
# 8. 모델 저장
joblib.dump(model, "high_tip_model.joblib")
print("\n모델 저장 완료: high_tip_model.joblib")
