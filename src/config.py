import os
from pathlib import Path

# Base Paths
BASE_DIR = Path("I:/Github/repo/DATA-S")
DATA_DIR = BASE_DIR
RAW_ARCHIVE_CSV = BASE_DIR / "archive" / "Automated_Traffic_Volume_Counts.csv"
CLEANED_TRAFFIC_CSV = BASE_DIR / "final_cleaned_traffic_data.csv"

# Processed & Models Paths
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
CHROMA_DIR = MODELS_DIR / "chroma_db"

STREET_CATALOG_PARQUET = PROCESSED_DIR / "street_catalog.parquet"
TRAIN_SAMPLE_PARQUET = PROCESSED_DIR / "traffic_train_sample.parquet"
MODEL_FILE = MODELS_DIR / "traffic_model.joblib"
ENCODERS_FILE = MODELS_DIR / "feature_encoders.joblib"
METRICS_FILE = MODELS_DIR / "model_metrics.json"

# Feature configuration
CATEGORICAL_FEATURES = ["Boro", "Direction"]
NUMERICAL_FEATURES = [
    "Yr", "M", "D", "HH", "MM", 
    "day_of_week", "is_weekend", "is_rush_hour", 
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "SegmentID"
]
TARGET_COL_VOL = "Vol"
TARGET_COL_LEVEL = "congestion_level"
TARGET_COL_RISK = "severe_risk"

# Thresholds for Congestion Levels (based on volume percentiles in NYC data)
# 0: Low/Normal (< 50)
# 1: Moderate (50 - 120)
# 2: Heavy (120 - 220)
# 3: Severe Bottleneck / Incident Prone (> 220)
CONGESTION_THRESHOLDS = [50, 120, 220]
CONGESTION_LABELS = {
    0: "Normal Flow (انسيابي)",
    1: "Moderate Traffic (كثافة متوسطة)",
    2: "Heavy Congestion (تكدس ملحوظ)",
    3: "Severe Bottleneck / High Risk (اختناق مروري حاد / منطقة خطر)"
}
