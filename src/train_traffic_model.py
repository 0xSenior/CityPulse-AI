import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    TRAIN_SAMPLE_PARQUET,
    MODEL_FILE,
    ENCODERS_FILE,
    METRICS_FILE,
    MODELS_DIR,
    CONGESTION_LABELS,
)

FEATURE_STATS_FILE = MODELS_DIR / "segment_feature_stats.joblib"

def compute_and_apply_feature_stats(train_df, test_df):
    """
    Computes segment and borough historical encodings on train data only
    to avoid data leakage, then applies to both train and test.
    """
    print("[*] Computing granular Segment-level historical statistics...")
    global_mean = float(train_df["Vol"].mean())
    
    seg_stats = train_df.groupby("SegmentID")["Vol"].agg(
        seg_mean="mean", seg_std="std", seg_max="max"
    ).reset_index()
    
    seg_hour_stats = train_df.groupby(["SegmentID", "HH"])["Vol"].agg(
        seg_hour_mean="mean"
    ).reset_index()
    
    seg_day_stats = train_df.groupby(["SegmentID", "day_of_week"])["Vol"].agg(
        seg_day_mean="mean"
    ).reset_index()
    
    boro_hour_stats = train_df.groupby(["Boro", "HH"])["Vol"].agg(
        boro_hour_mean="mean"
    ).reset_index()
    
    stats_dict = {
        "global_mean": global_mean,
        "seg_stats": seg_stats.set_index("SegmentID").to_dict(orient="index"),
        "seg_hour_stats": seg_hour_stats.set_index(["SegmentID", "HH"])["seg_hour_mean"].to_dict(),
        "seg_day_stats": seg_day_stats.set_index(["SegmentID", "day_of_week"])["seg_day_mean"].to_dict(),
        "boro_hour_stats": boro_hour_stats.set_index(["Boro", "HH"])["boro_hour_mean"].to_dict(),
    }
    
    joblib.dump(stats_dict, FEATURE_STATS_FILE)
    print(f"[+] Saved fast feature lookup stats to: {FEATURE_STATS_FILE}")
    
    def transform_df(d):
        d = d.merge(seg_stats, on="SegmentID", how="left")
        d = d.merge(seg_hour_stats, on=["SegmentID", "HH"], how="left")
        d = d.merge(seg_day_stats, on=["SegmentID", "day_of_week"], how="left")
        d = d.merge(boro_hour_stats, on=["Boro", "HH"], how="left")
        
        d["seg_mean"] = d["seg_mean"].fillna(d["boro_hour_mean"]).fillna(global_mean).astype("float32")
        d["seg_hour_mean"] = d["seg_hour_mean"].fillna(d["seg_mean"]).astype("float32")
        d["seg_day_mean"] = d["seg_day_mean"].fillna(d["seg_mean"]).astype("float32")
        d["seg_std"] = d["seg_std"].fillna(20.0).astype("float32")
        d["seg_max"] = d["seg_max"].fillna(d["seg_mean"] * 2.0).astype("float32")
        d["boro_hour_mean"] = d["boro_hour_mean"].fillna(global_mean).astype("float32")
        return d
        
    return transform_df(train_df), transform_df(test_df), stats_dict

def train_models():
    print(f"[*] Loading training dataset from: {TRAIN_SAMPLE_PARQUET}")
    start_time = time.time()
    df = pd.read_parquet(TRAIN_SAMPLE_PARQUET)
    print(f"[+] Loaded {len(df):,} records in {time.time() - start_time:.2f}s")
    
    # 80/20 Train Test Split
    print("[*] Splitting dataset (80% train, 20% test)...")
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    # Compute and merge segment feature stats
    train_df, test_df, stats_dict = compute_and_apply_feature_stats(train_df, test_df)
    
    feature_cols = [
        "Yr", "M", "D", "HH", "MM", "day_of_week", "is_weekend", "is_rush_hour",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "seg_mean", "seg_hour_mean", "seg_day_mean", "seg_std", "seg_max", "boro_hour_mean"
    ]
    
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    y_train_multi = train_df["congestion_level"].astype(int)
    y_test_multi = test_df["congestion_level"].astype(int)
    
    # Binary Severe Congestion / Incident Bottleneck Target (Level >= 2)
    y_train_risk = (train_df["congestion_level"] >= 2).astype(int)
    y_test_risk = (test_df["congestion_level"] >= 2).astype(int)
    
    y_train_reg = train_df["Vol"].astype(float)
    y_test_reg = test_df["Vol"].astype(float)
    
    # 1. Train High Risk Binary Classifier (Acc ~90%)
    print("[*] Training High-Accuracy Incident & Severe Bottleneck Classifier...")
    t0 = time.time()
    clf_risk = XGBClassifier(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    clf_risk.fit(X_train, y_train_risk)
    preds_risk = clf_risk.predict(X_test)
    risk_acc = accuracy_score(y_test_risk, preds_risk)
    risk_f1 = f1_score(y_test_risk, preds_risk)
    print(f"[+] Severe Congestion / Incident Accuracy: {risk_acc * 100:.2f}% | F1-Score: {risk_f1 * 100:.2f}% (in {time.time() - t0:.2f}s)")
    
    # 2. Train Multi-Class Classifier
    print("[*] Training Multi-Level Flow Classifier (4 classes)...")
    t1 = time.time()
    clf_multi = XGBClassifier(
        n_estimators=140,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    clf_multi.fit(X_train, y_train_multi)
    preds_multi = clf_multi.predict(X_test)
    multi_acc = accuracy_score(y_test_multi, preds_multi)
    multi_f1 = f1_score(y_test_multi, preds_multi, average="weighted")
    print(f"[+] Multi-Class Accuracy: {multi_acc * 100:.2f}% | F1-Score: {multi_f1 * 100:.2f}% (in {time.time() - t1:.2f}s)")
    
    # 3. Train Volume Regressor (R2 ~0.80)
    print("[*] Training Volume Regressor (R² ~ 0.80)...")
    t2 = time.time()
    reg = XGBRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train, y_train_reg)
    preds_reg = reg.predict(X_test)
    vol_mae = mean_absolute_error(y_test_reg, preds_reg)
    vol_r2 = r2_score(y_test_reg, preds_reg)
    print(f"[+] Volume Regression MAE: {vol_mae:.2f} vehicles | R²: {vol_r2:.4f} (in {time.time() - t2:.2f}s)")
    
    # Save Model Bundle
    model_bundle = {
        "clf_risk": clf_risk,
        "clf_multi": clf_multi,
        "classifier": clf_multi, # backward compatibility
        "regressor": reg,
        "feature_cols": feature_cols,
        "labels": CONGESTION_LABELS,
    }
    
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_FILE)
    
    # Feature importances
    importances = dict(zip(feature_cols, [round(float(v), 4) for v in clf_risk.feature_importances_]))
    
    metrics = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "high_risk_accuracy": round(float(risk_acc), 4),
        "high_risk_f1": round(float(risk_f1), 4),
        "classifier_accuracy": round(float(multi_acc), 4),
        "classifier_f1_weighted": round(float(multi_f1), 4),
        "volume_mae": round(float(vol_mae), 2),
        "volume_r2": round(float(vol_r2), 4),
        "feature_importances": importances,
    }
    
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Successfully saved enhanced models to: {MODEL_FILE}")
    print(f"[+] Successfully saved upgraded metrics to: {METRICS_FILE}")
    return metrics

if __name__ == "__main__":
    train_models()
