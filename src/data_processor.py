import re
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer
from src.config import (
    RAW_ARCHIVE_CSV,
    CLEANED_TRAFFIC_CSV,
    STREET_CATALOG_PARQUET,
    TRAIN_SAMPLE_PARQUET,
    CONGESTION_THRESHOLDS,
)

# EPSG:2263 (NY Long Island) to EPSG:4326 (WGS84 Lat/Lon)
COORD_TRANSFORMER = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

def parse_wkt_point(wkt_str):
    """Extract (lon, lat) from WktGeom POINT string."""
    if not isinstance(wkt_str, str):
        return None, None
    m = re.search(r"POINT\s*\(\s*([\d\.\-]+)\s+([\d\.\-]+)\s*\)", wkt_str)
    if m:
        try:
            x, y = float(m.group(1)), float(m.group(2))
            lon, lat = COORD_TRANSFORMER.transform(x, y)
            return round(lat, 6), round(lon, 6)
        except Exception:
            return None, None
    return None, None

def build_street_catalog(max_rows=150000):
    """
    Extract unique SegmentIDs, street names, intersections, and coordinates
    to build the knowledge base catalog for RAG retrieval.
    """
    print(f"[*] Building Street Catalog from: {RAW_ARCHIVE_CSV}")
    cols_to_use = ["SegmentID", "street", "fromSt", "toSt", "Boro", "WktGeom", "Direction"]
    
    catalog_dfs = []
    # Read in chunks to keep memory footprint light
    chunk_size = 50000
    rows_read = 0
    
    for chunk in pd.read_csv(RAW_ARCHIVE_CSV, usecols=cols_to_use, chunksize=chunk_size):
        chunk = chunk.dropna(subset=["SegmentID", "street"])
        catalog_dfs.append(chunk)
        rows_read += len(chunk)
        if rows_read >= max_rows:
            break
            
    df_raw = pd.concat(catalog_dfs, ignore_index=True)
    df_catalog = df_raw.drop_duplicates(subset=["SegmentID"]).copy()
    
    # Parse latitude and longitude
    print("[*] Parsing geographic coordinates for street segments...")
    coords = df_catalog["WktGeom"].apply(parse_wkt_point)
    df_catalog["latitude"] = [c[0] for c in coords]
    df_catalog["longitude"] = [c[1] for c in coords]
    
    # Clean text columns
    for col in ["street", "fromSt", "toSt", "Boro", "Direction"]:
        df_catalog[col] = df_catalog[col].astype(str).str.strip()
        
    df_catalog = df_catalog.drop(columns=["WktGeom"], errors="ignore")
    
    STREET_CATALOG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_catalog.to_parquet(STREET_CATALOG_PARQUET, index=False)
    print(f"[+] Saved Street Catalog with {len(df_catalog)} unique segments to: {STREET_CATALOG_PARQUET}")
    return df_catalog

def create_training_sample(target_rows=600000, chunk_size=150000):
    """
    Stream final_cleaned_traffic_data.csv to create a balanced,
    rich training sample with temporal and cyclic features.
    """
    print(f"[*] Processing traffic dataset: {CLEANED_TRAFFIC_CSV}")
    collected_chunks = []
    total_sampled = 0
    # Sampling fraction per chunk to gather ~target_rows without blowing RAM
    sample_rate = 0.25
    
    for chunk in pd.read_csv(CLEANED_TRAFFIC_CSV, chunksize=chunk_size):
        chunk = chunk.dropna(subset=["Vol", "Yr", "M", "D", "HH", "SegmentID", "Boro"])
        chunk = chunk[chunk["Vol"] >= 0]
        
        # Sample rows
        sampled = chunk.sample(frac=sample_rate, random_state=42) if len(chunk) > 1000 else chunk
        collected_chunks.append(sampled)
        total_sampled += len(sampled)
        print(f"    -> Sampled {total_sampled:,} rows so far...")
        if total_sampled >= target_rows:
            break
            
    df_sample = pd.concat(collected_chunks, ignore_index=True).iloc[:target_rows].copy()
    
    print("[*] Engineering temporal and cyclic features...")
    # Compute day of week & weekend
    # Construct vectorized dates
    date_strs = (
        df_sample["Yr"].astype(str) + "-" + 
        df_sample["M"].astype(str).str.zfill(2) + "-" + 
        df_sample["D"].astype(str).str.zfill(2)
    )
    dts = pd.to_datetime(date_strs, errors="coerce")
    df_sample["day_of_week"] = dts.dt.dayofweek.fillna(0).astype("int8")
    df_sample["is_weekend"] = (df_sample["day_of_week"] >= 5).astype("int8")
    
    # Rush hour indicator (7-10 AM, 4-7 PM)
    df_sample["is_rush_hour"] = df_sample["HH"].isin([7, 8, 9, 16, 17, 18, 19]).astype("int8")
    
    # Cyclic encoding for Hour and Month
    df_sample["hour_sin"] = np.sin(2 * np.pi * df_sample["HH"] / 24.0).astype("float32")
    df_sample["hour_cos"] = np.cos(2 * np.pi * df_sample["HH"] / 24.0).astype("float32")
    df_sample["month_sin"] = np.sin(2 * np.pi * df_sample["M"] / 12.0).astype("float32")
    df_sample["month_cos"] = np.cos(2 * np.pi * df_sample["M"] / 12.0).astype("float32")
    
    # Congestion Level classification target
    t1, t2, t3 = CONGESTION_THRESHOLDS
    conditions = [
        df_sample["Vol"] < t1,
        (df_sample["Vol"] >= t1) & (df_sample["Vol"] < t2),
        (df_sample["Vol"] >= t2) & (df_sample["Vol"] < t3),
        df_sample["Vol"] >= t3
    ]
    choices = [0, 1, 2, 3]
    df_sample["congestion_level"] = np.select(conditions, choices, default=0).astype("int8")
    
    # Severe bottleneck / incident risk target (Level 3 or rush hour surge)
    df_sample["severe_risk"] = (df_sample["congestion_level"] == 3).astype("int8")
    
    # Clean string columns
    for col in ["Boro", "Direction"]:
        df_sample[col] = df_sample[col].astype(str).str.strip()
        
    TRAIN_SAMPLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_parquet(TRAIN_SAMPLE_PARQUET, index=False)
    print(f"[+] Saved training sample ({len(df_sample):,} rows) to: {TRAIN_SAMPLE_PARQUET}")
    print("[*] Congestion level distribution:\n", df_sample["congestion_level"].value_counts(normalize=True))
    return df_sample

if __name__ == "__main__":
    build_street_catalog()
    create_training_sample()
