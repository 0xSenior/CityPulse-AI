import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import re
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import chromadb

from src.config import (
    STREET_CATALOG_PARQUET,
    TRAIN_SAMPLE_PARQUET,
    CHROMA_DIR,
    MODELS_DIR,
)

SEGMENT_STATS_PARQUET = MODELS_DIR / "segment_stats.parquet"

# Arabic to English keyword translation dictionary for NYC
ARABIC_NYC_MAP = {
    "مانهاتن": "Manhattan",
    "بروكلين": "Brooklyn",
    "كوينز": "Queens",
    "برونكس": "Bronx",
    "ستاتن آيلاند": "Staten Island",
    "فلاتبوش": "FLATBUSH",
    "برودواي": "BROADWAY",
    "واشنطن": "WASHINGTON",
    "ريتشموند": "RICHMOND",
    "هانتر": "HUNTS",
    "الخامس": "5 AVENUE",
    "أتلانتيك": "ATLANTIC",
    "فولتون": "FULTON",
}

def translate_query(query: str) -> str:
    """Translates common Arabic terms and normalizes ordinals (e.g. 94th -> 94) in queries."""
    translated = query
    # Normalize ordinals like 94th -> 94, 5th -> 5
    translated = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", translated, flags=re.IGNORECASE)
    for ar, en in ARABIC_NYC_MAP.items():
        if ar in translated:
            translated = translated.replace(ar, en)
    return translated

def compute_segment_stats():
    """Compute historical traffic baseline stats for each SegmentID."""
    print("[*] Computing historical segment baseline stats...")
    df_sample = pd.read_parquet(TRAIN_SAMPLE_PARQUET)
    
    stats = df_sample.groupby("SegmentID").agg(
        avg_vol=("Vol", "mean"),
        max_vol=("Vol", "max"),
        std_vol=("Vol", "std"),
        severe_risk_rate=("severe_risk", "mean"),
        count=("Vol", "count")
    ).reset_index()
    
    # Calculate most frequent peak hour for each segment
    peak_hours = df_sample.groupby(["SegmentID", "HH"])["Vol"].mean().reset_index()
    idx_max = peak_hours.groupby("SegmentID")["Vol"].idxmax()
    peak_df = peak_hours.loc[idx_max, ["SegmentID", "HH"]].rename(columns={"HH": "peak_hour"})
    
    stats = stats.merge(peak_df, on="SegmentID", how="left")
    stats["avg_vol"] = stats["avg_vol"].round(1)
    stats["max_vol"] = stats["max_vol"].round(1)
    stats["severe_risk_rate"] = stats["severe_risk_rate"].round(3)
    stats["peak_hour"] = stats["peak_hour"].fillna(17).astype(int)
    
    stats.to_parquet(SEGMENT_STATS_PARQUET, index=False)
    print(f"[+] Saved historical stats for {len(stats):,} segments to: {SEGMENT_STATS_PARQUET}")
    return stats

def get_chroma_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def build_vector_store(limit_segments: int = 1000):
    """
    Builds ChromaDB collection for top high-traffic streets and intersections.
    """
    print("[*] Building Vector Store for RAG...")
    df_catalog = pd.read_parquet(STREET_CATALOG_PARQUET)
    
    # Merge with stats if available
    if not SEGMENT_STATS_PARQUET.exists():
        df_stats = compute_segment_stats()
    else:
        df_stats = pd.read_parquet(SEGMENT_STATS_PARQUET)
        
    df_merged = df_catalog.merge(df_stats, on="SegmentID", how="left").fillna({
        "avg_vol": 80.0,
        "max_vol": 250.0,
        "severe_risk_rate": 0.10,
        "peak_hour": 17,
        "latitude": 40.7128,
        "longitude": -74.0060,
    })
    
    # Prioritize top segments
    df_merged = df_merged.sort_values(by="avg_vol", ascending=False).iloc[:limit_segments].reset_index(drop=True)
    
    client = get_chroma_client()
    collection_name = "traffic_knowledge_base"
    
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = []
    documents = []
    metadatas = []
    
    for idx, row in df_merged.iterrows():
        seg_id = str(row["SegmentID"])
        street = str(row["street"]).strip()
        from_st = str(row["fromSt"]).strip()
        to_st = str(row["toSt"]).strip()
        boro = str(row["Boro"]).strip()
        direction = str(row["Direction"]).strip()
        avg_v = float(row["avg_vol"])
        peak_h = int(row["peak_hour"])
        risk_r = float(row["severe_risk_rate"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        
        doc = (
            f"Street: {street}. Cross streets: from {from_st} to {to_st}. "
            f"Borough: {boro}. Direction: {direction}. Segment ID: {seg_id}. "
            f"Historical Average Volume: {avg_v} vehicles/15min. Peak hour: {peak_h}:00. "
            f"Historical Bottleneck Rate: {risk_r * 100:.1f}%."
        )
        
        ids.append(f"seg_{seg_id}_{idx}")
        documents.append(doc)
        metadatas.append({
            "segment_id": int(row["SegmentID"]),
            "street": street,
            "from_st": from_st,
            "to_st": to_st,
            "boro": boro,
            "direction": direction,
            "avg_vol": avg_v,
            "peak_hour": peak_h,
            "risk_rate": risk_r,
            "latitude": lat,
            "longitude": lon,
        })
        
    # Add in batches of 200
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"    Indexed {end}/{len(ids)} segments...")
        
    print(f"[+] Successfully built ChromaDB collection '{collection_name}' with {len(ids)} documents!")
    return collection

class TrafficRetriever:
    def __init__(self):
        self.catalog = pd.read_parquet(STREET_CATALOG_PARQUET)
        if SEGMENT_STATS_PARQUET.exists():
            self.stats = pd.read_parquet(SEGMENT_STATS_PARQUET)
        else:
            self.stats = None
            
        try:
            self.client = get_chroma_client()
            self.collection = self.client.get_or_create_collection("traffic_knowledge_base")
        except Exception:
            self.collection = None
            
    def search(self, query: str, boro: str = None, top_k: int = 3):
        """
        Hybrid search combining Arabic-to-English translation,
        exact street name keyword matching, and Chroma vector retrieval.
        """
        norm_query = translate_query(query).upper()
        
        # 1. High precision matching: First check full street name phrases, then significant tokens
        matched_records = []
        stop_words = {"TRAFFIC", "FORECAST", "FOR", "THE", "AND", "WITH", "NEAR", "FROM", "STATUS", "CONDITION", "IS", "ARE", "IN", "AT", "ON", "OF", "PM", "AM"}
        clean_tokens = [w for w in re.split(r"\W+", norm_query) if len(w) >= 2 and w not in stop_words]
        
        # Try finding multi-word consecutive matches first (e.g. '94 AVENUE', 'HUNTS POINT')
        for i in range(len(clean_tokens) - 1):
            phrase = f"{clean_tokens[i]} {clean_tokens[i+1]}"
            subset = self.catalog[self.catalog["street"].str.contains(phrase, na=False, case=False)]
            if boro:
                subset = subset[subset["Boro"].str.contains(boro, na=False, case=False)]
            if not subset.empty:
                for _, r in subset.head(top_k).iterrows():
                    matched_records.append(dict(r))
                    
        # If no phrase matched, match primary distinct tokens (avoiding generic 'AVENUE' alone if specific numbers/names exist)
        if not matched_records:
            distinct_tokens = [t for t in clean_tokens if t not in {"AVENUE", "STREET", "ROAD", "BOULEVARD", "AVE", "ST", "BLVD"}]
            search_tokens = distinct_tokens if distinct_tokens else clean_tokens
            for w in search_tokens:
                subset = self.catalog[self.catalog["street"].str.contains(rf"\b{re.escape(w)}\b", na=False, case=False, regex=True)]
                if boro:
                    subset = subset[subset["Boro"].str.contains(boro, na=False, case=False)]
                if not subset.empty:
                    for _, r in subset.head(top_k).iterrows():
                        matched_records.append(dict(r))
                    
        # 2. ChromaDB semantic search (if available)
        chroma_metas = []
        if self.collection:
            try:
                where_filter = {"boro": boro} if boro else None
                results = self.collection.query(
                    query_texts=[norm_query],
                    n_results=top_k,
                    where=where_filter
                )
                if results and results.get("metadatas"):
                    chroma_metas = results["metadatas"][0]
            except Exception:
                chroma_metas = []
                
        # Combine and deduplicate by SegmentID
        combined = []
        seen_segs = set()
        
        for r in matched_records:
            seg = r.get("SegmentID")
            if seg not in seen_segs:
                seen_segs.add(seg)
                combined.append({
                    "segment_id": int(r["SegmentID"]),
                    "street": str(r["street"]),
                    "from_st": str(r.get("fromSt", "")),
                    "to_st": str(r.get("toSt", "")),
                    "boro": str(r["Boro"]),
                    "direction": str(r.get("Direction", "NB")),
                    "latitude": float(r.get("latitude", 40.7128)),
                    "longitude": float(r.get("longitude", -74.0060)),
                    "avg_vol": 85.0,
                    "peak_hour": 17,
                    "risk_rate": 0.12,
                })
                
        for meta in chroma_metas:
            seg = meta.get("segment_id") or meta.get("SegmentID")
            if seg not in seen_segs:
                seen_segs.add(seg)
                combined.append(meta)
                
        return combined[:top_k]

if __name__ == "__main__":
    compute_segment_stats()
    build_vector_store()
