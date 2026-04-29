import pandas as pd
import numpy as np

def calcular_rfm(df, customer_col, date_col, amount_col, ref_date):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df = df.dropna(subset=[date_col, amount_col, customer_col])
    
    ref_date = pd.to_datetime(ref_date)
    
    rfm = df.groupby(customer_col).agg(
        recency_days=(date_col, lambda x: (ref_date - x.max()).days),
        frequency=(date_col, "count"),
        monetary=(amount_col, "sum")
    ).reset_index()
    
    rfm.columns = ["customer", "recency_days", "frequency", "monetary"]
    rfm["recency_days"] = rfm["recency_days"].astype(int)
    rfm["frequency"] = rfm["frequency"].astype(int)
    rfm["monetary"] = rfm["monetary"].round(2)
    
    def safe_qcut(series, q, labels):
        try:
            return pd.qcut(series, q=q, labels=labels, duplicates="drop").astype(float).fillna(1.0).astype(int)
        except Exception:
            return pd.Series([1] * len(series), index=series.index)
    
    rfm["recency_score"]  = safe_qcut(rfm["recency_days"], 4, [4, 3, 2, 1])
    rfm["frequency_score"] = safe_qcut(rfm["frequency"],   4, [1, 2, 3, 4])
    rfm["monetary_score"]  = safe_qcut(rfm["monetary"],    4, [1, 2, 3, 4])
    
    rfm["rfm_score"] = (rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]).astype(int)
    
    out = rfm[["customer", "recency_days", "frequency", "monetary", "rfm_score"]]
    out = out.sort_values(["rfm_score", "customer"], ascending=[False, True]).reset_index(drop=True)
    return out
