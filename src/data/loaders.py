from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW, TIME_COL, VALUE_COL

def load_sensor(sensor: str, raw_dir: Path = DATA_RAW, start=None, end=None, freq: str | None = None) -> pd.Series:
    csv_path = raw_dir / f"{sensor}.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    if "key" in df:
        df = df[df["key"].notna()].copy()
    df["ts"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df["v"] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    df = df.dropna(subset=["ts", "v"]).sort_values("ts")
    if start:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end:
        df = df[df["ts"] <= pd.Timestamp(end)]
    series = df.set_index("ts")["v"].rename(sensor)
    return series.resample(freq).mean() if freq else series

def load_consumption(raw_dir: Path = DATA_RAW, resample_freq: str = "1min", clip_outliers: bool = True) -> pd.DataFrame:
    y = load_sensor("Electricity_consumption", raw_dir)
    df = y.groupby(level=0).mean().sort_index().to_frame("y")
    if resample_freq:
        df = df.resample(resample_freq).mean().ffill()
    df = df.dropna().reset_index().rename(columns={"ts": "ds"})
    df["dy"] = df["y"].diff().clip(lower=0)
    df = df.dropna().reset_index(drop=True)
    if clip_outliers and len(df):
        q1, q3 = df["dy"].quantile([0.25, 0.75])
        upper = q3 + 3 * (q3 - q1)
        df = df[df["dy"] <= upper].reset_index(drop=True)
    return df

def split_time(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()

def evaluate(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    err = actual - predicted
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }

