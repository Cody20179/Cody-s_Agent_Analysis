from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, SENSOR_NAMES, TIME_COL
from src.data.tangram_api import dump_tangram

DATA_EARLIEST = datetime(2025, 9, 15, 15, 0, 0)

def _sensor_list(sensors: str | list[str]) -> list[str]:
    if isinstance(sensors, str):
        return SENSOR_NAMES if sensors.lower() == "all" else [sensors]
    return list(sensors)

def _latest_ts(csv_path: Path) -> datetime | None:
    if not csv_path.exists():
        return None
    ts = pd.to_datetime(pd.read_csv(csv_path, usecols=[TIME_COL], low_memory=False)[TIME_COL], errors="coerce").dropna()
    return ts.max().to_pydatetime() if len(ts) else None

def _append_new_rows(csv_path: Path, new_df: pd.DataFrame) -> int:
    new_df = new_df.drop(columns=[c for c in ("data_type", "data_key") if c in new_df.columns], errors="ignore")
    new_df = new_df[new_df["key"].notna()].copy()
    if new_df.empty:
        return 0
    if csv_path.exists():
        old_df = pd.read_csv(csv_path, low_memory=False)
        old_valid = int(old_df["key"].notna().sum()) if "key" in old_df else len(old_df)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[TIME_COL], keep="first")
    else:
        old_valid = 0
        combined = new_df
    combined[TIME_COL] = pd.to_datetime(combined[TIME_COL], errors="coerce")
    combined = combined.sort_values(TIME_COL).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return max(0, int(combined["key"].notna().sum()) - old_valid)

def status(raw_dir: Path = DATA_RAW) -> list[dict]:
    rows = []
    for sensor in SENSOR_NAMES:
        csv_path = raw_dir / f"{sensor}.csv"
        if not csv_path.exists():
            rows.append({"sensor": sensor, "exists": False})
            continue
        df = pd.read_csv(csv_path, usecols=[TIME_COL], low_memory=False)
        ts = pd.to_datetime(df[TIME_COL], errors="coerce").dropna()
        rows.append({
            "sensor": sensor,
            "exists": True,
            "rows": int(len(df)),
            "start": str(ts.min()) if len(ts) else None,
            "end": str(ts.max()) if len(ts) else None,
        })
    return rows

def update(sensors: str | list[str] = "all", raw_dir: Path = DATA_RAW, verbose: bool = True) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = raw_dir.parent / "_tmp_update"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().replace(second=0, microsecond=0)
    results = {}
    for sensor in _sensor_list(sensors):
        csv_path = raw_dir / f"{sensor}.csv"
        latest = _latest_ts(csv_path)
        start = latest + timedelta(minutes=1) if latest else DATA_EARLIEST
        if start >= now:
            results[sensor] = {"added": 0, "latest": str(latest)}
            continue
        tmp_path = tmp_dir / f"_upd_{sensor}.csv"
        try:
            dump_tangram(start, now, sensor, tmp_path)
            added = _append_new_rows(csv_path, pd.read_csv(tmp_path, low_memory=False))
            tmp_path.unlink(missing_ok=True)
            results[sensor] = {"added": added, "latest": str(_latest_ts(csv_path))}
            if verbose:
                print(f"{sensor}: +{added} rows")
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            results[sensor] = {"added": 0, "error": str(exc)}
    return results

