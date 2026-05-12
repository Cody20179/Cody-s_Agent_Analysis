from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from src.config import DATE_FORMAT, ENV_PATH, SENSOR_NAMES

DEFAULT_URL = "https://restapi.tangramaiot.com/frontstage-ts/v1/telemetries/value/timeseries"
MAX_DAYS_PER_REQUEST = 10

def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None

    if load_dotenv:
        load_dotenv(ENV_PATH)
        return
    if not ENV_PATH.exists():
        return
    with ENV_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def _ms_to_datetime_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000).strftime(DATE_FORMAT)

def ensure_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.strptime(value.strip(), DATE_FORMAT)

def generate_time_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    ranges = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=MAX_DAYS_PER_REQUEST), end)
        ranges.append((current, chunk_end))
        current = chunk_end
    return ranges

def load_sensor_options() -> list[tuple[str, str]]:
    load_env()
    return [(name, os.getenv(name, "").strip()) for name in SENSOR_NAMES if os.getenv(name, "").strip()]

def resolve_sensors(data_types: Sequence[str] | str) -> list[tuple[str, str]]:
    options = dict(load_sensor_options())
    if isinstance(data_types, str):
        if data_types.lower() == "all":
            return list(options.items())
        data_types = [data_types]
    selected = []
    for name in data_types:
        if name not in options:
            raise ValueError(f"unknown sensor: {name}")
        selected.append((name, options[name]))
    if not selected:
        raise ValueError("at least one sensor is required")
    return selected

def parse_records(body_text: str) -> list[dict]:
    body_text = body_text.strip()
    if not body_text:
        return []
    try:
        parsed = json.loads(body_text)
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    rows = []
    for line in body_text.splitlines():
        try:
            row = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows

def fetch_records(token: str, tangram_id: str, sensor_name: str, sensor_key: str, start: datetime, end: datetime) -> list[dict]:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("requests is required for API updates") from exc

    resp = requests.get(
        os.getenv("URL", "").strip() or DEFAULT_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={"tangramId": tangram_id, "key": sensor_key, "startTs": _to_ms(start), "endTs": _to_ms(end)},
        timeout=30,
    )
    print(f"[{sensor_name}] {start.strftime(DATE_FORMAT)} -> {end.strftime(DATE_FORMAT)} HTTP {resp.status_code}")
    resp.raise_for_status()
    normalized = []
    for row in parse_records(resp.text):
        item = dict(row)
        item["data_type"] = sensor_name
        item["data_key"] = sensor_key
        if isinstance(item.get("ts"), (int, float)):
            item["ts_datetime"] = _ms_to_datetime_str(int(item["ts"]))
        if isinstance(item.get("svc_recv_ts"), (int, float)):
            item["svc_recv_ts_datetime"] = _ms_to_datetime_str(int(item["svc_recv_ts"]))
        normalized.append(item)
    return normalized

def write_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["data_type", "data_key"]
    for row in records:
        for key in row:
            if key not in columns:
                columns.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

def dump_tangram(start_time: datetime | str, end_time: datetime | str, data_types: Sequence[str] | str = "all", output_csv: str | Path | None = None) -> Path:
    load_env()
    token = os.getenv("TANGRAM_TOKEN", "").strip()
    tangram_id = os.getenv("TANGRAM_ID", "").strip()
    if not token or not tangram_id:
        raise RuntimeError("set TANGRAM_TOKEN and TANGRAM_ID in .env before updating data")
    start = ensure_datetime(start_time)
    end = ensure_datetime(end_time)
    if end <= start:
        raise ValueError("end_time must be later than start_time")
    if output_csv is None:
        raise ValueError("output_csv is required in v2")

    records = []
    for sensor_name, sensor_key in resolve_sensors(data_types):
        for chunk_start, chunk_end in generate_time_ranges(start, end):
            records.extend(fetch_records(token, tangram_id, sensor_name, sensor_key, chunk_start, chunk_end))
            time.sleep(0.2)
    records.sort(key=lambda row: (row.get("data_type", ""), row.get("ts", 0) if isinstance(row.get("ts"), (int, float)) else 0))
    output_path = Path(output_csv)
    write_csv(records, output_path)
    return output_path

