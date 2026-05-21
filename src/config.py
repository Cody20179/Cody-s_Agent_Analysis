from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

STATE_DIR = OUTPUTS_DIR / "state"
FORECAST_DIR = OUTPUTS_DIR / "forecast"
ANOMALY_DIR = OUTPUTS_DIR / "anomaly"

STATE_MODELS_DIR = MODELS_DIR / "state"
FORECAST_MODELS_DIR = MODELS_DIR / "forecast"
ANOMALY_MODELS_DIR = MODELS_DIR / "anomaly"

ENV_PATH = ROOT / ".env"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_COL = "svc_recv_ts_datetime"
VALUE_COL = "v1"

SENSOR_NAMES = [
    "Current_A",
    "Current_B",
    "Current_C",
    "Votage_ab",
    "Votage_bc",
    "Votage_ca",
    "Instantaneous_Total_Power",
    "Electricity_consumption",
    "PF",
    "CT_Ratio",
    "kVAh",
    "Modbus_404",
    "Modbus_Timeout",
]

def ensure_dirs() -> None:
    for path in [
        DATA_RAW,
        DATA_PROCESSED,
        MODELS_DIR,
        OUTPUTS_DIR,
        STATE_DIR,
        FORECAST_DIR,
        ANOMALY_DIR,
        STATE_MODELS_DIR,
        FORECAST_MODELS_DIR,
        ANOMALY_MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
