# CNC Analysis v2 — Agent Skill Definition

Use this skill when the user asks about CNC power analysis, machine state, forecast, anomaly detection, data freshness, energy budget, OEE, or any question related to CNC machine monitoring and optimization.

---

## Project Root

`/Users/cody20179/Desktop/Code/碼論/CNC_Api_Data/CNC_Analysis_v2`

---

## Runtime

- Python: `/Users/cody20179/Desktop/Code/py312/bin/python`
- Always `cd` to project root before running commands.
- Call `main.py` public functions; do not import internal pipeline modules directly.

---

## Public API (main.py)

All functions return a dict with keys: `status`, `summary`, `output_folder`, `key_files`, `metrics`.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `data_status()` | — | Check raw sensor data coverage and freshness |
| `update_data(sensors="all")` | `sensors`: `"all"` or list of sensor names | Pull latest data from Tangram API |
| `run_state_analysis(start=None, end=None)` | `start`/`end`: ISO datetime strings | GMM clustering → state labels |
| `train_forecast(target="dy", models=None, months_back=None)` | `target`: `"dy"` or `"y"`; `models`: list; `months_back`: int | Train forecast models |
| `forecast_future(days=None, model_names=None)` | `days`: list of int; `model_names`: list | Predict future consumption |
| `train_anomaly_detection()` | — | Train 3 anomaly detection models |
| `check_anomaly(start, end, min_models=2)` | `start`/`end`: ISO datetime; `min_models`: int | Check a time range for anomalies |
| `run_all()` | — | Execute the full pipeline |

---

## MCP Tools (mirror main.py)

`tool_update_data`, `tool_data_status`, `tool_run_state_analysis`, `tool_train_forecast`, `tool_forecast_future`, `tool_train_anomaly_detection`, `tool_check_anomaly`, `tool_run_all`

---

## Pipeline Dependencies

```
data_status → update_data → run_state_analysis → train_forecast → forecast_future
                                            → train_anomaly_detection → check_anomaly
```

- **State analysis must run before** anomaly training (anomaly features depend on `data/processed/state_labeled.csv`).
- **Forecast training must run before** future forecast (depends on `models/forecast/training_config.json` and model files).

---

## Key File Paths

### Data

| Path | Description |
|------|-------------|
| `data/raw/` | Raw sensor CSV files (one per sensor) |
| `data/processed/state_labeled.csv` | State-labeled dataset |
| `data/processed/detection_features.csv` | Anomaly detection features |

### Models

| Path | Description |
|------|-------------|
| `models/forecast/` | Forecast model files (`.json`, `.pkl`, `.pt`) |
| `models/forecast/training_config.json` | Forecast training config (required for future forecast) |
| `models/anomaly/scaler.pkl` | Feature scaler |
| `models/anomaly/thresholds.json` | Anomaly thresholds |
| `models/anomaly/isolation_forest.pkl` | IsolationForest model |
| `models/anomaly/one_class_svm.pkl` | OneClassSVM model |
| `models/anomaly/autoencoder.pt` | Autoencoder model |

### Outputs

| Path | Description |
|------|-------------|
| `outputs/state/state_labeled.csv` | State labels (copy) |
| `outputs/state/state_report.txt` | State analysis report |
| `outputs/state/bic_curve.png` | BIC curve chart |
| `outputs/state/state_timeline.png` | State timeline chart |
| `outputs/forecast/forecast_metrics.json` | Forecast model metrics |
| `outputs/forecast/forecast_test_overlay.png` | Test set prediction overlay |
| `outputs/forecast/future_forecast_report.json` | Future forecast summary |
| `outputs/forecast/future_forecast.png` | Future forecast chart |
| `outputs/forecast/future_<Model>.csv` | Per-model future predictions |
| `outputs/anomaly/anomaly_metrics.json` | Anomaly model metrics |
| `outputs/anomaly/score_distribution.png` | Score distribution chart |
| `outputs/anomaly/last_anomaly_check.csv` | Last anomaly check detail |

---

## Sensors

Known sensor names for `update_data`:

- `Current_A`, `Current_B`, `Current_C`
- `Votage_ab`, `Votage_bc`, `Votage_ca`
- `Instantaneous_Total_Power`, `Electricity_consumption`
- `PF`, `CT_Ratio`, `kVAh`
- `Modbus_404`, `Modbus_Timeout`

---

## State Analysis Details

- Implementation: `src/state/gmm.py`
- Resample: `5min`
- Features: `I_mean`, `I_imbalance`, `Power`, `PF_abs`, `kVAh_rate`
- GMM tests K=2..4, best K by BIC
- State names: `Off`, `Idle`, `Running_Low`, `Running_High`, `Running_Peak`

---

## Forecast Details

- Implementation: `src/forecast/pipeline.py`
- Targets: `dy` (per-minute increment), `y` (cumulative)
- Models: `BaselineLastWeek`, `Prophet`, `XGBoost`, `LightGBM`
- Default models for quick validation: `["BaselineLastWeek", "Prophet"]`
- Full model list for thesis comparison: `["BaselineLastWeek", "Prophet", "XGBoost", "LightGBM"]`

---

## Anomaly Detection Details

- Implementation: `src/anomaly/pipeline.py`
- Training states: `Running_Low`, `Running_High`
- Models: `IsolationForest`, `OneClassSVM`, `Autoencoder`
- Voting rule: `min_models=2` (at least 2 models agree → anomalous)

---

## Intent → Action Mapping

When the user's message matches these patterns, take the corresponding action.

### Data Freshness

| User says | Action |
|-----------|--------|
| 「資料最新到什麼時候？」 / 「資料狀態」 / 「data status」 | `tool_data_status` |
| 「幫我更新資料」 / 「拉最新資料」 / 「update data」 | `tool_update_data(sensors="all")` |
| 「只更新電流感測器」 / 「update Current_A only」 | `tool_update_data(sensors="Current_A,Current_B,Current_C")` |

### State Analysis

| User says | Action |
|-----------|--------|
| 「機台狀態分析」 / 「跑狀態」 / 「state analysis」 | `tool_run_state_analysis` |
| 「機台稼動率多少？」 | `tool_run_state_analysis` → read `outputs/state/state_report.txt` → compute Running% |
| 「待機時間佔多少？」 | `tool_run_state_analysis` → read `outputs/state/state_report.txt` → report Idle% |
| 「只分析最近一個月」 | `tool_run_state_analysis(start="2026-04-11")` |

### Forecast

| User says | Action |
|-----------|--------|
| 「幫我預測兩週後的能耗」 / 「預測14天」 | `tool_forecast_future(days="14")` |
| 「預測下個月的用電量」 / 「預測30天」 | `tool_forecast_future(days="30")` |
| 「預測明天和後天」 | `tool_forecast_future(days="1,2")` |
| 「用 XGBoost 預測」 | `tool_train_forecast(models="XGBoost")` → `tool_forecast_future(model_names="XGBoost")` |
| 「用全部模型預測比較」 | `tool_train_forecast(models="BaselineLastWeek,Prophet,XGBoost,LightGBM")` → `tool_forecast_future(model_names="BaselineLastWeek,Prophet,XGBoost,LightGBM")` |
| 「快速驗證預測」 | `tool_train_forecast(models="BaselineLastWeek,Prophet")` → `tool_forecast_future` |
| 「哪個模型最準？」 | Read `outputs/forecast/forecast_metrics.json` → compare R² / MAE |

### Anomaly

| User says | Action |
|-----------|--------|
| 「昨天加工有沒有異常？」 | Compute yesterday's range → `tool_check_anomaly(start, end)` |
| 「上週五夜班正常嗎？」 | Compute that shift's range → `tool_check_anomaly(start, end)` |
| 「最近3天有異常嗎？」 | `tool_check_anomaly(3_days_ago, now)` |
| 「放寬異常判定」 / 「1個模型就判定」 | `tool_check_anomaly(start, end, min_models=1)` |
| 「重新訓練異常模型」 | `tool_train_anomaly_detection` |
| 「異常偵測誤報率高嗎？」 | Read `outputs/anomaly/anomaly_metrics.json` → report Precision |

### Full Pipeline

| User says | Action |
|-----------|--------|
| 「跑完整流程」 / 「run all」 | `tool_run_all` |
| 「幫我從頭跑一次」 | Execute step-by-step: `data_status` → `update_data` → `run_state_analysis` → `train_forecast` → `forecast_future` → `train_anomaly_detection` → `check_anomaly` |

### Energy & Cost Scenarios

| User says | Action |
|-----------|--------|
| 「下週電費大概多少？」 | `tool_forecast_future(days="7")` → multiply predicted kWh by electricity rate |
| 「待機能省多少電？」 | Read state report → Idle% × avg power × hours × rate |
| 「排程最佳化建議」 | Read state report → suggest reducing Idle by shifting schedules |
| 「需量反應」 / 「DR」 | `tool_forecast_future(days="7")` → identify peak periods → suggest load reduction |

### Model Comparison & Research

| User says | Action |
|-----------|--------|
| 「模型比較」 / 「4個模型比較」 | `tool_train_forecast(models="BaselineLastWeek,Prophet,XGBoost,LightGBM")` → compare metrics |
| 「不同 months_back 影響？」 | Run `train_forecast(months_back=1)` vs `months_back=3` vs `months_back=6` → compare |
| 「特徵工程實驗」 | Explain: need to modify `src/state/gmm.py` or `src/anomaly/pipeline.py` feature config |

---

## Response Guidelines

1. **Always reply in Traditional Chinese** unless the user explicitly asks for another language.
2. **Check dependencies first**: if the user asks for `forecast_future` but no forecast model exists, run `train_forecast` first. If the user asks for `check_anomaly` but no anomaly model exists, run `train_anomaly_detection` first (which requires `run_state_analysis` first).
3. **Mention key output files**: after each action, tell the user which files were generated (from `key_files`).
4. **For time-based queries** (e.g., "yesterday", "last week"), compute the actual datetime strings before calling tools.
5. **For cost estimates**, use Taiwan industrial electricity rate (~NT$2.5–5/kWh depending on tier) as a reference unless the user specifies otherwise.
6. **Quick validation vs full run**: default to quick validation (`BaselineLastWeek + Prophet`) unless the user says "全部模型" or "完整比較".

---

## Quick Reference Commands

```bash
cd /Users/cody20179/Desktop/Code/碼論/CNC_Api_Data/CNC_Analysis_v2

# Data status
/Users/cody20179/Desktop/Code/py312/bin/python -c "import json; from main import data_status; print(json.dumps(data_status(), ensure_ascii=False, indent=2))"

# Update all sensors
/Users/cody20179/Desktop/Code/py312/bin/python -c "import json; from main import update_data; print(json.dumps(update_data('all'), ensure_ascii=False, indent=2))"

# State analysis
/Users/cody20179/Desktop/Code/py312/bin/python -c "import json; from main import run_state_analysis; print(json.dumps(run_state_analysis(), ensure_ascii=False, indent=2))"

# Quick forecast
/Users/cody20179/Desktop/Code/py312/bin/python -c "import json; from main import train_forecast; print(json.dumps(train_forecast(target='dy', models=['BaselineLastWeek','Prophet'], months_back=1), ensure_ascii=False, indent=2))"

# Future forecast
/Users/cody20179/Desktop/Code/py312/bin/python -c "import json; from main import forecast_future; print(json.dumps(forecast_future(days=[3,7], model_names=['Prophet']), ensure_ascii=False, indent=2))"

# Anomaly training
/Users/cody20179/Desktop/Code/py312/bin/python -c "import json; from main import train_anomaly_detection; print(json.dumps(train_anomaly_detection(), ensure_ascii=False, indent=2))"

# Anomaly check
/Users/cody20179/Desktop/Code/py312/bin/python -c "import json; from main import check_anomaly; print(json.dumps(check_anomaly('2026-05-01 00:00:00','2026-05-02 00:00:00'), ensure_ascii=False, indent=2))"
```

---

## Environment

- Env file: `.env` (contains `TANGRAM_TOKEN`, `TANGRAM_ID`, sensor name keys)
- API module: `src/data/tangram_api.py`
- Update module: `src/data/update.py`
- New raw files start from `2025-09-15 15:00:00`
- `dump_tangram(...)` chunks by `MAX_DAYS_PER_REQUEST = 10`
- Deduplicate by `svc_recv_ts_datetime`

---

*Skill version: 2026-05-11*