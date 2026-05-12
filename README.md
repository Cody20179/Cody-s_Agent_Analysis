# Cody Agent Analysis

## Purpose

Deployable CNC analysis workspace for:

- Tangram telemetry update
- CNC machine state clustering
- Electricity forecast training and future prediction
- Anomaly detection training and interval checks
- MCP stdio tool exposure for agent integration

## Scope

This repository is copied from:

```text
/Users/cody20179/Desktop/Code/碩論/CNC_Api_Data/CNC_Analysis_v2
```

It is organized as a portable runtime project. It includes code, raw CSV data,
processed data, trained models, and generated outputs. Local secrets are not
included.

## Structure

```text
Cody-s_Agent_Analysis/
├── main.py                 # Python function API
├── mcp_server.py           # MCP stdio server
├── pyproject.toml          # uv dependency contract
├── .env.example            # Tangram update config template
├── data/
│   ├── raw/                # source sensor CSVs
│   └── processed/          # derived features and labels
├── models/
│   ├── anomaly/            # trained anomaly models
│   └── forecast/           # trained forecast models
├── outputs/
│   ├── anomaly/
│   ├── forecast/
│   └── state/
└── src/
    ├── data/
    ├── state/
    ├── forecast/
    └── anomaly/
```

## Contract

Public functions in `main.py` return a dict with:

| Field | Meaning |
| --- | --- |
| `status` | `ok` or `error` |
| `summary` | operation summary |
| `output_folder` | primary output folder |
| `key_files` | important output paths |
| `metrics` | model metrics, data status, or check result |

Main callable API:

| Function | Purpose |
| --- | --- |
| `data_status()` | inspect raw sensor CSV coverage |
| `update_data(sensors="all")` | fetch new Tangram telemetry into `data/raw/` |
| `run_state_analysis(start=None, end=None)` | run GMM state clustering |
| `train_forecast(target="dy", models=None, months_back=None)` | train forecast models |
| `forecast_future(days=None, model_names=None)` | use trained models for future prediction |
| `train_anomaly_detection()` | train anomaly models |
| `check_anomaly(start, end, min_models=2)` | check a time interval for anomalies |
| `run_all()` | run the main pipeline sequence |

## Runtime

Install dependencies:

```bash
cd /Users/cody20179/Desktop/Code/Git_My_Project/Cody-s_Agent_Analysis
uv sync
```

Quick health check:

```bash
uv run python main.py
```

Run selected functions:

```bash
uv run python -c "import json; from main import data_status; print(json.dumps(data_status(), ensure_ascii=False, indent=2))"
uv run python -c "import json; from main import forecast_future; print(json.dumps(forecast_future(days=[3,7], model_names=['Prophet']), ensure_ascii=False, indent=2))"
uv run python -c "import json; from main import check_anomaly; print(json.dumps(check_anomaly('2026-05-01 00:00:00','2026-05-02 00:00:00'), ensure_ascii=False, indent=2))"
```

## MCP

Start the MCP stdio server:

```bash
uv run python mcp_server.py
```

Tool names exposed by `mcp_server.py`:

- `tool_update_data`
- `tool_data_status`
- `tool_run_state_analysis`
- `tool_train_forecast`
- `tool_forecast_future`
- `tool_train_anomaly_detection`
- `tool_check_anomaly`
- `tool_run_all`

## Tangram Update Config

`update_data()` needs `.env`. Create it from the template:

```bash
cp .env.example .env
```

Fill:

- `TANGRAM_TOKEN`
- `TANGRAM_ID`
- sensor key mappings such as `Current_A`, `Electricity_consumption`, `PF`

The current repository already contains `data/raw/`, so read-only analysis,
forecasting, and anomaly checks can run without `.env`.

## Tracked Artifacts

The following deployable artifacts are intentionally tracked:

- `data/raw/*.csv`
- `data/processed/*.csv`
- `models/anomaly/*`
- `models/forecast/*`
- `outputs/**/*`

Ignored local-only artifacts:

- `.env`
- `.cache/`
- `__pycache__/`
- `data/_tmp_update/`

## Acceptance

Before deployment handoff:

```bash
uv run python -m compileall main.py mcp_server.py src
uv run python main.py
```

For model artifact verification:

```bash
uv run python -c "import json; from main import forecast_future; print(json.dumps(forecast_future(days=[1], model_names=['Prophet']), ensure_ascii=False, indent=2))"
```
