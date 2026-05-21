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
| `artifacts` | image artifacts derived from `key_files` |

Main callable API:

| Function | Purpose |
| --- | --- |
| `data_status()` | inspect raw sensor CSV coverage |
| `update_data(sensors="all")` | fetch new Tangram telemetry into `data/raw/` |
| `run_state_analysis(start=None, end=None)` | train and apply GMM state clustering |
| `train_state_model(start=None, end=None)` | train the GMM state model only |
| `apply_state_model(start=None, end=None)` | apply an existing GMM state model |
| `train_forecast(target="dy", models=None, months_back=None)` | train forecast models |
| `forecast_future(days=None, model_names=None)` | use trained models for future prediction |
| `train_direct_tree_forecast(target="dy", days=None, models=None, months_back=None)` | train direct tree forecast models for selected horizons |
| `train_anomaly_detection()` | train anomaly models |
| `check_anomaly(start, end, min_models=2)` | check a time interval for anomalies |
| `run_all()` | run the main pipeline sequence |

Image outputs are returned structurally. The model does not need to write image
paths in natural language. For example:

```json
{
  "artifacts": [
    {
      "type": "image",
      "label": "plot",
      "path": "/absolute/path/to/outputs/forecast/future_forecast.png"
    }
  ]
}
```

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

## Docker Compose

Build and run the MCP stdio service container:

```bash
docker compose up --build
```

The compose service uses:

| File | Purpose |
| --- | --- |
| `Dockerfile` | Builds a Python 3.12 runtime with `uv sync --frozen --no-dev` |
| `compose.yaml` | Runs `uv run python mcp_server.py` |
| `.dockerignore` | Excludes local secrets, git metadata, caches, and virtualenvs |

Runtime volumes:

| Host path | Container path |
| --- | --- |
| `./data` | `/app/data` |
| `./models` | `/app/models` |
| `./outputs` | `/app/outputs` |

These runtime directories are local deployment state and are ignored by git.

`compose.yaml` does not reference `.env` directly, so `docker compose config`
will not print local secrets. If Tangram update credentials are needed in a
deployment environment, inject them through that environment's secret manager or
compose override file. `.env` is ignored and must not be committed to the public
repository.

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
- `tool_train_state_model`
- `tool_apply_state_model`
- `tool_train_forecast`
- `tool_forecast_future`
- `tool_train_direct_tree_forecast`
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

## Local Artifacts

Ignored local-only artifacts:

- `.env`
- `.cache/`
- `__pycache__/`
- `data/raw/`
- `data/processed/`
- `data/_tmp_update/`
- `models/`
- `outputs/`

The public repository does not include local datasets, trained model weights,
generated outputs, or thesis chart notes.

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
