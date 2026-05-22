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
| `list_scenarios()` | list user-facing scenarios |
| `route_scenario(text)` | map plain-language text to a scenario |
| `run_scenario(scenario_id, **kwargs)` | execute one stable scenario workflow |
| `validate_scenarios(scenario_ids=None, repeat=1, model_name="workflow-baseline")` | run repeatable scenario validation |

## Scenario Flows

User-facing flows are separated from low-level model tools. Daily flows are for
operators. Maintenance flows are for setup or retraining.

| Scenario | User intent | Workflow |
| --- | --- | --- |
| `today_status` | 查詢今日機台狀態 | apply state model over today's period |
| `period_status` | 查詢指定期間機台狀態 | apply state model over `start` / `end` |
| `last_week_runtime` | 查詢上週開機 / 關機時間 | apply state model over last calendar week |
| `utilization` | 查詢機台稼動率 | apply state model and summarize running ratio |
| `next_week_power_forecast` | 預測下週耗電 | run future forecast for 7 days |
| `next_month_cost_forecast` | 預測下個月電費 | run 30-day forecast and cost calculation |
| `period_anomaly_check` | 檢查指定期間異常 | run anomaly check over `start` / `end` |
| `initialize_project` | 初始化專案模型 | optional data update, state training, forecast training, anomaly training |
| `retrain_all_models` | 重新訓練全部模型 | state, forecast, and anomaly retraining |

Each scenario returns a fixed contract in `metrics`:

| Field | Meaning |
| --- | --- |
| `scenario_id` | stable scenario key |
| `expected_tools` | expected tool flow for the scenario |
| `tool_trace` | called tool names, order, duration, and success |
| `tool_flow_signature` | compact tool sequence |
| `required_outputs` | scenario-specific required fields |
| `output_schema_passed` | whether required fields were produced |

Scenario validation records `model_name`, repeat index, success, duration,
tool counts, tool flow signature, artifact count, and placeholder token /
thinking fields for later System-level LLM metrics.

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
uv run python -c "import json; from main import list_scenarios; print(json.dumps(list_scenarios(), ensure_ascii=False, indent=2))"
uv run python -c "import json; from main import route_scenario; print(json.dumps(route_scenario('幫我預測下週耗電多少'), ensure_ascii=False, indent=2))"
uv run python -c "import json; from main import run_scenario; print(json.dumps(run_scenario('next_week_power_forecast'), ensure_ascii=False, indent=2))"
uv run python -m src.validation.runner --scenario next_week_power_forecast --repeat 3 --model-name workflow-baseline
uv run python -m src.validation.report outputs/validation/manual/tool_workflow_pilot_20260522_184205.csv --output-dir outputs/validation/manual/report_20260522_184205
uv run python -m src.validation.system_test --api-base-url http://127.0.0.1:8000 --user-id Cody
uv run python -m src.validation.journal_report outputs/validation/system/report_e71a789e105619d6
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
- `tool_list_scenarios`
- `tool_route_scenario`
- `tool_run_scenario`
- `tool_validate_scenarios`

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
