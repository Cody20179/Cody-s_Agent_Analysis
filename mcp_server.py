from __future__ import annotations

import asyncio
import contextlib
import json
import sys

_real_stdout = sys.stdout
sys.stdout = sys.stderr

from main import (  # noqa: E402
    check_anomaly,
    data_status,
    forecast_future,
    run_all,
    run_state_analysis,
    train_anomaly_detection,
    train_forecast,
    update_data,
)

sys.stdout = _real_stdout

try:
    from mcp.server import FastMCP
except ModuleNotFoundError as exc:
    raise SystemExit("mcp package is required to run mcp_server.py") from exc

mcp = FastMCP("cnc-analysis-v2")

@contextlib.contextmanager
def _capture_stdout():
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old

async def _run(fn, *args, **kwargs) -> str:
    def call():
        with _capture_stdout():
            return fn(*args, **kwargs)
    result = await asyncio.to_thread(call)
    return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
async def tool_update_data(sensors: str = "all") -> str:
    sensor_arg = sensors if sensors == "all" else [x.strip() for x in sensors.split(",") if x.strip()]
    return await _run(update_data, sensors=sensor_arg, verbose=True)

@mcp.tool()
async def tool_data_status() -> str:
    return await _run(data_status)

@mcp.tool()
async def tool_run_state_analysis(start: str = "", end: str = "") -> str:
    return await _run(run_state_analysis, start=start or None, end=end or None)

@mcp.tool()
async def tool_train_forecast(target: str = "dy", models: str = "", months_back: int = 0) -> str:
    model_arg = [x.strip() for x in models.split(",") if x.strip()] if models else None
    return await _run(train_forecast, target=target, models=model_arg, months_back=months_back or None)

@mcp.tool()
async def tool_forecast_future(days: str = "3,7,14,30", model_names: str = "") -> str:
    day_arg = [int(x.strip()) for x in days.split(",") if x.strip()]
    model_arg = [x.strip() for x in model_names.split(",") if x.strip()] if model_names else None
    return await _run(forecast_future, days=day_arg, model_names=model_arg)

@mcp.tool()
async def tool_train_anomaly_detection() -> str:
    return await _run(train_anomaly_detection)

@mcp.tool()
async def tool_check_anomaly(start: str, end: str, min_models: int = 2) -> str:
    return await _run(check_anomaly, start=start, end=end, min_models=min_models)

@mcp.tool()
async def tool_run_all() -> str:
    return await _run(run_all)

if __name__ == "__main__":
    mcp.run(transport="stdio")

