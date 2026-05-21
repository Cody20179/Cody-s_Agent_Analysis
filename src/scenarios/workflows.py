from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Callable

from src.anomaly.pipeline import check_anomaly as _check_anomaly
from src.common import err, ok
from src.config import OUTPUTS_DIR
from src.data.update import status as _data_status
from src.data.update import update as _update_data
from src.forecast.pipeline import forecast_future as _forecast_future
from src.forecast.pipeline import train_forecast as _train_forecast
from src.scenarios.registry import SCENARIOS, Scenario
from src.state.gmm import apply_state_model as _apply_state_model
from src.state.gmm import train_state_model as _train_state_model


DEFAULT_COST_RATE = 4.0


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _today_range() -> tuple[str, str]:
    now = datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return _iso(start), _iso(now)


def _last_week_range() -> tuple[str, str]:
    today = datetime.now().date()
    this_monday = today - timedelta(days=today.weekday())
    start = datetime.combine(this_monday - timedelta(days=7), datetime.min.time())
    end = datetime.combine(this_monday, datetime.min.time()) - timedelta(seconds=1)
    return _iso(start), _iso(end)


def _period(start: str | None, end: str | None) -> dict[str, str | None]:
    return {"start": start, "end": end}


def _state_utilization(state_summary: dict[str, Any]) -> dict[str, Any]:
    running_hours = 0.0
    total_hours = 0.0
    by_state = {}
    for state, values in state_summary.items():
        hours = float(values.get("hours", values.get("duration_hours", 0)) or 0)
        pct = float(values.get("pct", 0) or 0)
        label = str(values.get("label", state))
        by_state[state] = {"label": label, "hours": hours, "pct": pct}
        total_hours += hours
        if "running" in label.lower() or "run" in label.lower():
            running_hours += hours
    utilization = running_hours / total_hours if total_hours else 0.0
    return {
        "running_hours": running_hours,
        "total_hours": total_hours,
        "utilization_rate": utilization,
        "utilization_percent": round(utilization * 100, 2),
        "by_state": by_state,
    }


def _call_tool(trace: list[dict[str, Any]], name: str, fn: Callable, *args, **kwargs) -> Any:
    started = time.perf_counter()
    index = len(trace) + 1
    try:
        result = fn(*args, **kwargs)
        trace.append({
            "index": index,
            "tool": name,
            "success": True,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return result
    except Exception as exc:
        trace.append({
            "index": index,
            "tool": name,
            "success": False,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": str(exc),
        })
        raise


def _with_contract(scenario: Scenario, payload: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        **payload,
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "category": scenario.category,
        "expected_tools": list(scenario.expected_tools),
        "tool_trace": trace,
        "tool_flow_signature": ">".join(item["tool"] for item in trace),
    }
    missing = [key for key in scenario.required_outputs if key not in payload]
    payload["required_outputs"] = list(scenario.required_outputs)
    payload["output_schema_passed"] = not missing
    payload["missing_outputs"] = missing
    return payload


def _state_period_scenario(scenario: Scenario, start: str | None, end: str | None) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    result = _call_tool(trace, "tool_apply_state_model", _apply_state_model, start=start, end=end)
    state_summary = result.get("state_summary", {})
    payload = {
        "period": _period(start, end),
        "state_summary": state_summary,
        "utilization": _state_utilization(state_summary),
        "key_files": result.get("files", {}),
        "rows": result.get("rows"),
        "best_k": result.get("best_k"),
    }
    return ok(OUTPUTS_DIR / "scenarios", scenario.display_name, key_files=payload["key_files"], metrics=_with_contract(scenario, payload, trace))


def query_today_status() -> dict[str, Any]:
    start, end = _today_range()
    return _state_period_scenario(SCENARIOS["today_status"], start, end)


def query_status_period(start: str | None = None, end: str | None = None) -> dict[str, Any]:
    return _state_period_scenario(SCENARIOS["period_status"], start, end)


def query_utilization(start: str | None = None, end: str | None = None) -> dict[str, Any]:
    return _state_period_scenario(SCENARIOS["utilization"], start, end)


def query_last_week_runtime() -> dict[str, Any]:
    start, end = _last_week_range()
    result = _state_period_scenario(SCENARIOS["last_week_runtime"], start, end)
    metrics = result["metrics"]
    metrics["runtime_summary"] = {
        "running_hours": metrics["utilization"]["running_hours"],
        "total_hours": metrics["utilization"]["total_hours"],
        "note": "Runtime is estimated from model state labels over the last calendar week.",
    }
    metrics["output_schema_passed"] = all(key in metrics for key in metrics["required_outputs"])
    metrics["missing_outputs"] = [key for key in metrics["required_outputs"] if key not in metrics]
    return result


def predict_next_week_power(model_names: list[str] | None = None) -> dict[str, Any]:
    scenario = SCENARIOS["next_week_power_forecast"]
    trace: list[dict[str, Any]] = []
    result = _call_tool(trace, "tool_forecast_future", _forecast_future, days=[7], model_names=model_names)
    payload = {
        "period": {"days": 7},
        "forecast_summary": result.get("summary", {}),
        "model_names": list((model_names or result.get("summary", {}).keys())),
        "key_files": result.get("files", {}),
    }
    return ok(OUTPUTS_DIR / "scenarios", scenario.display_name, key_files=payload["key_files"], metrics=_with_contract(scenario, payload, trace))


def predict_next_month_cost(rate: float = DEFAULT_COST_RATE, model_names: list[str] | None = None) -> dict[str, Any]:
    scenario = SCENARIOS["next_month_cost_forecast"]
    trace: list[dict[str, Any]] = []
    result = _call_tool(trace, "tool_forecast_future", _forecast_future, days=[30], model_names=model_names)
    summary = result.get("summary", {})
    estimates = {}
    for model, values in summary.items():
        if not isinstance(values, dict):
            continue
        horizon = values.get("30d", {})
        predicted_kwh = float(horizon.get("increment", horizon.get("yhat", 0)) or 0)
        estimates[model] = {
            "predicted_kwh": predicted_kwh,
            "estimated_cost": round(predicted_kwh * rate, 2),
        }
    trace.append({
        "index": len(trace) + 1,
        "tool": "cost_calculation",
        "success": True,
        "duration_ms": 0,
    })
    payload = {
        "period": {"days": 30},
        "forecast_summary": summary,
        "rate": rate,
        "estimated_cost": estimates,
        "key_files": result.get("files", {}),
    }
    return ok(OUTPUTS_DIR / "scenarios", scenario.display_name, key_files=payload["key_files"], metrics=_with_contract(scenario, payload, trace))


def check_period_anomaly(start: str, end: str, min_models: int = 2) -> dict[str, Any]:
    scenario = SCENARIOS["period_anomaly_check"]
    trace: list[dict[str, Any]] = []
    result = _call_tool(trace, "tool_check_anomaly", _check_anomaly, start=start, end=end, min_models=min_models)
    payload = {
        "period": _period(start, end),
        "verdict": result.get("verdict"),
        "anomaly": result,
        "key_files": {
            "detail_csv": result.get("detail_csv"),
            "timeline_plot": result.get("timeline_plot"),
            "score_plot": result.get("score_plot"),
            "vote_plot": result.get("vote_plot"),
        },
    }
    return ok(OUTPUTS_DIR / "scenarios", scenario.display_name, key_files=payload["key_files"], metrics=_with_contract(scenario, payload, trace))


def initialize_project(start: str | None = None, end: str | None = None, sensors: str | list[str] = "all", fetch_data: bool = False) -> dict[str, Any]:
    scenario = SCENARIOS["initialize_project"]
    trace: list[dict[str, Any]] = []
    steps: dict[str, Any] = {}
    if fetch_data:
        steps["update_data"] = _call_tool(trace, "tool_update_data", _update_data, sensors=sensors, verbose=True)
    else:
        steps["data_status"] = _call_tool(trace, "tool_data_status", _data_status)
    steps["state"] = _call_tool(trace, "tool_train_state_model", _train_state_model, start=start, end=end)
    steps["forecast"] = _call_tool(trace, "tool_train_forecast", _train_forecast, target="dy", models=["BaselineLastWeek", "Prophet"])
    steps["anomaly"] = _call_tool(trace, "tool_train_anomaly_detection", __import__("src.anomaly.pipeline", fromlist=["train_anomaly_detection"]).train_anomaly_detection)
    payload = {
        "initialization": steps,
        "period": _period(start, end),
    }
    return ok(OUTPUTS_DIR / "scenarios", scenario.display_name, metrics=_with_contract(scenario, payload, trace))


def retrain_all_models(start: str | None = None, end: str | None = None) -> dict[str, Any]:
    scenario = SCENARIOS["retrain_all_models"]
    trace: list[dict[str, Any]] = []
    training = {
        "state": _call_tool(trace, "tool_train_state_model", _train_state_model, start=start, end=end),
        "forecast": _call_tool(trace, "tool_train_forecast", _train_forecast, target="dy", models=["BaselineLastWeek", "Prophet"]),
        "anomaly": _call_tool(trace, "tool_train_anomaly_detection", __import__("src.anomaly.pipeline", fromlist=["train_anomaly_detection"]).train_anomaly_detection),
    }
    payload = {"training": training, "period": _period(start, end)}
    return ok(OUTPUTS_DIR / "scenarios", scenario.display_name, metrics=_with_contract(scenario, payload, trace))


def run_scenario(scenario_id: str, **kwargs) -> dict[str, Any]:
    workflows = {
        "today_status": query_today_status,
        "period_status": query_status_period,
        "last_week_runtime": query_last_week_runtime,
        "utilization": query_utilization,
        "next_week_power_forecast": predict_next_week_power,
        "next_month_cost_forecast": predict_next_month_cost,
        "period_anomaly_check": check_period_anomaly,
        "initialize_project": initialize_project,
        "retrain_all_models": retrain_all_models,
    }
    if scenario_id not in workflows:
        return err(f"unknown scenario: {scenario_id}")
    try:
        return workflows[scenario_id](**kwargs)
    except Exception as exc:
        return err(str(exc))
