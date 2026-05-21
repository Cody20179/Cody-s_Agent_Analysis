from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import OUTPUTS_DIR, ensure_dirs
from src.scenarios.registry import SCENARIOS
from src.scenarios.workflows import run_scenario


VALIDATION_DIR = OUTPUTS_DIR / "validation"


def _default_inputs(scenario_id: str) -> dict[str, Any]:
    if scenario_id in {"period_status", "utilization"}:
        return {"start": None, "end": None}
    if scenario_id == "period_anomaly_check":
        return {"start": "2026-05-01 00:00:00", "end": "2026-05-02 00:00:00"}
    if scenario_id == "next_month_cost_forecast":
        return {"rate": 4.0}
    if scenario_id == "initialize_project":
        return {"fetch_data": False}
    return {}


def _record(run_id: str, scenario_id: str, model_name: str, iteration: int, result: dict[str, Any], duration_ms: float) -> dict[str, Any]:
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    trace = metrics.get("tool_trace", [])
    tool_success = sum(1 for item in trace if item.get("success"))
    return {
        "validation_run_id": run_id,
        "scenario_id": scenario_id,
        "scenario_name": SCENARIOS[scenario_id].display_name,
        "model_name": model_name,
        "iteration": iteration,
        "success": result.get("status") == "ok" and bool(metrics.get("output_schema_passed", False)),
        "status": result.get("status"),
        "error": result.get("message", ""),
        "duration_ms": round(duration_ms, 2),
        "tool_call_count": len(trace),
        "tool_success_count": tool_success,
        "tool_failure_count": len(trace) - tool_success,
        "tool_flow_signature": metrics.get("tool_flow_signature", ""),
        "output_schema_passed": bool(metrics.get("output_schema_passed", False)),
        "missing_outputs": ",".join(metrics.get("missing_outputs", [])),
        "artifact_count": len(result.get("artifacts", [])),
        "answer_length": len(json.dumps(result, ensure_ascii=False, default=str)),
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "thinking_duration_ms": None,
        "thinking_length": None,
    }


def run_validation(scenario_ids: list[str], repeat: int = 1, model_name: str = "workflow-baseline") -> dict[str, Any]:
    ensure_dirs()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    records = []
    for scenario_id in scenario_ids:
        if scenario_id not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario_id}")
        for iteration in range(1, repeat + 1):
            started = time.perf_counter()
            result = run_scenario(scenario_id, **_default_inputs(scenario_id))
            duration_ms = (time.perf_counter() - started) * 1000
            records.append(_record(run_id, scenario_id, model_name, iteration, result, duration_ms))
    json_path = VALIDATION_DIR / f"{run_id}.json"
    csv_path = VALIDATION_DIR / f"{run_id}.csv"
    payload = {
        "validation_run_id": run_id,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "repeat": repeat,
        "records": records,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    return {"status": "ok", "json": str(json_path), "csv": str(csv_path), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario validation.")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), help="Scenario id. Repeat for multiple scenarios.")
    parser.add_argument("--all", action="store_true", help="Run all scenarios.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model-name", default="workflow-baseline")
    args = parser.parse_args()
    scenario_ids = list(SCENARIOS) if args.all else (args.scenario or ["today_status"])
    result = run_validation(scenario_ids, repeat=args.repeat, model_name=args.model_name)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
