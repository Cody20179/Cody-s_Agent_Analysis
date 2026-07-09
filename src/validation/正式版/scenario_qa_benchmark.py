from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from main import apply_state_model, check_anomaly, data_status, forecast_future
from src.config import OUTPUTS_DIR


MODEL_LIST = Path("模型列表.md")
DEFAULT_USER_ID = "Cody"
ANALYSIS_ROOT = "/Users/cody20179/Desktop/Code/Git_My_Project/Cody-s_Agent_Analysis"
CLOUD_MODELS = {"glm-5.1", "kimi-k2.6", "deepseek-v4-flash"}
EMBEDDING_MODELS = {"qwen3-embedding:4b", "qwen3-embedding:8b"}
VISION_MODELS = {"qwen3-vl:32b", "qwen3-vl:8b"}
MODEL_SCOPES = {"cloud", "local-language", "all-language"}
MODEL_ALIASES = {"gml-5.1": "glm-5.1"}
SYSTEM_PROMPT = f"""
你是 Cody Agent 的 CNC 分析助手。回答工廠使用者的自然語言問題時，必須優先用 run_python 調用
{ANALYSIS_ROOT}/main.py 內的分析函式，不要憑空推測數據。

可用函式：
- data_status()
- apply_state_model(start=None, end=None)
- forecast_future(days=None, model_names=None, target=None)
- check_anomaly(start, end, min_models=2)

測試基準日期是 2026-05-23。
若題目提到本週，指 2026-05-18 00:00:00 到 2026-05-23 00:00:00。
若題目提到上週，指 2026-05-11 00:00:00 到 2026-05-18 00:00:00。
請用繁體中文回答，列出關鍵數值、單位、使用工具，以及一句結論。
""".strip()


@dataclass(frozen=True)
class ModelSpec:
    source: str
    provider: str
    base_url: str
    api_key: str
    model_name: str


@dataclass(frozen=True)
class ScenarioCase:
    scenario_id: str
    category: str
    question: str
    expected_functions: tuple[str, ...]
    numeric_expectations: tuple[str, ...]
    exact_expectations: tuple[str, ...] = ()


SCENARIO_CASES = [
    ScenarioCase(
        "S01_data_coverage",
        "data_query",
        "目前 CNC 有哪些感測器資料？資料時間範圍到哪？",
        ("data_status",),
        ("sensor_count",),
    ),
    ScenarioCase(
        "S02_period_state_distribution",
        "state_query",
        "幫我查 2026-05-01 的機台狀態分布。",
        ("apply_state_model",),
        ("rows", "off_hours", "running_high_hours", "running_low_hours", "idle_hours"),
    ),
    ScenarioCase(
        "S03_period_utilization",
        "state_query",
        "幫我看 2026-05-01 的機台稼動率。",
        ("apply_state_model",),
        ("running_hours", "total_hours", "utilization_percent"),
    ),
    ScenarioCase(
        "S04_last_week_runtime",
        "state_query",
        "幫我看上週機台運轉與停機時數。",
        ("apply_state_model",),
        ("running_hours", "off_hours", "total_hours", "utilization_percent"),
    ),
    ScenarioCase(
        "S05_this_week_status",
        "state_query",
        "幫我看本週機台狀態。",
        ("apply_state_model",),
        ("running_hours", "off_hours", "total_hours", "utilization_percent"),
    ),
    ScenarioCase(
        "S06_next_week_power",
        "forecast",
        "幫我預測下週耗電量。",
        ("forecast_future",),
        ("forecast_7d_increment",),
    ),
    ScenarioCase(
        "S07_next_month_cost",
        "forecast",
        "幫我預測下個月電費，電價用每度 4 元。",
        ("forecast_future",),
        ("forecast_30d_increment", "estimated_30d_cost"),
    ),
    ScenarioCase(
        "S08_period_anomaly",
        "anomaly",
        "幫我檢查 2026-05-01 到 2026-05-02 是否異常。",
        ("check_anomaly",),
        ("rows", "anomaly_count", "anomaly_rate"),
        ("verdict",),
    ),
]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80]


def _normalize_base_url(url: str) -> str:
    url = url.strip().strip('"')
    url = re.sub(r"/v1/models/?$", "", url)
    url = re.sub(r"/v1/?$", "", url)
    return url.rstrip("/")


def _parse_model_list(path: Path) -> list[ModelSpec]:
    text = path.read_text(encoding="utf-8")
    source_a = re.search(r"Model From A.*?URL:\"([^\"]+)\".*?Key:\"([^\"]*)\".*?Model List:\s*(.*?)\n\s*\n\s*Model From B", text, re.S)
    source_b = re.search(r"Model From B.*?URL:\"([^\"]+)\".*?Key:\"([^\"]*)\".*?Model List:\"(.*)\"\s*$", text, re.S)
    models: list[ModelSpec] = []
    if source_a:
        url, key, raw_models = source_a.groups()
        for name in re.findall(r"^[A-Za-z0-9_.:-]+$", raw_models, re.M):
            clean_name = name.strip()
            models.append(ModelSpec("A", "ollama", _normalize_base_url(url), key, MODEL_ALIASES.get(clean_name, clean_name)))
    if source_b:
        url, key, raw_json = source_b.groups()
        try:
            payload = json.loads(raw_json)
            names = [item["id"] for item in payload.get("data", []) if item.get("id")]
        except Exception:
            names = re.findall(r'"id"\s*:\s*"([^"]+)"', raw_json)
        for name in names:
            clean_name = name.strip()
            models.append(ModelSpec("B", "openai", _normalize_base_url(url), key, MODEL_ALIASES.get(clean_name, clean_name)))
    return models


def _is_local_language_model(model: ModelSpec) -> bool:
    return (
        model.model_name not in CLOUD_MODELS
        and model.model_name not in EMBEDDING_MODELS
        and model.model_name not in VISION_MODELS
    )


def _select_models(models: list[ModelSpec], model_scope: str) -> list[ModelSpec]:
    if model_scope not in MODEL_SCOPES:
        raise ValueError(f"unknown model scope: {model_scope}")
    if model_scope == "cloud":
        return [model for model in models if model.model_name in CLOUD_MODELS]
    if model_scope == "local-language":
        return [model for model in models if _is_local_language_model(model)]
    return [
        model for model in models
        if model.model_name not in EMBEDDING_MODELS and model.model_name not in VISION_MODELS
    ]


def _token_proxy(text: object) -> int:
    if text is None:
        return 0
    s = str(text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    non_cjk = len(s) - cjk
    return int(math.ceil(cjk * 1.15 + non_cjk / 4.0))


def _request_json(
    base_url: str,
    path: str,
    method: str = "GET",
    body: dict | None = None,
    user_id: str = DEFAULT_USER_ID,
    timeout: int = 60,
) -> tuple[int, Any]:
    data = None
    headers = {"X-User-ID": user_id}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"detail": raw}
        return exc.code, payload


def _create_group(base_url: str, user_id: str, name: str, description: str) -> dict[str, Any]:
    status, payload = _request_json(
        base_url,
        "/sessions/groups",
        method="POST",
        body={"name": name, "description": description},
        user_id=user_id,
    )
    if status not in {200, 201}:
        raise RuntimeError(f"cannot create group: {status} {payload}")
    return payload["group"]


def _create_session(base_url: str, user_id: str, group_id: str, model: ModelSpec, case: ScenarioCase) -> str:
    body = {
        "user_id": user_id,
        "group_id": group_id,
        "provider": model.provider,
        "base_url": model.base_url,
        "api_key": model.api_key,
        "model_name": model.model_name,
        "system_name": f"scenario-qa-{case.scenario_id}",
        "tool_names": ["run_python"],
        "mcp_names": [],
    }
    status, payload = _request_json(base_url, "/sessions", method="POST", body=body, user_id=user_id, timeout=120)
    if status not in {200, 201}:
        raise RuntimeError(f"cannot create session: {status} {payload}")
    return payload["session_id"]


def _chat(base_url: str, session_id: str, prompt: str, user_id: str, timeout: int) -> dict[str, Any]:
    body = {"message": prompt, "use_memory": False}
    req = urllib.request.Request(
        base_url + f"/sessions/{session_id}/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-User-ID": user_id},
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    turn = 0
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                if time.perf_counter() - started > timeout:
                    error = f"hard timeout after {timeout}s"
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                events.append(payload)
                tag = payload.get("tag", "")
                if tag == "result":
                    answer_parts.append(payload.get("chunk", ""))
                elif tag == "done":
                    turn = int(payload.get("turn") or 0)
                    break
                elif tag == "error":
                    error = payload.get("chunk", "")
    except Exception as exc:
        error = str(exc)
    answer = "".join(answer_parts)
    return {
        "turn": turn,
        "status": "error" if error else "success",
        "error": error,
        "duration_ms_client": int((time.perf_counter() - started) * 1000),
        "answer": answer,
        "events": events,
    }


def _benchmark(base_url: str, user_id: str, group_id: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"group_id": group_id, "limit": 50000})
    status, payload = _request_json(base_url, f"/evaluation/benchmark?{query}", user_id=user_id, timeout=120)
    if status != 200:
        raise RuntimeError(f"cannot read benchmark: {status} {payload}")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _running_summary(state_summary: dict[str, Any]) -> tuple[float, float, float]:
    running_hours = 0.0
    total_hours = 0.0
    off_hours = 0.0
    for state, values in state_summary.items():
        hours = float(values.get("hours", 0) or 0)
        total_hours += hours
        if "running" in str(state).lower():
            running_hours += hours
        if str(state).lower() == "off":
            off_hours += hours
    return running_hours, total_hours, off_hours


def build_ground_truth() -> dict[str, dict[str, Any]]:
    status = data_status()
    sensors = status.get("metrics", {}).get("sensors", [])

    state_day = apply_state_model(start="2026-05-01 00:00:00", end="2026-05-02 00:00:00")
    day_metrics = state_day.get("metrics", {})
    day_summary = day_metrics.get("state_summary", {})
    day_running, day_total, day_off = _running_summary(day_summary)

    last_week = apply_state_model(start="2026-05-11 00:00:00", end="2026-05-18 00:00:00")
    last_metrics = last_week.get("metrics", {})
    last_summary = last_metrics.get("state_summary", {})
    last_running, last_total, last_off = _running_summary(last_summary)

    this_week = apply_state_model(start="2026-05-18 00:00:00", end="2026-05-23 00:00:00")
    this_metrics = this_week.get("metrics", {})
    this_summary = this_metrics.get("state_summary", {})
    this_running, this_total, this_off = _running_summary(this_summary)

    forecast = forecast_future(days=[7, 30], model_names=["Prophet"])
    prophet = forecast.get("metrics", {}).get("Prophet", {})

    anomaly = check_anomaly("2026-05-01 00:00:00", "2026-05-02 00:00:00")
    anomaly_metrics = anomaly.get("metrics", {})

    return {
        "S01_data_coverage": {
            "sensor_count": len(sensors),
            "sensors": sensors,
            "required_keywords": ["感測器", "資料", "時間"],
        },
        "S02_period_state_distribution": {
            "rows": day_metrics.get("rows"),
            "state_summary": day_summary,
            "off_hours": float(day_summary.get("Off", {}).get("hours", 0) or 0),
            "running_high_hours": float(day_summary.get("Running_High", {}).get("hours", 0) or 0),
            "running_low_hours": float(day_summary.get("Running_Low", {}).get("hours", 0) or 0),
            "idle_hours": float(day_summary.get("Idle", {}).get("hours", 0) or 0),
            "required_keywords": ["Off", "Running", "Idle"],
        },
        "S03_period_utilization": {
            "running_hours": round(day_running, 3),
            "total_hours": round(day_total, 3),
            "utilization_percent": round((day_running / day_total * 100) if day_total else 0, 2),
            "required_keywords": ["稼動", "Running"],
        },
        "S04_last_week_runtime": {
            "running_hours": round(last_running, 3),
            "off_hours": round(last_off, 3),
            "total_hours": round(last_total, 3),
            "utilization_percent": round((last_running / last_total * 100) if last_total else 0, 2),
            "required_keywords": ["上週", "運轉", "停機"],
        },
        "S05_this_week_status": {
            "running_hours": round(this_running, 3),
            "off_hours": round(this_off, 3),
            "total_hours": round(this_total, 3),
            "utilization_percent": round((this_running / this_total * 100) if this_total else 0, 2),
            "state_summary": this_summary,
            "required_keywords": ["本週", "狀態"],
        },
        "S06_next_week_power": {
            "forecast_7d_increment": float(prophet.get("7d", {}).get("increment", 0) or 0),
            "forecast_7d_yhat": float(prophet.get("7d", {}).get("yhat", 0) or 0),
            "required_keywords": ["下週", "耗電", "預測"],
        },
        "S07_next_month_cost": {
            "forecast_30d_increment": float(prophet.get("30d", {}).get("increment", 0) or 0),
            "forecast_30d_yhat": float(prophet.get("30d", {}).get("yhat", 0) or 0),
            "rate": 4.0,
            "estimated_30d_cost": round(float(prophet.get("30d", {}).get("increment", 0) or 0) * 4.0, 2),
            "required_keywords": ["下個月", "電費", "4"],
        },
        "S08_period_anomaly": {
            "verdict": anomaly_metrics.get("verdict"),
            "rows": anomaly_metrics.get("rows"),
            "anomaly_count": anomaly_metrics.get("anomaly_count"),
            "anomaly_rate": anomaly_metrics.get("anomaly_rate"),
            "model_votes": anomaly_metrics.get("model_votes"),
            "required_keywords": ["異常", "NORMAL"],
        },
    }


def _extract_numbers(text: str) -> list[float]:
    values = []
    for raw in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return values


def _number_present(answer: str, expected: float, tolerance: float) -> bool:
    return any(abs(value - expected) <= tolerance for value in _extract_numbers(answer))


def _numeric_tolerance(key: str, value: float) -> float:
    if "percent" in key or "rate" in key:
        return 0.8
    if "cost" in key:
        return 15.0
    if "forecast" in key:
        return 2.0
    if "hours" in key:
        return 0.25
    return max(1.0, abs(float(value)) * 0.01)


def _event_text(events: list[dict[str, Any]], tag: str) -> str:
    return "\n".join(str(event.get("chunk", "")) for event in events if event.get("tag") == tag)


def _called_functions(events: list[dict[str, Any]]) -> set[str]:
    text = _event_text(events, "tool_input") + "\n" + _event_text(events, "tool_output")
    return {name for name in ("data_status", "apply_state_model", "forecast_future", "check_anomaly") if name in text}


def score_answer(case: ScenarioCase, ground_truth: dict[str, Any], answer: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    called = _called_functions(events)
    expected = set(case.expected_functions)
    tool_score = 25 if expected.issubset(called) else 0

    answer_text = answer
    event_text = _event_text(events, "tool_input") + "\n" + _event_text(events, "tool_output")
    numeric_total = len(case.numeric_expectations)
    numeric_hits = 0
    misses = []
    for key in case.numeric_expectations:
        value = ground_truth.get(key)
        if value is None:
            misses.append(key)
            continue
        ok = _number_present(answer_text, float(value), _numeric_tolerance(key, float(value)))
        numeric_hits += int(ok)
        if not ok:
            misses.append(key)
    fact_score = round(30 * numeric_hits / numeric_total, 2) if numeric_total else 30

    exact_total = len(case.exact_expectations)
    exact_hits = 0
    for key in case.exact_expectations:
        value = str(ground_truth.get(key, ""))
        ok = bool(value) and value.lower() in answer_text.lower()
        exact_hits += int(ok)
        if not ok:
            misses.append(key)
    exact_score = round(20 * exact_hits / exact_total, 2) if exact_total else 20

    keywords = ground_truth.get("required_keywords", [])
    keyword_hits = sum(1 for keyword in keywords if str(keyword).lower() in answer.lower())
    format_score = 10 if len(answer.strip()) >= 50 and keyword_hits >= max(1, min(2, len(keywords))) else 5 if answer.strip() else 0

    hallucination_penalty = 0
    if "無法" in answer and tool_score > 0 and fact_score >= 15:
        hallucination_penalty += 5
    if "我猜" in answer or "可能是" in answer:
        hallucination_penalty += 5

    parameter_score = 15
    if case.scenario_id in {"S02_period_state_distribution", "S03_period_utilization", "S08_period_anomaly"}:
        parameter_score = 15 if ("2026-05-01" in event_text and "2026-05-02" in event_text) else 5
    elif case.scenario_id == "S04_last_week_runtime":
        parameter_score = 15 if ("2026-05-11" in event_text and "2026-05-18" in event_text) else 5
    elif case.scenario_id == "S05_this_week_status":
        parameter_score = 15 if ("2026-05-18" in event_text and "2026-05-23" in event_text) else 5
    elif case.scenario_id == "S06_next_week_power":
        parameter_score = 15 if ("7" in event_text) else 5
    elif case.scenario_id == "S07_next_month_cost":
        parameter_score = 15 if ("30" in event_text and "4" in answer_text) else 5

    total = max(0, round(tool_score + parameter_score + fact_score + exact_score + format_score - hallucination_penalty, 2))
    return {
        "tool_score": tool_score,
        "parameter_score": parameter_score,
        "fact_score": fact_score,
        "exact_score": exact_score,
        "format_score": format_score,
        "hallucination_penalty": hallucination_penalty,
        "total_score": total,
        "called_functions": "|".join(sorted(called)),
        "missing_expectations": "|".join(misses),
    }


def _case_result(
    run_id: str,
    model: ModelSpec,
    group: dict[str, Any],
    case: ScenarioCase,
    repeat_index: int,
    session_id: str,
    chat: dict[str, Any],
    gt: dict[str, Any],
) -> dict[str, Any]:
    score = score_answer(case, gt, chat.get("answer", ""), chat.get("events", []))
    tool_inputs = [event for event in chat.get("events", []) if event.get("tag") == "tool_input"]
    tool_outputs = [event for event in chat.get("events", []) if event.get("tag") == "tool_output"]
    thinking = [event for event in chat.get("events", []) if event.get("tag") == "thinking"]
    answer = chat.get("answer", "")
    return {
        "run_id": run_id,
        "model_source": model.source,
        "model_name": model.model_name,
        "group_id": group["group_id"],
        "group_name": group["name"],
        "scenario_id": case.scenario_id,
        "category": case.category,
        "repeat_index": repeat_index,
        "session_id": session_id,
        "turn": chat.get("turn", 0),
        "status": chat.get("status", ""),
        "question": case.question,
        "expected_functions": "|".join(case.expected_functions),
        "tool_input_events": len(tool_inputs),
        "tool_output_events": len(tool_outputs),
        "thinking_events_client": len(thinking),
        "thinking_chars_client": sum(len(str(event.get("chunk", ""))) for event in thinking),
        "duration_ms_client": chat.get("duration_ms_client", 0),
        "answer_chars_client": len(answer),
        "estimated_prompt_tokens": _token_proxy(case.question),
        "estimated_answer_tokens_client": _token_proxy(answer),
        "actual_prompt_tokens": "",
        "actual_completion_tokens": "",
        "actual_total_tokens": "",
        "token_data_quality": "estimated_from_text_length; actual provider usage not exposed by System benchmark API",
        "error": chat.get("error", ""),
        "answer_preview": answer[:1000],
        **score,
    }


def _write_outputs(out: Path, rows: list[dict[str, Any]], ground_truth: dict[str, Any], benchmark: dict[str, Any]) -> None:
    _write_csv(out / "scenario_qa_runs.csv", rows)
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    summary = df.groupby(["scenario_id", "category"], dropna=False).agg(
        runs=("scenario_id", "count"),
        avg_total_score=("total_score", "mean"),
        avg_tool_score=("tool_score", "mean"),
        avg_parameter_score=("parameter_score", "mean"),
        avg_fact_score=("fact_score", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_tool_input_events=("tool_input_events", "mean"),
        avg_answer_chars=("answer_chars_client", "mean"),
        pass_rate=("total_score", lambda s: float((pd.Series(s) >= 80).mean())),
    ).reset_index()
    for col in summary.columns:
        if col.startswith("avg_") or col.endswith("_rate"):
            summary[col] = summary[col].astype(float).round(3)
    summary.to_csv(out / "scenario_qa_summary.csv", index=False, encoding="utf-8-sig")

    model_summary = df.groupby("model_name", dropna=False).agg(
        runs=("model_name", "count"),
        avg_total_score=("total_score", "mean"),
        pass_rate=("total_score", lambda s: float((pd.Series(s) >= 80).mean())),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
    ).reset_index()
    model_summary.to_csv(out / "model_summary.csv", index=False, encoding="utf-8-sig")
    _plot_outputs(out, summary, model_summary, df)


def _plot_outputs(out: Path, summary: pd.DataFrame, model_summary: pd.DataFrame, runs: pd.DataFrame) -> None:
    chart_dir = out / "charts"
    chart_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
    })

    def barh(data: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str, filename: str, suffix: str = "", precision: int = 1) -> None:
        data = data.sort_values(value_col)
        labels = [str(v).replace("_", "\n") for v in data[label_col]]
        values = data[value_col].astype(float).tolist()
        fig, ax = plt.subplots(figsize=(11, max(6, 0.55 * len(data) + 2)))
        bars = ax.barh(labels, values, color="#0f8fa8")
        ax.set_title(title, weight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        high = max(values) if values else 1
        ax.set_xlim(0, max(1, high * 1.22))
        for bar, value in zip(bars, values):
            ax.text(value + max(high, 1) * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.{precision}f}{suffix}", va="center", weight="bold")
        fig.savefig(chart_dir / filename, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    barh(summary, "scenario_id", "avg_total_score", "Scenario QA Score by Case", "Average score", "fig01_score_by_scenario.png", "", 1)
    barh(model_summary, "model_name", "avg_total_score", "Scenario QA Score by Model", "Average score", "fig02_score_by_model.png", "", 1)
    barh(summary, "scenario_id", "avg_duration_s", "Scenario QA Runtime by Case", "Average runtime (s)", "fig03_runtime_by_scenario.png", "s", 1)

    pivot = runs.pivot_table(index="model_name", columns="scenario_id", values="total_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12.5, max(5.5, 0.5 * len(pivot) + 2)))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_title("Model by Scenario QA Score Matrix", weight="bold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c).replace("_", "\n") for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=10, weight="bold", color="white" if value >= 75 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Score")
    fig.savefig(chart_dir / "fig04_model_scenario_score_matrix.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(
    base_url: str,
    user_id: str,
    output_dir: Path | None,
    repeat: int,
    timeout: int,
    model_scope: str,
    model_limit: int | None,
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = output_dir or OUTPUTS_DIR / "validation" / "scenario_qa_agent" / run_id
    out.mkdir(parents=True, exist_ok=True)
    ground_truth = build_ground_truth()
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    models = _select_models(_parse_model_list(MODEL_LIST), model_scope)
    if model_limit is not None:
        models = models[:model_limit]
    group = _create_group(base_url, user_id, f"scenario-qa-{run_id}", "Scenario QA benchmark with direct-tool ground truth")
    rows: list[dict[str, Any]] = []
    for model in models:
        for case in SCENARIO_CASES:
            for repeat_index in range(1, repeat + 1):
                try:
                    session_id = _create_session(base_url, user_id, group["group_id"], model, case)
                    prompt = f"{SYSTEM_PROMPT}\n\n使用者問題：{case.question}"
                    chat = _chat(base_url, session_id, prompt, user_id, timeout)
                    row = _case_result(run_id, model, group, case, repeat_index, session_id, chat, ground_truth[case.scenario_id])
                except Exception as exc:
                    row = {
                        "run_id": run_id,
                        "model_source": model.source,
                        "model_name": model.model_name,
                        "group_id": group["group_id"],
                        "group_name": group["name"],
                        "scenario_id": case.scenario_id,
                        "category": case.category,
                        "repeat_index": repeat_index,
                        "session_id": "",
                        "turn": 0,
                        "status": "error",
                        "question": case.question,
                        "expected_functions": "|".join(case.expected_functions),
                        "tool_input_events": 0,
                        "tool_output_events": 0,
                        "thinking_events_client": 0,
                        "thinking_chars_client": 0,
                        "duration_ms_client": 0,
                        "answer_chars_client": 0,
                        "estimated_prompt_tokens": _token_proxy(case.question),
                        "estimated_answer_tokens_client": 0,
                        "actual_prompt_tokens": "",
                        "actual_completion_tokens": "",
                        "actual_total_tokens": "",
                        "token_data_quality": "estimated_from_text_length; actual provider usage not exposed by System benchmark API",
                        "error": str(exc),
                        "answer_preview": "",
                        "tool_score": 0,
                        "parameter_score": 0,
                        "fact_score": 0,
                        "exact_score": 0,
                        "format_score": 0,
                        "hallucination_penalty": 0,
                        "total_score": 0,
                        "called_functions": "",
                        "missing_expectations": "|".join(case.numeric_expectations + case.exact_expectations),
                    }
                rows.append(row)
                _write_csv(out / "scenario_qa_runs.csv", rows)
    benchmark = _benchmark(base_url, user_id, group["group_id"])
    _write_outputs(out, rows, ground_truth, benchmark)
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "user_id": user_id,
        "models": [model.model_name for model in models],
        "model_scope": model_scope,
        "scenario_count": len(SCENARIO_CASES),
        "repeat": repeat,
        "rows": len(rows),
        "group_id": group["group_id"],
        "output_dir": str(out),
        "ground_truth_source": "Generated by direct calls to main.py analysis functions before model QA.",
        "actual_provider_token_usage_available": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario QA benchmark with direct-tool ground truth.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--model-scope", choices=sorted(MODEL_SCOPES), default="cloud")
    parser.add_argument("--model-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(
        run(
            base_url=args.base_url.rstrip("/"),
            user_id=args.user_id,
            output_dir=args.output_dir,
            repeat=args.repeat,
            timeout=args.timeout,
            model_scope=args.model_scope,
            model_limit=args.model_limit,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
