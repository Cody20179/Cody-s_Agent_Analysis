from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.error
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
from src.validation.正式版.scenario_qa_benchmark import (
    ANALYSIS_ROOT,
    CLOUD_MODELS,
    DEFAULT_USER_ID,
    MODEL_LIST,
    MODEL_SCOPES,
    ModelSpec,
    _parse_model_list,
    _request_json,
    _select_models,
    _token_proxy,
)


SYSTEM_PROMPT = f"""
你是 Cody Agent 的 CNC 分析助手。回答多輪工廠問題時，必須維持前文條件，尤其是日期區間、電價與使用者剛剛指定的比較對象。
必須優先用 run_python 調用 {ANALYSIS_ROOT}/main.py 內的分析函式，不要憑空推測數據。

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
class TurnSpec:
    conversation_id: str
    turn_index: int
    user_message: str
    expected_functions: tuple[str, ...]
    numeric_expectations: tuple[str, ...]
    exact_expectations: tuple[str, ...] = ()
    expected_markers: tuple[str, ...] = ()
    required_keywords: tuple[str, ...] = ()


CONVERSATIONS: dict[str, list[TurnSpec]] = {
    "state_followup": [
        TurnSpec(
            "state_followup",
            1,
            "幫我查 2026-05-01 的機台狀態分布。",
            ("apply_state_model",),
            ("rows", "off_hours", "running_high_hours", "running_low_hours"),
            expected_markers=("2026-05-01|2026/05/01|5/1|05-01", "2026-05-02|2026/05/02|5/2|05-02"),
            required_keywords=("Off", "Running"),
        ),
        TurnSpec(
            "state_followup",
            2,
            "那稼動率是多少？",
            ("apply_state_model",),
            ("running_hours", "total_hours", "utilization_percent"),
            expected_markers=("2026-05-01|2026/05/01|5/1|05-01",),
            required_keywords=("稼動", "Running"),
        ),
        TurnSpec(
            "state_followup",
            3,
            "改成上週，運轉和停機各多久？",
            ("apply_state_model",),
            ("last_week_running_hours", "last_week_off_hours", "last_week_total_hours"),
            expected_markers=("上週|2026-05-11|2026/05/11|5/11|05-11", "2026-05-18|2026/05/18|5/18|05-18"),
            required_keywords=("上週", "運轉", "停機"),
        ),
        TurnSpec(
            "state_followup",
            4,
            "只看停機時間，給我數值和比例。",
            ("apply_state_model",),
            ("last_week_off_hours", "last_week_off_percent"),
            expected_markers=("上週|2026-05-11|2026/05/11|5/11|05-11",),
            required_keywords=("停機", "Off"),
        ),
        TurnSpec(
            "state_followup",
            5,
            "用一句話比較 2026-05-01 和上週哪個稼動率比較高。",
            ("apply_state_model",),
            ("utilization_percent", "last_week_utilization_percent"),
            expected_markers=("2026-05-01|2026/05/01|5/1|05-01", "上週|2026-05-11|2026/05/11|5/11|05-11"),
            required_keywords=("2026-05-01", "上週", "稼動"),
        ),
    ],
    "forecast_followup": [
        TurnSpec(
            "forecast_followup",
            1,
            "幫我預測下週耗電量。",
            ("forecast_future",),
            ("forecast_7d_increment",),
            expected_markers=("7",),
            required_keywords=("下週", "耗電"),
        ),
        TurnSpec(
            "forecast_followup",
            2,
            "那下個月電費呢？電價用每度 4 元。",
            ("forecast_future",),
            ("forecast_30d_increment", "estimated_30d_cost_4"),
            expected_markers=("30",),
            required_keywords=("下個月", "電費", "4"),
        ),
        TurnSpec(
            "forecast_followup",
            3,
            "如果每度改成 5 元，電費是多少？",
            ("forecast_future",),
            ("forecast_30d_increment", "estimated_30d_cost_5"),
            expected_markers=("30",),
            required_keywords=("5", "電費"),
        ),
        TurnSpec(
            "forecast_followup",
            4,
            "跟剛剛 4 元相比差多少？",
            ("forecast_future",),
            ("cost_delta_5_vs_4",),
            expected_markers=("30",),
            required_keywords=("差", "4", "5"),
        ),
        TurnSpec(
            "forecast_followup",
            5,
            "總結下週耗電與下個月電費風險。",
            ("forecast_future",),
            ("forecast_7d_increment", "estimated_30d_cost_4"),
            expected_markers=("7", "30"),
            required_keywords=("下週", "下個月", "風險"),
        ),
    ],
    "anomaly_to_state_followup": [
        TurnSpec(
            "anomaly_to_state_followup",
            1,
            "幫我檢查 2026-05-01 到 2026-05-02 是否異常。",
            ("check_anomaly",),
            ("anomaly_rows", "anomaly_count", "anomaly_rate"),
            ("NORMAL",),
            expected_markers=("2026-05-01|2026/05/01|5/1|05-01", "2026-05-02|2026/05/02|5/2|05-02"),
            required_keywords=("異常", "NORMAL"),
        ),
        TurnSpec(
            "anomaly_to_state_followup",
            2,
            "哪些模型投票為異常？",
            ("check_anomaly",),
            ("isolation_votes", "svm_votes", "autoencoder_votes"),
            expected_markers=("2026-05-01|2026/05/01|5/1|05-01",),
            required_keywords=("IsolationForest", "OneClassSVM", "Autoencoder"),
        ),
        TurnSpec(
            "anomaly_to_state_followup",
            3,
            "如果沒有異常，這段期間可以怎麼解讀？",
            ("check_anomaly",),
            ("anomaly_count", "anomaly_rate"),
            ("NORMAL",),
            expected_markers=("這段|2026-05-01|2026/05/01|5/1|05-01",),
            required_keywords=("沒有異常", "NORMAL"),
        ),
        TurnSpec(
            "anomaly_to_state_followup",
            4,
            "改查本週的狀態，機台主要在哪些狀態？",
            ("apply_state_model",),
            ("this_week_running_hours", "this_week_off_hours", "this_week_utilization_percent"),
            expected_markers=("本週|2026-05-18|2026/05/18|5/18|05-18", "2026-05-23|2026/05/23|5/23|05-23"),
            required_keywords=("本週", "狀態"),
        ),
        TurnSpec(
            "anomaly_to_state_followup",
            5,
            "給我一句管理摘要，不要編造異常。",
            ("apply_state_model",),
            ("this_week_running_hours", "this_week_off_hours"),
            expected_markers=("本週|2026-05-18|2026/05/18|5/18|05-18",),
            required_keywords=("摘要", "異常"),
        ),
    ],
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def _create_session(base_url: str, user_id: str, group_id: str, model: ModelSpec, conversation_id: str) -> str:
    body = {
        "user_id": user_id,
        "group_id": group_id,
        "provider": model.provider,
        "base_url": model.base_url,
        "api_key": model.api_key,
        "model_name": model.model_name,
        "system_name": f"multi-turn-{conversation_id}",
        "tool_names": ["run_python"],
        "mcp_names": [],
    }
    status, payload = _request_json(base_url, "/sessions", method="POST", body=body, user_id=user_id, timeout=120)
    if status not in {200, 201}:
        raise RuntimeError(f"cannot create session: {status} {payload}")
    return payload["session_id"]


def _chat(base_url: str, session_id: str, prompt: str, user_id: str, timeout: int) -> dict[str, Any]:
    body = {"message": prompt, "use_memory": True}
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
    answer = "".join(answer_parts).strip()
    return {
        "status": "error" if error else "success",
        "error": error,
        "answer": answer,
        "events": events,
        "turn": turn,
        "duration_ms_client": int((time.perf_counter() - started) * 1000),
    }


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


def build_ground_truth() -> dict[str, Any]:
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
    forecast_30d_increment = float(prophet.get("30d", {}).get("increment", 0) or 0)

    anomaly = check_anomaly("2026-05-01 00:00:00", "2026-05-02 00:00:00")
    anomaly_metrics = anomaly.get("metrics", {})
    votes = anomaly_metrics.get("model_votes", {}) or {}

    return {
        "sensor_count": len(sensors),
        "rows": day_metrics.get("rows"),
        "off_hours": float(day_summary.get("Off", {}).get("hours", 0) or 0),
        "running_high_hours": float(day_summary.get("Running_High", {}).get("hours", 0) or 0),
        "running_low_hours": float(day_summary.get("Running_Low", {}).get("hours", 0) or 0),
        "idle_hours": float(day_summary.get("Idle", {}).get("hours", 0) or 0),
        "running_hours": round(day_running, 3),
        "total_hours": round(day_total, 3),
        "utilization_percent": round((day_running / day_total * 100) if day_total else 0, 2),
        "last_week_running_hours": round(last_running, 3),
        "last_week_off_hours": round(last_off, 3),
        "last_week_total_hours": round(last_total, 3),
        "last_week_off_percent": round((last_off / last_total * 100) if last_total else 0, 2),
        "last_week_utilization_percent": round((last_running / last_total * 100) if last_total else 0, 2),
        "this_week_running_hours": round(this_running, 3),
        "this_week_off_hours": round(this_off, 3),
        "this_week_total_hours": round(this_total, 3),
        "this_week_utilization_percent": round((this_running / this_total * 100) if this_total else 0, 2),
        "forecast_7d_increment": float(prophet.get("7d", {}).get("increment", 0) or 0),
        "forecast_7d_yhat": float(prophet.get("7d", {}).get("yhat", 0) or 0),
        "forecast_30d_increment": forecast_30d_increment,
        "forecast_30d_yhat": float(prophet.get("30d", {}).get("yhat", 0) or 0),
        "estimated_30d_cost_4": round(forecast_30d_increment * 4.0, 2),
        "estimated_30d_cost_5": round(forecast_30d_increment * 5.0, 2),
        "cost_delta_5_vs_4": round(forecast_30d_increment, 2),
        "verdict": anomaly_metrics.get("verdict"),
        "anomaly_rows": anomaly_metrics.get("rows"),
        "anomaly_count": anomaly_metrics.get("anomaly_count"),
        "anomaly_rate": anomaly_metrics.get("anomaly_rate"),
        "isolation_votes": votes.get("IsolationForest", 0),
        "svm_votes": votes.get("OneClassSVM", 0),
        "autoencoder_votes": votes.get("Autoencoder", 0),
    }


def _extract_numbers(text: str) -> list[float]:
    values = []
    for raw in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return values


def _numeric_tolerance(key: str, value: float) -> float:
    if "percent" in key or "rate" in key:
        return 0.8
    if "cost" in key:
        return 15.0
    if "forecast" in key:
        return 2.0
    if "hours" in key:
        return 0.25
    if "votes" in key or "count" in key:
        return 0.01
    return max(1.0, abs(float(value)) * 0.01)


def _number_present(answer: str, expected: float, tolerance: float) -> bool:
    return any(abs(value - expected) <= tolerance for value in _extract_numbers(answer))


def _event_text(events: list[dict[str, Any]], tag: str) -> str:
    return "\n".join(str(event.get("chunk", "")) for event in events if event.get("tag") == tag)


def _called_functions(events: list[dict[str, Any]]) -> set[str]:
    text = _event_text(events, "tool_input") + "\n" + _event_text(events, "tool_output")
    return {name for name in ("data_status", "apply_state_model", "forecast_future", "check_anomaly") if name in text}


def _marker_hit(marker: str, text: str) -> bool:
    return any(option and option in text for option in marker.split("|"))


def _score_turn(spec: TurnSpec, ground_truth: dict[str, Any], answer: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    event_text = _event_text(events, "tool_input") + "\n" + _event_text(events, "tool_output")
    called = _called_functions(events)
    expected = set(spec.expected_functions)
    tool_score = 20 if expected.issubset(called) else 0

    marker_source = event_text + "\n" + answer
    marker_hits = sum(1 for marker in spec.expected_markers if _marker_hit(marker, marker_source))
    context_score = round(20 * marker_hits / len(spec.expected_markers), 2) if spec.expected_markers else 20

    numeric_hits = 0
    misses: list[str] = []
    for key in spec.numeric_expectations:
        value = ground_truth.get(key)
        if value is None:
            misses.append(key)
            continue
        ok = _number_present(answer, float(value), _numeric_tolerance(key, float(value)))
        numeric_hits += int(ok)
        if not ok:
            misses.append(key)
    fact_score = round(35 * numeric_hits / len(spec.numeric_expectations), 2) if spec.numeric_expectations else 35

    exact_hits = 0
    for expected_text in spec.exact_expectations:
        ok = expected_text.lower() in answer.lower()
        exact_hits += int(ok)
        if not ok:
            misses.append(expected_text)
    exact_score = round(10 * exact_hits / len(spec.exact_expectations), 2) if spec.exact_expectations else 10

    keyword_hits = sum(1 for keyword in spec.required_keywords if keyword.lower() in answer.lower())
    format_score = 10 if len(answer.strip()) >= 35 and keyword_hits >= max(1, min(2, len(spec.required_keywords))) else 5 if answer.strip() else 0

    drift_penalty = 0
    if spec.turn_index > 1 and context_score < 20:
        drift_penalty += 10
    if "可能是" in answer or "我猜" in answer:
        drift_penalty += 5
    if "無法" in answer and fact_score >= 15:
        drift_penalty += 5

    total = max(0, round(tool_score + context_score + fact_score + exact_score + format_score - drift_penalty, 2))
    return {
        "tool_score": tool_score,
        "context_score": context_score,
        "fact_score": fact_score,
        "exact_score": exact_score,
        "format_score": format_score,
        "drift_penalty": drift_penalty,
        "total_score": total,
        "context_drift": context_score < 20,
        "called_functions": "|".join(sorted(called)),
        "missing_expectations": "|".join(misses),
    }


def _turn_result(
    run_id: str,
    model: ModelSpec,
    group: dict[str, Any],
    session_id: str,
    spec: TurnSpec,
    chat: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    answer = chat.get("answer", "")
    events = chat.get("events", [])
    score = _score_turn(spec, ground_truth, answer, events)
    tool_inputs = [event for event in events if event.get("tag") == "tool_input"]
    tool_outputs = [event for event in events if event.get("tag") == "tool_output"]
    thinking = [event for event in events if event.get("tag") == "thinking"]
    return {
        "run_id": run_id,
        "model_source": model.source,
        "model_name": model.model_name,
        "group_id": group["group_id"],
        "group_name": group["name"],
        "session_id": session_id,
        "conversation_id": spec.conversation_id,
        "turn_index": spec.turn_index,
        "turn": chat.get("turn", 0),
        "status": chat.get("status", ""),
        "user_message": spec.user_message,
        "expected_functions": "|".join(spec.expected_functions),
        "expected_markers": "|".join(spec.expected_markers),
        "tool_input_events": len(tool_inputs),
        "tool_output_events": len(tool_outputs),
        "thinking_events_client": len(thinking),
        "thinking_chars_client": sum(len(str(event.get("chunk", ""))) for event in thinking),
        "duration_ms_client": chat.get("duration_ms_client", 0),
        "answer_chars_client": len(answer),
        "estimated_prompt_tokens": _token_proxy(spec.user_message),
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
    _write_csv(out / "multi_turn_runs.csv", rows)
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    model_summary = df.groupby("model_name", dropna=False).agg(
        turns=("model_name", "count"),
        avg_total_score=("total_score", "mean"),
        pass_rate=("total_score", lambda s: float((pd.Series(s) >= 80).mean())),
        context_drift_rate=("context_drift", "mean"),
        avg_context_score=("context_score", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_tool_calls=("tool_input_events", "mean"),
        avg_thinking_chars=("thinking_chars_client", "mean"),
    ).reset_index().round(3)
    conversation_summary = df.groupby(["conversation_id", "model_name"], dropna=False).agg(
        turns=("turn_index", "count"),
        avg_total_score=("total_score", "mean"),
        min_total_score=("total_score", "min"),
        context_drift_rate=("context_drift", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
    ).reset_index().round(3)
    turn_summary = df.groupby("turn_index", dropna=False).agg(
        turns=("turn_index", "count"),
        avg_total_score=("total_score", "mean"),
        avg_context_score=("context_score", "mean"),
        context_drift_rate=("context_drift", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
    ).reset_index().round(3)
    model_summary.to_csv(out / "multi_turn_summary_by_model.csv", index=False, encoding="utf-8-sig")
    conversation_summary.to_csv(out / "multi_turn_summary_by_conversation.csv", index=False, encoding="utf-8-sig")
    turn_summary.to_csv(out / "multi_turn_summary_by_turn.csv", index=False, encoding="utf-8-sig")
    _plot_outputs(out, df, model_summary, conversation_summary, turn_summary)


def _plot_outputs(
    out: Path,
    runs: pd.DataFrame,
    model_summary: pd.DataFrame,
    conversation_summary: pd.DataFrame,
    turn_summary: pd.DataFrame,
) -> None:
    chart_dir = out / "charts"
    chart_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 21,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 14,
    })

    def barh(data: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str, filename: str, suffix: str = "") -> None:
        data = data.sort_values(value_col)
        fig, ax = plt.subplots(figsize=(11, max(5.5, 0.55 * len(data) + 2)))
        values = data[value_col].astype(float)
        bars = ax.barh(data[label_col].astype(str), values, color="#1f7a8c")
        high = max(100, float(values.max()) if len(values) else 100)
        ax.set_xlim(0, high * 1.15)
        ax.set_xlabel(xlabel)
        ax.set_title(title, weight="bold")
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(value + high * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.1f}{suffix}", va="center", weight="bold")
        fig.savefig(chart_dir / filename, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    barh(model_summary, "model_name", "avg_total_score", "Multi-turn Stability Score by Model", "Average score", "fig01_score_by_model.png")

    drift = model_summary.copy()
    drift["context_drift_pct"] = drift["context_drift_rate"].astype(float) * 100
    barh(drift, "model_name", "context_drift_pct", "Context Drift Rate by Model", "Drift rate (%)", "fig02_context_drift_by_model.png", "%")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for model_name, part in runs.groupby("model_name"):
        by_turn = part.groupby("turn_index")["total_score"].mean()
        ax.plot(by_turn.index, by_turn.values, marker="o", linewidth=2.5, label=model_name)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Average score")
    ax.set_title("Score Change Across Multi-turn Conversation", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(chart_dir / "fig03_score_by_turn.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pivot = conversation_summary.pivot(index="model_name", columns="conversation_id", values="avg_total_score")
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.65 * len(pivot) + 2)))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_title("Model by Conversation Stability Matrix", weight="bold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c).replace("_", "\n") for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=12, weight="bold", color="white" if value >= 70 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Score")
    fig.savefig(chart_dir / "fig04_model_conversation_matrix.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(model_summary["avg_duration_s"], model_summary["avg_total_score"], s=220, color="#d62828", marker="*", edgecolor="white", linewidth=1.2)
    for _, row in model_summary.iterrows():
        ax.text(row["avg_duration_s"] + 0.2, row["avg_total_score"] + 0.6, f"{row['model_name']}\n{row['avg_total_score']:.1f}, {row['avg_duration_s']:.1f}s", fontsize=11, weight="bold")
    ax.set_xlabel("Average response time (s)")
    ax.set_ylabel("Multi-turn stability score")
    ax.set_title("Multi-turn Quality-Efficiency Frontier", weight="bold")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    fig.savefig(chart_dir / "fig05_quality_efficiency_frontier.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _benchmark(base_url: str, user_id: str, group_id: str) -> dict[str, Any]:
    status, payload = _request_json(base_url, f"/metrics/benchmark?group_id={group_id}", user_id=user_id, timeout=120)
    return payload if status == 200 else {"error": payload, "status": status}


def run(
    base_url: str,
    user_id: str,
    output_dir: Path | None,
    timeout: int,
    model_scope: str,
    model_limit: int | None,
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = output_dir or OUTPUTS_DIR / "validation" / "multi_turn_agent" / run_id
    out.mkdir(parents=True, exist_ok=True)
    ground_truth = build_ground_truth()
    (out / "ground_truth.json").write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    models = _select_models(_parse_model_list(MODEL_LIST), model_scope)
    if model_limit is not None:
        models = models[:model_limit]
    group = _create_group(base_url, user_id, f"multi-turn-{run_id}", "Multi-turn context stability benchmark")
    rows: list[dict[str, Any]] = []
    for model in models:
        for conversation_id, turns in CONVERSATIONS.items():
            session_id = _create_session(base_url, user_id, group["group_id"], model, conversation_id)
            for spec in turns:
                prompt = spec.user_message if spec.turn_index > 1 else f"{SYSTEM_PROMPT}\n\n使用者問題：{spec.user_message}"
                try:
                    chat = _chat(base_url, session_id, prompt, user_id, timeout)
                    row = _turn_result(run_id, model, group, session_id, spec, chat, ground_truth)
                except Exception as exc:
                    row = {
                        "run_id": run_id,
                        "model_source": model.source,
                        "model_name": model.model_name,
                        "group_id": group["group_id"],
                        "group_name": group["name"],
                        "session_id": session_id,
                        "conversation_id": spec.conversation_id,
                        "turn_index": spec.turn_index,
                        "turn": 0,
                        "status": "error",
                        "user_message": spec.user_message,
                        "expected_functions": "|".join(spec.expected_functions),
                        "expected_markers": "|".join(spec.expected_markers),
                        "tool_input_events": 0,
                        "tool_output_events": 0,
                        "thinking_events_client": 0,
                        "thinking_chars_client": 0,
                        "duration_ms_client": 0,
                        "answer_chars_client": 0,
                        "estimated_prompt_tokens": _token_proxy(spec.user_message),
                        "estimated_answer_tokens_client": 0,
                        "actual_prompt_tokens": "",
                        "actual_completion_tokens": "",
                        "actual_total_tokens": "",
                        "token_data_quality": "estimated_from_text_length; actual provider usage not exposed by System benchmark API",
                        "error": str(exc),
                        "answer_preview": "",
                        "tool_score": 0,
                        "context_score": 0,
                        "fact_score": 0,
                        "exact_score": 0,
                        "format_score": 0,
                        "drift_penalty": 0,
                        "total_score": 0,
                        "context_drift": True,
                        "called_functions": "",
                        "missing_expectations": "|".join(spec.numeric_expectations + spec.exact_expectations),
                    }
                rows.append(row)
                _write_csv(out / "multi_turn_runs.csv", rows)
    benchmark = _benchmark(base_url, user_id, group["group_id"])
    _write_outputs(out, rows, ground_truth, benchmark)
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "user_id": user_id,
        "models": [model.model_name for model in models],
        "model_scope": model_scope,
        "conversation_count": len(CONVERSATIONS),
        "turns_per_conversation": {key: len(value) for key, value in CONVERSATIONS.items()},
        "rows": len(rows),
        "group_id": group["group_id"],
        "output_dir": str(out),
        "ground_truth_source": "Generated by direct calls to main.py analysis functions before multi-turn QA.",
        "actual_provider_token_usage_available": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-turn context stability benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--model-scope", choices=sorted(MODEL_SCOPES), default="cloud")
    parser.add_argument("--model-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(
        run(
            base_url=args.base_url.rstrip("/"),
            user_id=args.user_id,
            output_dir=args.output_dir,
            timeout=args.timeout,
            model_scope=args.model_scope,
            model_limit=args.model_limit,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
