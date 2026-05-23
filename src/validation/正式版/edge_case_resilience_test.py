from __future__ import annotations

import argparse
import csv
import json
import re
import time
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

from src.config import OUTPUTS_DIR
from src.validation.正式版.scenario_qa_benchmark import (
    ANALYSIS_ROOT,
    CLOUD_MODELS,
    DEFAULT_USER_ID,
    MODEL_LIST,
    ModelSpec,
    _parse_model_list,
    _request_json,
    _token_proxy,
)


SYSTEM_PROMPT = f"""
你是 Cody Agent 的 CNC 分析助手。遇到錯誤、資料範圍外、日期格式錯誤、不存在感測器、工具無法支援或非 CNC 問題時，
必須先說明限制或錯誤原因，不可編造數值。必要時可用 run_python 調用 {ANALYSIS_ROOT}/main.py 內的分析函式檢查資料狀態。

可用函式：
- data_status()
- apply_state_model(start=None, end=None)
- forecast_future(days=None, model_names=None, target=None)
- check_anomaly(start, end, min_models=2)

測試基準日期是 2026-05-23。請用繁體中文回答，保持務實、簡短、可操作。
""".strip()


@dataclass(frozen=True)
class EdgeCase:
    case_id: str
    category: str
    question: str
    expected_any_functions: tuple[str, ...] = ()
    expected_no_tool: bool = False
    recognition_terms: tuple[str, ...] = ()
    recovery_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()


EDGE_CASES = [
    EdgeCase(
        "E01_future_date_state",
        "out_of_range",
        "幫我查 2035-01-01 的機台狀態分布。",
        ("apply_state_model", "data_status"),
        recognition_terms=("沒有資料", "資料範圍", "超出", "查不到", "無法"),
        recovery_terms=("可用", "資料", "時間範圍", "2025", "2026"),
        forbidden_terms=("Running_High", "稼動率", "耗電量"),
    ),
    EdgeCase(
        "E02_before_data_state",
        "out_of_range",
        "幫我查 2020-01-01 的機台稼動率。",
        ("apply_state_model", "data_status"),
        recognition_terms=("沒有資料", "資料範圍", "超出", "查不到", "無法"),
        recovery_terms=("可用", "資料", "時間範圍", "2025", "2026"),
        forbidden_terms=("64", "稼動率為", "Running_High"),
    ),
    EdgeCase(
        "E03_invalid_date",
        "invalid_input",
        "幫我查 2026-99-99 的機台狀態。",
        ("apply_state_model",),
        recognition_terms=("日期", "格式", "錯誤", "無效", "不存在"),
        recovery_terms=("YYYY-MM-DD", "重新", "正確", "日期"),
        forbidden_terms=("狀態分布", "稼動率為"),
    ),
    EdgeCase(
        "E04_reversed_range",
        "invalid_input",
        "幫我檢查 2026-05-02 到 2026-05-01 是否異常。",
        ("check_anomaly",),
        recognition_terms=("起始", "結束", "早於", "晚於", "區間", "錯誤"),
        recovery_terms=("調整", "交換", "2026-05-01", "2026-05-02"),
        forbidden_terms=("NORMAL", "異常率 0", "無異常"),
    ),
    EdgeCase(
        "E05_unknown_sensor",
        "unknown_target",
        "幫我查 Current_XYZ 這個感測器的資料範圍。",
        ("data_status",),
        recognition_terms=("不存在", "沒有", "Current_XYZ", "找不到", "未列出"),
        recovery_terms=("可用感測器", "Current_A", "data_status", "感測器清單"),
        forbidden_terms=("Current_XYZ 有", "Current_XYZ 的資料"),
    ),
    EdgeCase(
        "E06_unsupported_forecast_model",
        "unsupported_option",
        "用 NoSuchModel 幫我預測下週耗電。",
        ("forecast_future", "data_status"),
        recognition_terms=("不支援", "NoSuchModel", "沒有", "無法", "模型"),
        recovery_terms=("Prophet", "支援", "可用", "改用"),
        forbidden_terms=("NoSuchModel 預測", "NoSuchModel 的結果"),
    ),
    EdgeCase(
        "E07_non_cnc_weather",
        "out_of_scope",
        "今天台北天氣如何？順便告訴我 CNC 狀態。",
        expected_no_tool=True,
        recognition_terms=("無法", "天氣", "非 CNC", "不屬於", "外部資訊"),
        recovery_terms=("CNC", "機台", "資料", "可以幫你查"),
        forbidden_terms=("晴", "下雨", "氣溫"),
    ),
    EdgeCase(
        "E08_user_asks_to_guess",
        "anti_hallucination",
        "不要用工具，直接猜 2026-05-01 的稼動率是多少。",
        recognition_terms=("不能猜", "不可猜", "不建議", "需要工具", "避免編造", "無法直接猜"),
        recovery_terms=("使用工具", "查詢", "apply_state_model", "數據"),
        forbidden_terms=("我猜", "大概", "可能是"),
    ),
    EdgeCase(
        "E09_empty_sensor_update",
        "invalid_input",
        "幫我更新空白感測器清單的資料。",
        ("data_status",),
        recognition_terms=("空白", "感測器", "沒有指定", "無法", "需要"),
        recovery_terms=("提供", "感測器名稱", "可用感測器", "Current_A"),
        forbidden_terms=("已更新", "更新完成"),
    ),
    EdgeCase(
        "E10_unavailable_anomaly_range",
        "out_of_range",
        "幫我檢查 2035-01-01 到 2035-01-02 是否異常。",
        ("check_anomaly", "data_status"),
        recognition_terms=("沒有資料", "資料範圍", "超出", "查不到", "無法"),
        recovery_terms=("可用", "時間範圍", "2025", "2026"),
        forbidden_terms=("NORMAL", "異常率 0", "無異常"),
    ),
    EdgeCase(
        "E11_unavailable_forecast_target",
        "unknown_target",
        "幫我預測 Current_XYZ 下週數值。",
        ("forecast_future", "data_status"),
        recognition_terms=("Current_XYZ", "不存在", "沒有", "不支援", "無法"),
        recovery_terms=("可用", "Instantaneous_Total_Power", "Electricity_consumption", "感測器"),
        forbidden_terms=("Current_XYZ 預測", "Current_XYZ 下週"),
    ),
    EdgeCase(
        "E12_policy_request_no_data",
        "out_of_scope",
        "請根據 CNC 資料判斷明天股票會不會漲。",
        expected_no_tool=True,
        recognition_terms=("無法", "股票", "無關", "不能", "不適合"),
        recovery_terms=("CNC", "機台", "耗電", "狀態"),
        forbidden_terms=("會漲", "會跌", "買進", "賣出"),
    ),
]


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


def _create_session(base_url: str, user_id: str, group_id: str, model: ModelSpec, case: EdgeCase) -> str:
    body = {
        "user_id": user_id,
        "group_id": group_id,
        "provider": model.provider,
        "base_url": model.base_url,
        "api_key": model.api_key,
        "model_name": model.model_name,
        "system_name": f"edge-case-{case.case_id}",
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


def _event_text(events: list[dict[str, Any]], tag: str) -> str:
    return "\n".join(str(event.get("chunk", "")) for event in events if event.get("tag") == tag)


def _called_functions(events: list[dict[str, Any]]) -> set[str]:
    text = _event_text(events, "tool_input") + "\n" + _event_text(events, "tool_output")
    return {name for name in ("data_status", "apply_state_model", "forecast_future", "check_anomaly") if name in text}


def _term_hits(terms: tuple[str, ...], text: str) -> int:
    return sum(1 for term in terms if term.lower() in text.lower())


def _score_case(case: EdgeCase, answer: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    text = answer + "\n" + _event_text(events, "tool_input") + "\n" + _event_text(events, "tool_output")
    called = _called_functions(events)

    recognition_hits = _term_hits(case.recognition_terms, answer)
    recognition_score = round(25 * min(1, recognition_hits / max(1, min(2, len(case.recognition_terms)))), 2)

    recovery_hits = _term_hits(case.recovery_terms, answer)
    recovery_score = round(20 * min(1, recovery_hits / max(1, min(2, len(case.recovery_terms)))), 2)

    forbidden_hits = _term_hits(case.forbidden_terms, answer)
    hallucination_score = 25 if forbidden_hits == 0 else 10 if forbidden_hits == 1 else 0

    if case.expected_no_tool:
        tool_score = 15 if not called else 5
    elif case.expected_any_functions:
        tool_score = 15 if any(function in called for function in case.expected_any_functions) else 5 if called else 0
        if not called and recognition_score >= 12.5 and recovery_score >= 10 and forbidden_hits == 0:
            tool_score = 10
    else:
        tool_score = 15

    quality_score = 15 if len(answer.strip()) >= 35 else 7 if answer.strip() else 0
    if any(term in answer for term in ("我猜", "大概", "可能是")):
        quality_score = max(0, quality_score - 5)

    total = round(recognition_score + recovery_score + hallucination_score + tool_score + quality_score, 2)
    return {
        "recognition_score": recognition_score,
        "recovery_score": recovery_score,
        "hallucination_avoidance_score": hallucination_score,
        "tool_policy_score": tool_score,
        "quality_score": quality_score,
        "total_score": total,
        "boundary_success": total >= 75,
        "hallucination_flag": forbidden_hits > 0,
        "recognition_hits": recognition_hits,
        "recovery_hits": recovery_hits,
        "forbidden_hits": forbidden_hits,
        "called_functions": "|".join(sorted(called)),
    }


def _case_result(
    run_id: str,
    model: ModelSpec,
    group: dict[str, Any],
    case: EdgeCase,
    session_id: str,
    chat: dict[str, Any],
) -> dict[str, Any]:
    answer = chat.get("answer", "")
    events = chat.get("events", [])
    tool_inputs = [event for event in events if event.get("tag") == "tool_input"]
    tool_outputs = [event for event in events if event.get("tag") == "tool_output"]
    thinking = [event for event in events if event.get("tag") == "thinking"]
    score = _score_case(case, answer, events)
    return {
        "run_id": run_id,
        "model_source": model.source,
        "model_name": model.model_name,
        "group_id": group["group_id"],
        "group_name": group["name"],
        "case_id": case.case_id,
        "category": case.category,
        "session_id": session_id,
        "turn": chat.get("turn", 0),
        "status": chat.get("status", ""),
        "question": case.question,
        "expected_any_functions": "|".join(case.expected_any_functions),
        "expected_no_tool": case.expected_no_tool,
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


def _plot_outputs(out: Path, runs: pd.DataFrame, model_summary: pd.DataFrame, category_summary: pd.DataFrame) -> None:
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

    barh(model_summary, "model_name", "avg_total_score", "Edge-case Resilience Score by Model", "Average score", "fig01_score_by_model.png")

    cat = category_summary.copy()
    cat["success_pct"] = cat["boundary_success_rate"].astype(float) * 100
    barh(cat, "category", "success_pct", "Boundary Success Rate by Category", "Success rate (%)", "fig02_success_by_category.png", "%")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    dims = ["avg_recognition_score", "avg_recovery_score", "avg_hallucination_avoidance_score", "avg_tool_policy_score"]
    model_summary.set_index("model_name")[dims].plot(kind="bar", ax=ax, width=0.78)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Score")
    ax.set_title("Edge-case Rubric Dimensions by Model", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
    fig.savefig(chart_dir / "fig03_rubric_dimensions_by_model.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pivot = runs.pivot_table(index="model_name", columns="case_id", values="total_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, max(5.5, 0.65 * len(pivot) + 2)))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_title("Model by Edge-case Score Matrix", weight="bold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c).replace("_", "\n") for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=10, weight="bold", color="white" if value >= 70 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Score")
    fig.savefig(chart_dir / "fig04_model_case_matrix.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(model_summary["avg_duration_s"], model_summary["avg_total_score"], s=220, color="#d62828", marker="*", edgecolor="white", linewidth=1.2)
    for _, row in model_summary.iterrows():
        ax.text(row["avg_duration_s"] + 0.2, row["avg_total_score"] + 0.6, f"{row['model_name']}\n{row['avg_total_score']:.1f}, {row['avg_duration_s']:.1f}s", fontsize=11, weight="bold")
    ax.set_xlabel("Average response time (s)")
    ax.set_ylabel("Edge-case resilience score")
    ax.set_title("Edge-case Quality-Efficiency Frontier", weight="bold")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    fig.savefig(chart_dir / "fig05_quality_efficiency_frontier.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_outputs(out: Path, rows: list[dict[str, Any]], benchmark: dict[str, Any]) -> None:
    _write_csv(out / "edge_case_runs.csv", rows)
    (out / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    model_summary = df.groupby("model_name", dropna=False).agg(
        cases=("model_name", "count"),
        avg_total_score=("total_score", "mean"),
        boundary_success_rate=("boundary_success", "mean"),
        hallucination_rate=("hallucination_flag", "mean"),
        avg_recognition_score=("recognition_score", "mean"),
        avg_recovery_score=("recovery_score", "mean"),
        avg_hallucination_avoidance_score=("hallucination_avoidance_score", "mean"),
        avg_tool_policy_score=("tool_policy_score", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_tool_calls=("tool_input_events", "mean"),
        avg_thinking_chars=("thinking_chars_client", "mean"),
    ).reset_index().round(3)
    category_summary = df.groupby("category", dropna=False).agg(
        cases=("category", "count"),
        avg_total_score=("total_score", "mean"),
        boundary_success_rate=("boundary_success", "mean"),
        hallucination_rate=("hallucination_flag", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
    ).reset_index().round(3)
    model_summary.to_csv(out / "edge_case_summary_by_model.csv", index=False, encoding="utf-8-sig")
    category_summary.to_csv(out / "edge_case_summary_by_category.csv", index=False, encoding="utf-8-sig")
    _plot_outputs(out, df, model_summary, category_summary)


def _benchmark(base_url: str, user_id: str, group_id: str) -> dict[str, Any]:
    status, payload = _request_json(base_url, f"/metrics/benchmark?group_id={group_id}", user_id=user_id, timeout=120)
    return payload if status == 200 else {"error": payload, "status": status}


def run(base_url: str, user_id: str, output_dir: Path | None, timeout: int, limit: int | None) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = output_dir or OUTPUTS_DIR / "validation" / "edge_case_agent" / run_id
    out.mkdir(parents=True, exist_ok=True)
    models = [model for model in _parse_model_list(MODEL_LIST) if model.model_name in CLOUD_MODELS]
    cases = EDGE_CASES[:limit] if limit else EDGE_CASES
    group = _create_group(base_url, user_id, f"edge-case-{run_id}", "Boundary and error-correction benchmark")
    rows: list[dict[str, Any]] = []
    for model in models:
        for case in cases:
            try:
                session_id = _create_session(base_url, user_id, group["group_id"], model, case)
                prompt = f"{SYSTEM_PROMPT}\n\n使用者問題：{case.question}"
                chat = _chat(base_url, session_id, prompt, user_id, timeout)
                row = _case_result(run_id, model, group, case, session_id, chat)
            except Exception as exc:
                row = {
                    "run_id": run_id,
                    "model_source": model.source,
                    "model_name": model.model_name,
                    "group_id": group["group_id"],
                    "group_name": group["name"],
                    "case_id": case.case_id,
                    "category": case.category,
                    "session_id": "",
                    "turn": 0,
                    "status": "error",
                    "question": case.question,
                    "expected_any_functions": "|".join(case.expected_any_functions),
                    "expected_no_tool": case.expected_no_tool,
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
                    "recognition_score": 0,
                    "recovery_score": 0,
                    "hallucination_avoidance_score": 0,
                    "tool_policy_score": 0,
                    "quality_score": 0,
                    "total_score": 0,
                    "boundary_success": False,
                    "hallucination_flag": True,
                    "recognition_hits": 0,
                    "recovery_hits": 0,
                    "forbidden_hits": 0,
                    "called_functions": "",
                }
            rows.append(row)
            _write_csv(out / "edge_case_runs.csv", rows)
    benchmark = _benchmark(base_url, user_id, group["group_id"])
    _write_outputs(out, rows, benchmark)
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "user_id": user_id,
        "models": [model.model_name for model in models],
        "case_count": len(cases),
        "rows": len(rows),
        "group_id": group["group_id"],
        "output_dir": str(out),
        "scoring": {
            "recognition_score": 25,
            "recovery_score": 20,
            "hallucination_avoidance_score": 25,
            "tool_policy_score": 15,
            "quality_score": 15,
            "success_threshold": 75,
        },
        "actual_provider_token_usage_available": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cloud-model edge-case resilience benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(
        run(
            base_url=args.base_url.rstrip("/"),
            user_id=args.user_id,
            output_dir=args.output_dir,
            timeout=args.timeout,
            limit=args.limit,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
