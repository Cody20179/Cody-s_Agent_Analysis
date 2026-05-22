from __future__ import annotations

import argparse
import csv
import json
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR


THEME = {
    "domain": "#2563eb",
    "out_of_scope": "#64748b",
    "unplanned": "#9333ea",
    "error_recovery": "#dc2626",
    "multi_step": "#16a34a",
    "api_error": "#f59e0b",
}


@dataclass(frozen=True)
class AgentCase:
    case_id: str
    category: str
    prompt: str
    expected_min_tools: int = 0
    expected_max_tools: int | None = None
    repeat: int = 1


AGENT_CASES = [
    AgentCase(
        "domain_today_status",
        "domain",
        "幫我查詢今日機台狀態，列出主要狀態與可用圖表。",
        expected_min_tools=1,
    ),
    AgentCase(
        "domain_period_status",
        "domain",
        "幫我查詢 2026-05-01 00:00:00 到 2026-05-02 00:00:00 的機台狀態，整理狀態分布。",
        expected_min_tools=1,
    ),
    AgentCase(
        "out_of_scope_weather",
        "out_of_scope",
        "請告訴我今天台北天氣如何；如果你沒有外部天氣工具，請直接說明限制，不要硬猜。",
        expected_max_tools=0,
    ),
    AgentCase(
        "out_of_scope_translation",
        "out_of_scope",
        "請把這句翻成英文：機台今日狀態穩定，無明顯異常。",
        expected_max_tools=0,
    ),
    AgentCase(
        "unplanned_sensor_compare",
        "unplanned",
        "請比較 Current_A、Current_B、Current_C 哪個資料量最大，並說明可用資料時間範圍；這不是預設情境時也請盡量用可用工具回答。",
        expected_min_tools=1,
        repeat=2,
    ),
    AgentCase(
        "unplanned_operation_advice",
        "unplanned",
        "如果我要用現有 CNC 資料判斷哪個時段最適合安排保養，請你用可用分析結果推論，工具不夠時要說明缺口。",
        expected_min_tools=1,
    ),
    AgentCase(
        "error_invalid_date",
        "error_recovery",
        "請檢查 2026-99-99 00:00:00 到 2026-05-02 00:00:00 是否異常；如果日期格式錯誤，請修正或要求我更正。",
        repeat=2,
    ),
    AgentCase(
        "error_no_data_range",
        "error_recovery",
        "請檢查 2035-01-01 00:00:00 到 2035-01-02 00:00:00 是否異常；如果沒有資料，請說明怎麼處理。",
        expected_min_tools=1,
    ),
    AgentCase(
        "multi_step_status_forecast",
        "multi_step",
        "先查資料狀態，再查今日機台狀態，再預測下週耗電，最後列出每一步使用的工具與結果。",
        expected_min_tools=3,
        repeat=3,
    ),
    AgentCase(
        "multi_step_guarded_forecast",
        "multi_step",
        "先確認模型是否可用；如果不需要請不要重新訓練，再預測下週耗電並說明是否使用既有模型。",
        expected_min_tools=1,
        repeat=2,
    ),
    AgentCase(
        "multi_step_anomaly_forecast",
        "multi_step",
        "檢查 2026-05-01 00:00:00 到 2026-05-02 00:00:00 是否異常，接著預測未來 7 天耗電，最後比較兩者對維護決策的用途。",
        expected_min_tools=2,
    ),
]


API_CASES = [
    {
        "case_id": "api_empty_message",
        "category": "api_error",
        "method": "POST",
        "path": "/chat",
        "body": {"user_id": "Cody", "message": ""},
        "expected_status": 400,
    },
    {
        "case_id": "api_missing_group",
        "category": "api_error",
        "method": "POST",
        "path": "/sessions/simple",
        "body": {"user_id": "Cody", "group_id": "missing-group"},
        "expected_status": 404,
    },
    {
        "case_id": "api_missing_session_cancel",
        "category": "api_error",
        "method": "POST",
        "path": "/sessions/not-a-session/cancel",
        "body": {},
        "expected_status": 404,
    },
]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 260,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
            "font.family": "Arial",
            "font.size": 17,
            "axes.titlesize": 24,
            "axes.labelsize": 19,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 1.1,
            "grid.color": "#cbd5e1",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
        }
    )


def _request_json(base_url: str, path: str, method: str = "GET", body: dict | None = None, user_id: str = "Cody") -> tuple[int, Any]:
    data = None
    headers = {"X-User-ID": user_id}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"detail": raw}
        return exc.code, payload


def _chat(base_url: str, user_id: str, group_id: str, case: AgentCase, iteration: int, timeout: int) -> dict[str, Any]:
    body = {
        "user_id": user_id,
        "group_id": group_id,
        "message": case.prompt,
        "use_memory": False,
    }
    req = urllib.request.Request(
        base_url + "/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-User-ID": user_id},
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    session_id = ""
    turn = 0
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                events.append(payload)
                tag = payload.get("tag", "")
                if tag == "session":
                    session_id = payload.get("session_id", "")
                elif tag == "result":
                    answer_parts.append(payload.get("chunk", ""))
                elif tag == "done":
                    session_id = payload.get("session_id", session_id)
                    turn = int(payload.get("turn") or 0)
                    break
                elif tag == "error":
                    error = payload.get("chunk", "")
    except Exception as exc:
        error = str(exc)
    duration_ms = int((time.perf_counter() - started) * 1000)
    tool_inputs = [event for event in events if event.get("tag") == "tool_input"]
    tool_outputs = [event for event in events if event.get("tag") == "tool_output"]
    thinking = [event for event in events if event.get("tag") == "thinking"]
    answer = "".join(answer_parts)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "iteration": iteration,
        "session_id": session_id,
        "turn": turn,
        "status": "error" if error else "success",
        "error": error,
        "duration_ms_client": duration_ms,
        "tool_input_events": len(tool_inputs),
        "tool_output_events": len(tool_outputs),
        "thinking_events_client": len(thinking),
        "thinking_chars_client": sum(len(str(event.get("chunk", ""))) for event in thinking),
        "answer_chars_client": len(answer),
        "estimated_text_tokens": int((len(case.prompt) + len(answer)) / 4),
        "expected_min_tools": case.expected_min_tools,
        "expected_max_tools": case.expected_max_tools,
        "expectation_passed": _expectation_passed(len(tool_inputs), case.expected_min_tools, case.expected_max_tools),
        "prompt": case.prompt,
        "answer_preview": answer[:500],
        "events": events,
    }


def _expectation_passed(tool_count: int, minimum: int = 0, maximum: int | None = None) -> bool:
    if tool_count < minimum:
        return False
    if maximum is not None and tool_count > maximum:
        return False
    return True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    excluded = {"events"}
    fields = [key for key in rows[0].keys() if key not in excluded]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _benchmark(base_url: str, user_id: str, group_id: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"group_id": group_id, "limit": 50000})
    status, payload = _request_json(base_url, f"/evaluation/benchmark?{query}", user_id=user_id)
    if status != 200:
        raise RuntimeError(f"benchmark failed: {status} {payload}")
    return payload


def _clean_text(value: str, width: int = 26) -> str:
    return textwrap.fill(str(value).replace("_", " "), width=width)


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _barh(ax: plt.Axes, labels: list[str], values: list[float], colors: list[str], title: str, xlabel: str, path: Path, suffix: str = "", precision: int = 1) -> None:
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y, [_clean_text(label) for label in labels])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    high = max(values) if values else 1
    ax.set_xlim(0, max(1, high) * 1.25)
    for bar in bars:
        value = bar.get_width()
        text = f"{value:.{precision}f}{suffix}"
        ax.text(
            min(value + max(high, 1) * 0.02, ax.get_xlim()[1] * 0.98),
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            fontsize=15,
            fontweight="bold",
            clip_on=True,
        )
    _finish(ax.figure, path)


def _plot_counts_by_category(agent_df: pd.DataFrame, out: Path) -> None:
    data = agent_df.groupby("category").agg(
        passed=("expectation_passed", lambda s: int(s.sum())),
        failed=("expectation_passed", lambda s: int((~s).sum())),
    ).reset_index()
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.barh(y, data["passed"], color="#16a34a", label="Expectation passed")
    ax.barh(y, data["failed"], left=data["passed"], color="#dc2626", label="Expectation failed")
    ax.set_yticks(y, [_clean_text(label, 22) for label in data["category"]])
    ax.set_xlabel("Case count")
    ax.set_title("Expectation Result Counts")
    ax.grid(axis="x")
    ax.legend(loc="lower right", ncols=2, frameon=False)
    ax.set_xlim(0, max(1, int((data["passed"] + data["failed"]).max())) * 1.35)
    _finish(fig, out / "expectation_result_counts.png")


def _plot_agent_metrics(agent_df: pd.DataFrame, out: Path) -> None:
    grouped = agent_df.groupby(["case_id", "category"], as_index=False).agg(
        duration_s=("duration_ms_client", lambda s: float(s.mean()) / 1000),
        tool_calls=("tool_input_events", "mean"),
        thinking_events=("thinking_events_client", "mean"),
        thinking_chars=("thinking_chars_client", "mean"),
        answer_chars=("answer_chars_client", "mean"),
        estimated_text_tokens=("estimated_text_tokens", "mean"),
    )
    grouped = grouped.sort_values("duration_s")
    labels = grouped["case_id"].tolist()
    colors = [THEME.get(category, "#64748b") for category in grouped["category"]]

    fig, ax = plt.subplots(figsize=(15.5, max(8, len(grouped) * 0.62 + 2.4)))
    _barh(ax, labels, grouped["duration_s"].tolist(), colors, "Agent Runtime by Case", "Average runtime (seconds)", out / "agent_runtime_by_case.png", "s", 1)

    fig, ax = plt.subplots(figsize=(15.5, max(8, len(grouped) * 0.62 + 2.4)))
    _barh(ax, labels, grouped["tool_calls"].tolist(), colors, "Tool Use by Case", "Average tool call count", out / "tool_calls_by_case.png", "", 1)

    fig, ax = plt.subplots(figsize=(15.5, max(8, len(grouped) * 0.62 + 2.4)))
    _barh(ax, labels, grouped["thinking_events"].tolist(), colors, "Thinking Event Count by Case", "Average thinking event count", out / "thinking_events_by_case.png", "", 1)

    fig, ax = plt.subplots(figsize=(15.5, max(8, len(grouped) * 0.62 + 2.4)))
    _barh(ax, labels, grouped["thinking_chars"].tolist(), colors, "Thinking Length by Case", "Average thinking characters", out / "thinking_chars_by_case.png", "", 0)

    fig, ax = plt.subplots(figsize=(15.5, max(8, len(grouped) * 0.62 + 2.4)))
    _barh(ax, labels, grouped["estimated_text_tokens"].tolist(), colors, "Estimated Text Token Proxy by Case", "Estimated tokens from prompt + answer text", out / "estimated_token_proxy_by_case.png", "", 0)


def _plot_benchmark_metrics(turns: pd.DataFrame, tools: pd.DataFrame, out: Path) -> None:
    if turns.empty:
        return
    for col in ["duration_ms", "loop_count", "tool_call_count", "thinking_event_count", "thinking_char_count", "answer_char_count", "first_response_ms"]:
        if col in turns:
            turns[col] = pd.to_numeric(turns[col], errors="coerce").fillna(0)

    labels = [
        str(case_id) if str(case_id) else str(message)[:60]
        for case_id, message in zip(turns.get("case_id", ""), turns["user_message"].astype(str))
    ]
    colors = [THEME.get(category, "#64748b") for category in turns.get("case_category", pd.Series(["domain"] * len(turns)))]

    metrics = [
        ("loop_count", "Loop Count by Turn", "Loop count", "loop_count_by_turn.png"),
        ("tool_call_count", "Tool Call Count by Turn", "Tool calls", "tool_calls_by_turn.png"),
        ("thinking_event_count", "Thinking Count by Turn", "Thinking events", "thinking_count_by_turn.png"),
        ("thinking_char_count", "Thinking Length by Turn", "Thinking characters", "thinking_length_by_turn.png"),
        ("answer_char_count", "Answer Length by Turn", "Answer characters", "answer_length_by_turn.png"),
        ("first_response_ms", "First Response Time by Turn", "Milliseconds", "first_response_time_by_turn.png"),
    ]
    for column, title, xlabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(15.5, max(8, len(turns) * 0.58 + 2.4)))
        _barh(ax, labels, turns[column].tolist(), colors, title, xlabel, out / filename, "", 0)

    if not tools.empty:
        tools["duration_ms"] = pd.to_numeric(tools["duration_ms"], errors="coerce").fillna(0)
        tool_summary = tools.groupby("tool_name", as_index=False).agg(
            calls=("tool_name", "size"),
            avg_duration_ms=("duration_ms", "mean"),
        ).sort_values("calls")
        fig, ax = plt.subplots(figsize=(13.8, max(7.2, len(tool_summary) * 0.7 + 2)))
        _barh(ax, tool_summary["tool_name"].tolist(), tool_summary["calls"].tolist(), ["#2563eb"] * len(tool_summary), "Tool Call Frequency", "Call count", out / "tool_frequency.png", "", 0)

        tool_summary = tool_summary.sort_values("avg_duration_ms")
        fig, ax = plt.subplots(figsize=(13.8, max(7.2, len(tool_summary) * 0.7 + 2)))
        _barh(ax, tool_summary["tool_name"].tolist(), (tool_summary["avg_duration_ms"] / 1000).tolist(), ["#9333ea"] * len(tool_summary), "Average Tool Runtime", "Seconds", out / "tool_runtime_by_tool.png", "s", 1)


def _write_report(out: Path, raw: dict[str, Any], agent_df: pd.DataFrame, turns: pd.DataFrame, tools: pd.DataFrame) -> None:
    lines = [
        "# Cody Agent System Test Report",
        "",
        f"- Generated: {raw['generated_at']}",
        f"- API base: `{raw['api_base_url']}`",
        f"- User: `{raw['user_id']}`",
        f"- Group: `{raw['group_name']}` / `{raw['group_id']}`",
        f"- Agent cases: {len(agent_df)}",
        f"- API error cases: {len(raw['api_cases'])}",
        "",
        "## Important Limitations",
        "",
        "- Runtime success only means the API turn completed; semantic correctness still needs human review.",
        "- True provider token usage is not currently exposed in System metrics. `estimated_token_proxy_by_case.png` uses text length as a relative proxy, not billing-grade tokens.",
        "- Thinking metrics are recorded only when the model stream emits `thinking` events. Zero means not observed, not necessarily no reasoning.",
        "",
        "## Figures",
        "",
        "- `expectation_result_counts.png`",
        "- `agent_runtime_by_case.png`",
        "- `tool_calls_by_case.png`",
        "- `loop_count_by_turn.png`",
        "- `thinking_count_by_turn.png`",
        "- `thinking_length_by_turn.png`",
        "- `estimated_token_proxy_by_case.png`",
        "- `first_response_time_by_turn.png`",
        "- `tool_frequency.png`",
        "- `tool_runtime_by_tool.png`",
        "",
        "## Case Summary",
        "",
        "| Case | Category | Iteration | Status | Expectation | Tools | Runtime |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for row in agent_df.itertuples(index=False):
        lines.append(
            f"| `{row.case_id}` | {row.category} | {row.iteration} | {row.status} | "
            f"{'pass' if row.expectation_passed else 'fail'} | {int(row.tool_input_events)} | {row.duration_ms_client / 1000:.1f}s |"
        )
    lines.extend(["", "## API Error Handling", "", "| Case | Status | Expected | Passed |", "|---|---:|---:|---|"])
    for row in raw["api_cases"]:
        lines.append(
            f"| `{row['case_id']}` | {row['status_code']} | {row['expected_status']} | "
            f"{'pass' if row['passed'] else 'fail'} |"
        )
    lines.extend(["", "## Benchmark Summary", ""])
    overall = raw.get("benchmark", {}).get("overall", {})
    for key in ["turns", "success_rate", "avg_duration_ms", "total_tool_calls", "tool_success_rate", "avg_loop_count", "avg_thinking_chars", "avg_answer_chars"]:
        lines.append(f"- {key}: {overall.get(key, '')}")
    (out / "system_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_from_prompt(prompt: str) -> AgentCase | None:
    for case in AGENT_CASES:
        if case.prompt == prompt:
            return case
    return None


def _agent_rows_from_turns(turns: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    if turns.empty:
        return rows
    ordered = turns.sort_values("created_at")
    for turn in ordered.to_dict(orient="records"):
        prompt = str(turn.get("user_message") or "")
        case = _case_from_prompt(prompt)
        if not case:
            continue
        counters[case.case_id] = counters.get(case.case_id, 0) + 1
        tool_count = int(turn.get("tool_call_count") or 0)
        answer_chars = int(turn.get("answer_char_count") or len(str(turn.get("assistant_answer") or "")))
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "iteration": counters[case.case_id],
                "session_id": turn.get("session_id", ""),
                "turn": int(turn.get("turn") or 0),
                "status": turn.get("status", ""),
                "error": turn.get("error_message", ""),
                "duration_ms_client": int(turn.get("duration_ms") or 0),
                "tool_input_events": tool_count,
                "tool_output_events": int(turn.get("tool_output_count") or 0),
                "thinking_events_client": int(turn.get("thinking_event_count") or 0),
                "thinking_chars_client": int(turn.get("thinking_char_count") or 0),
                "answer_chars_client": answer_chars,
                "estimated_text_tokens": int((len(prompt) + answer_chars) / 4),
                "expected_min_tools": case.expected_min_tools,
                "expected_max_tools": case.expected_max_tools,
                "expectation_passed": _expectation_passed(tool_count, case.expected_min_tools, case.expected_max_tools),
                "prompt": prompt,
                "answer_preview": str(turn.get("assistant_answer") or "")[:500],
            }
        )
    return rows


def _api_error_cases(base_url: str, user_id: str) -> list[dict[str, Any]]:
    rows = []
    for case in API_CASES:
        status_code, payload = _request_json(
            base_url,
            case["path"],
            method=case["method"],
            body=case["body"],
            user_id=user_id,
        )
        rows.append(
            {
                **case,
                "status_code": status_code,
                "passed": status_code == case["expected_status"],
                "response": payload,
            }
        )
    return rows


def report_existing_group(base_url: str, user_id: str, group_id: str, output_dir: Path | None = None) -> dict[str, Any]:
    _style()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir or OUTPUTS_DIR / "validation" / "system" / f"{run_id}_{group_id}"
    out.mkdir(parents=True, exist_ok=True)
    benchmark = _benchmark(base_url, user_id, group_id)
    turns = pd.DataFrame(benchmark.get("turns", []))
    tools = pd.DataFrame(benchmark.get("tools", []))
    agent_rows = _agent_rows_from_turns(turns)
    api_rows = _api_error_cases(base_url, user_id)

    raw = {
        "generated_at": datetime.now().isoformat(),
        "api_base_url": base_url,
        "user_id": user_id,
        "group_name": "existing-group",
        "group_id": group_id,
        "agent_cases": agent_rows,
        "api_cases": api_rows,
        "benchmark": benchmark,
    }
    (out / "system_test_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out / "system_test_agent_cases.csv", agent_rows)
    _write_csv(out / "system_test_api_cases.csv", api_rows)

    agent_df = pd.DataFrame(agent_rows)
    if not turns.empty:
        turns["case_id"] = ""
        turns["case_category"] = ""
        for idx, turn in turns.iterrows():
            case = _case_from_prompt(str(turn.get("user_message") or ""))
            if case:
                turns.at[idx, "case_id"] = case.case_id
                turns.at[idx, "case_category"] = case.category
        turns.to_csv(out / "system_test_turns.csv", index=False)
    if not tools.empty:
        tools.to_csv(out / "system_test_tools.csv", index=False)

    if not agent_df.empty:
        _plot_counts_by_category(agent_df, out)
        _plot_agent_metrics(agent_df, out)
    _plot_benchmark_metrics(turns, tools, out)
    _write_report(out, raw, agent_df, turns, tools)
    return {"status": "ok", "output_dir": str(out), "group_id": group_id, "agent_case_count": len(agent_rows), "api_case_count": len(api_rows)}


def run_system_test(base_url: str, user_id: str, output_dir: Path | None, timeout: int) -> dict[str, Any]:
    _style()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir or OUTPUTS_DIR / "validation" / "system" / run_id
    out.mkdir(parents=True, exist_ok=True)

    group_name = f"system-validation-{run_id}"
    status, group_payload = _request_json(
        base_url,
        "/sessions/groups",
        method="POST",
        body={"name": group_name, "description": "Automated system validation"},
        user_id=user_id,
    )
    if status != 201:
        raise RuntimeError(f"cannot create test group: {status} {group_payload}")
    group_id = group_payload["group"]["group_id"]

    agent_rows: list[dict[str, Any]] = []
    for case in AGENT_CASES:
        for iteration in range(1, case.repeat + 1):
            agent_rows.append(_chat(base_url, user_id, group_id, case, iteration, timeout))

    api_rows = _api_error_cases(base_url, user_id)

    benchmark = _benchmark(base_url, user_id, group_id)
    raw = {
        "generated_at": datetime.now().isoformat(),
        "api_base_url": base_url,
        "user_id": user_id,
        "group_name": group_name,
        "group_id": group_id,
        "agent_cases": agent_rows,
        "api_cases": api_rows,
        "benchmark": benchmark,
    }
    (out / "system_test_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out / "system_test_agent_cases.csv", agent_rows)
    _write_csv(out / "system_test_api_cases.csv", api_rows)

    agent_df = pd.DataFrame([{k: v for k, v in row.items() if k != "events"} for row in agent_rows])
    turns = pd.DataFrame(benchmark.get("turns", []))
    tools = pd.DataFrame(benchmark.get("tools", []))

    if not turns.empty:
        turns["case_id"] = ""
        turns["case_category"] = ""
        for idx, turn in turns.iterrows():
            match = agent_df[agent_df["session_id"] == turn.get("session_id")]
            if not match.empty:
                turns.at[idx, "case_id"] = match.iloc[0]["case_id"]
                turns.at[idx, "case_category"] = match.iloc[0]["category"]
        turns.to_csv(out / "system_test_turns.csv", index=False)
    if not tools.empty:
        tools.to_csv(out / "system_test_tools.csv", index=False)

    _plot_counts_by_category(agent_df, out)
    _plot_agent_metrics(agent_df, out)
    _plot_benchmark_metrics(turns, tools, out)
    _write_report(out, raw, agent_df, turns, tools)
    return {"status": "ok", "output_dir": str(out), "group_id": group_id, "agent_case_count": len(agent_rows), "api_case_count": len(api_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end Cody Agent System validation through the System API.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default="Cody")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--group-id", help="Generate a report from an existing validation group without rerunning agent prompts.")
    args = parser.parse_args()
    base_url = args.api_base_url.rstrip("/")
    if args.group_id:
        result = report_existing_group(base_url, args.user_id, args.group_id, args.output_dir)
    else:
        result = run_system_test(base_url, args.user_id, args.output_dir, args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
