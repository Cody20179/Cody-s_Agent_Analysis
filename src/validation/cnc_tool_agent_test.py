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

from src.config import OUTPUTS_DIR


ANALYSIS_ROOT = "/Users/cody20179/Desktop/Code/Git_My_Project/Cody-s_Agent_Analysis"
MODEL_LIST = Path("模型列表.md")
DEFAULT_USER_ID = "Cody"
MODEL_ALIASES = {"gml-5.1": "glm-5.1"}
EMBEDDING_MODELS = {"qwen3-embedding:4b", "qwen3-embedding:8b"}
CLOUD_MODELS = {"glm-5.1", "kimi-k2.6", "deepseek-v4-flash"}
LIGHT_LOCAL_MODELS = {
    "qwen3.5:9b",
    "nemotron-cascade-2:latest",
    "gemma4:latest",
    "nemotron-3-nano:latest",
    "qwen3-vl:8b",
    "gpt-oss:20b",
    "qwen3:8b",
}
SUITES = {"light", "full"}


@dataclass(frozen=True)
class ModelSpec:
    source: str
    provider: str
    base_url: str
    api_key: str
    model_name: str


@dataclass(frozen=True)
class CncToolCase:
    case_id: str
    cnc_tool_name: str
    layer: str
    prompt: str
    expected_system_tool: str = "run_python"


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


def _analysis_prompt(tool_name: str, code: str, summary_request: str) -> str:
    return (
        f"請只用一次 run_python 執行 Cody-s_Agent_Analysis/main.py 的 {tool_name}。"
        "run_python 的 code 請使用下列程式碼，不要改用其他函式；timeout 設為 300 秒。\n\n"
        "```python\n"
        f"{code.strip()}\n"
        "```\n\n"
        f"執行後請用中文簡短回覆：{summary_request}"
    )


CNC_TOOL_CASES = [
    CncToolCase(
        "data_status",
        "data_status",
        "data",
        _analysis_prompt(
            "data_status()",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import data_status
print(json.dumps(data_status(), ensure_ascii=False, indent=2, default=str))
""",
            "資料覆蓋範圍、感測器數量與是否成功。",
        ),
    ),
    CncToolCase(
        "update_data",
        "update_data",
        "data",
        _analysis_prompt(
            "update_data(sensors=['Current_A'], verbose=False)",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import update_data
result = update_data(sensors=["Current_A"], verbose=False)
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "更新是否成功、更新感測器與輸出資料夾。",
        ),
    ),
    CncToolCase(
        "run_state_analysis",
        "run_state_analysis",
        "training_application",
        _analysis_prompt(
            "run_state_analysis(start='2026-05-01 00:00:00', end='2026-05-02 00:00:00')",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import run_state_analysis
result = run_state_analysis(start="2026-05-01 00:00:00", end="2026-05-02 00:00:00")
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "GMM 狀態分析是否成功、資料筆數與狀態摘要。",
        ),
    ),
    CncToolCase(
        "train_forecast",
        "train_forecast",
        "training",
        _analysis_prompt(
            "train_forecast(target='dy', models=['BaselineLastWeek'], months_back=1)",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import train_forecast
result = train_forecast(target="dy", models=["BaselineLastWeek"], months_back=1)
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "訓練是否成功、使用模型與輸出檔案。",
        ),
    ),
    CncToolCase(
        "forecast_future",
        "forecast_future",
        "application",
        _analysis_prompt(
            "forecast_future(days=[7], model_names=['Prophet'])",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import forecast_future
result = forecast_future(days=[7], model_names=["Prophet"])
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "7 天預測是否成功、預測摘要與輸出檔案。",
        ),
    ),
    CncToolCase(
        "train_anomaly_detection",
        "train_anomaly_detection",
        "training",
        _analysis_prompt(
            "train_anomaly_detection()",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import train_anomaly_detection
result = train_anomaly_detection()
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "異常偵測訓練是否成功、模型數與輸出檔案。",
        ),
    ),
    CncToolCase(
        "check_anomaly",
        "check_anomaly",
        "application",
        _analysis_prompt(
            "check_anomaly('2026-05-01 00:00:00', '2026-05-02 00:00:00')",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import check_anomaly
result = check_anomaly("2026-05-01 00:00:00", "2026-05-02 00:00:00")
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "是否異常、票數或 verdict、輸出檔案與是否成功。",
        ),
    ),
    CncToolCase(
        "run_all",
        "run_all",
        "workflow",
        _analysis_prompt(
            "run_all()",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import run_all
result = run_all()
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "完整管線各步驟是否成功，若有錯誤請列出步驟。",
        ),
    ),
]


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
    return {
        "turn": turn,
        "status": "error" if error else "success",
        "error": error,
        "duration_ms_client": int((time.perf_counter() - started) * 1000),
        "answer": "".join(answer_parts),
        "events": events,
    }


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


def _create_session(base_url: str, user_id: str, group_id: str, model: ModelSpec, case: CncToolCase) -> str:
    body = {
        "user_id": user_id,
        "group_id": group_id,
        "provider": model.provider,
        "base_url": model.base_url,
        "api_key": model.api_key,
        "model_name": model.model_name,
        "system_name": f"cnc-tool-{case.case_id}",
        "tool_names": [case.expected_system_tool],
        "mcp_names": [],
    }
    status, payload = _request_json(base_url, "/sessions", method="POST", body=body, user_id=user_id, timeout=120)
    if status not in {200, 201}:
        raise RuntimeError(f"cannot create session for {model.model_name}/{case.case_id}: {status} {payload}")
    return payload["session_id"]


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


def _output_text(events: list[dict[str, Any]]) -> str:
    return "\n".join(str(event.get("chunk", "")) for event in events if event.get("tag") == "tool_output")


def _tool_names(events: list[dict[str, Any]]) -> list[str]:
    names = []
    for event in events:
        if event.get("tag") != "tool_input":
            continue
        first = str(event.get("chunk", "")).splitlines()[0]
        names.append(first.replace("Tool:", "").strip())
    return names


def _tool_output_status_ok(text: str) -> bool:
    if not text:
        return False
    if '"status": "ok"' in text or "'status': 'ok'" in text:
        return True
    match = re.search(r"Output:\s*(\{.*\})", text, re.S)
    if not match:
        return False
    try:
        return json.loads(match.group(1)).get("status") == "ok"
    except Exception:
        return False


def _tool_output_error(text: str) -> bool:
    markers = ("Traceback", "[exit code", "ModuleNotFoundError", "Error:", '"status": "error"', "'status': 'error'")
    return any(marker in text for marker in markers)


def _token_proxy(text: object) -> int:
    if text is None:
        return 0
    s = str(text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    non_cjk = len(s) - cjk
    return int(math.ceil(cjk * 1.15 + non_cjk / 4.0))


def _case_result(
    run_id: str,
    model: ModelSpec,
    group: dict[str, Any],
    case: CncToolCase,
    repeat_index: int,
    session_id: str,
    chat: dict[str, Any],
) -> dict[str, Any]:
    events = chat["events"]
    names = _tool_names(events)
    output_text = _output_text(events)
    expected_called = case.expected_system_tool in names
    tool_output_error = _tool_output_error(output_text)
    output_status_ok = _tool_output_status_ok(output_text)
    functional_success = chat.get("status") == "success" and expected_called and output_status_ok and not tool_output_error
    answer = chat.get("answer", "")
    return {
        "run_id": run_id,
        "model_source": model.source,
        "model_deployment": "Cloud" if model.model_name in CLOUD_MODELS else "Local",
        "model_name": model.model_name,
        "model_label": f"{model.model_name} (Cloud)" if model.model_name in CLOUD_MODELS else model.model_name,
        "group_id": group["group_id"],
        "group_name": group["name"],
        "case_id": case.case_id,
        "cnc_tool_name": case.cnc_tool_name,
        "cnc_tool_layer": case.layer,
        "repeat_index": repeat_index,
        "session_id": session_id,
        "turn": chat.get("turn", 0),
        "status": chat.get("status", ""),
        "expected_system_tool": case.expected_system_tool,
        "system_tool_called": "|".join(names),
        "expected_system_tool_called": expected_called,
        "tool_output_status_ok": output_status_ok,
        "tool_output_error_detected": tool_output_error,
        "functional_success": functional_success,
        "tool_input_events": len([event for event in events if event.get("tag") == "tool_input"]),
        "tool_output_events": len([event for event in events if event.get("tag") == "tool_output"]),
        "duration_ms_client": chat.get("duration_ms_client", 0),
        "answer_chars_client": len(answer),
        "estimated_prompt_tokens": _token_proxy(case.prompt),
        "estimated_answer_tokens_client": _token_proxy(answer),
        "estimated_visible_tokens_client": _token_proxy(case.prompt) + _token_proxy(answer),
        "actual_prompt_tokens": "",
        "actual_completion_tokens": "",
        "actual_total_tokens": "",
        "token_data_quality": "estimated_from_text_length; actual provider usage not exposed by System benchmark API",
        "error": chat.get("error", ""),
        "answer_preview": answer[:1000],
        "prompt": case.prompt,
    }


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _merge_benchmark_metrics(out: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    runs = pd.DataFrame(rows)
    turn_frames = []
    tool_frames = []
    for path in out.glob("*/turns.csv"):
        df = _read_csv_or_empty(path)
        if not df.empty:
            turn_frames.append(df)
    for path in out.glob("*/tools.csv"):
        df = _read_csv_or_empty(path)
        if not df.empty:
            tool_frames.append(df)
    turns = pd.concat(turn_frames, ignore_index=True) if turn_frames else pd.DataFrame()
    tools = pd.concat(tool_frames, ignore_index=True) if tool_frames else pd.DataFrame()

    if not tools.empty:
        tool_summary = tools.groupby(["session_id", "turn"], dropna=False).agg(
            executed_system_tools=("tool_name", lambda s: "|".join(sorted(set(map(str, s))))),
            system_tool_call_count=("tool_name", "count"),
            system_tool_success_count=("success", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            system_tool_duration_ms_total=("duration_ms", "sum"),
            system_tool_duration_ms_avg=("duration_ms", "mean"),
        ).reset_index()
    else:
        tool_summary = pd.DataFrame(columns=[
            "session_id",
            "turn",
            "executed_system_tools",
            "system_tool_call_count",
            "system_tool_success_count",
            "system_tool_duration_ms_total",
            "system_tool_duration_ms_avg",
        ])

    if not turns.empty:
        cols = [
            "session_id",
            "turn",
            "duration_ms",
            "first_response_ms",
            "loop_count",
            "tool_call_count",
            "tool_output_count",
            "thinking_event_count",
            "thinking_char_count",
            "thinking_duration_ms",
            "answer_char_count",
            "tool_duration_ms_total",
            "tool_duration_ms_avg",
            "artifact_count",
        ]
        turns = turns[[col for col in cols if col in turns.columns]]
        runs = runs.merge(turns, on=["session_id", "turn"], how="left", suffixes=("", "_benchmark"))
    runs = runs.merge(tool_summary, on=["session_id", "turn"], how="left")

    numeric_defaults = [
        "duration_ms",
        "first_response_ms",
        "loop_count",
        "tool_call_count",
        "tool_output_count",
        "thinking_event_count",
        "thinking_char_count",
        "thinking_duration_ms",
        "answer_char_count",
        "tool_duration_ms_total",
        "tool_duration_ms_avg",
        "artifact_count",
        "system_tool_call_count",
        "system_tool_success_count",
        "system_tool_duration_ms_total",
        "system_tool_duration_ms_avg",
    ]
    for col in numeric_defaults:
        if col not in runs.columns:
            runs[col] = 0
        runs[col] = runs[col].fillna(0)
    if "executed_system_tools" not in runs.columns:
        runs["executed_system_tools"] = ""
    runs["executed_system_tools"] = runs["executed_system_tools"].fillna("")
    runs["estimated_thinking_tokens"] = runs["thinking_char_count"].apply(lambda n: _token_proxy("x" * int(n or 0)))
    runs["estimated_answer_tokens"] = runs["answer_char_count"].apply(lambda n: _token_proxy("x" * int(n or 0)))
    runs["estimated_total_text_tokens"] = runs["estimated_prompt_tokens"] + runs["estimated_answer_tokens"] + runs["estimated_thinking_tokens"]
    return runs


def _summaries(out: Path, detailed: pd.DataFrame) -> None:
    detailed.to_csv(out / "cnc_tool_runs_detailed.csv", index=False, encoding="utf-8-sig")
    summary = detailed.groupby(["cnc_tool_name", "cnc_tool_layer"], dropna=False).agg(
        runs=("cnc_tool_name", "count"),
        models=("model_name", "nunique"),
        functional_success_rate=("functional_success", "mean"),
        expected_system_tool_call_rate=("expected_system_tool_called", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_benchmark_duration_s=("duration_ms", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_first_response_s=("first_response_ms", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_loop_count=("loop_count", "mean"),
        avg_tool_call_count=("tool_call_count", "mean"),
        avg_tool_output_count=("tool_output_count", "mean"),
        avg_system_tool_call_count=("system_tool_call_count", "mean"),
        avg_system_tool_duration_s=("system_tool_duration_ms_total", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_thinking_event_count=("thinking_event_count", "mean"),
        avg_thinking_duration_s=("thinking_duration_ms", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_thinking_chars=("thinking_char_count", "mean"),
        avg_answer_chars=("answer_char_count", "mean"),
        avg_estimated_prompt_tokens=("estimated_prompt_tokens", "mean"),
        avg_estimated_answer_tokens=("estimated_answer_tokens", "mean"),
        avg_estimated_thinking_tokens=("estimated_thinking_tokens", "mean"),
        avg_estimated_total_text_tokens=("estimated_total_text_tokens", "mean"),
    ).reset_index()
    for col in summary.columns:
        if col.startswith("avg_") or col.endswith("_rate"):
            summary[col] = summary[col].astype(float).round(3)
    summary.to_csv(out / "cnc_tool_summary.csv", index=False, encoding="utf-8-sig")

    matrix = detailed.groupby(["model_label", "cnc_tool_name"], dropna=False)["functional_success"].mean().reset_index()
    matrix.to_csv(out / "model_cnc_tool_matrix.csv", index=False, encoding="utf-8-sig")

    system_events = detailed.groupby(["executed_system_tools", "cnc_tool_name"], dropna=False).agg(
        runs=("cnc_tool_name", "count"),
        avg_system_tool_call_count=("system_tool_call_count", "mean"),
        avg_system_tool_duration_s=("system_tool_duration_ms_total", lambda s: float(pd.Series(s).mean() / 1000)),
    ).reset_index()
    system_events.to_csv(out / "system_tool_usage_by_cnc_tool.csv", index=False, encoding="utf-8-sig")
    _plot_summary(out, summary, matrix)


def _plot_summary(out: Path, summary: pd.DataFrame, matrix: pd.DataFrame) -> None:
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

    def barh(
        metric: str,
        title: str,
        xlabel: str,
        filename: str,
        suffix: str = "",
        precision: int = 1,
        percent: bool = False,
    ) -> None:
        data = summary.sort_values(metric)
        labels = [str(name).replace("_", "\n") for name in data["cnc_tool_name"]]
        values = data[metric].astype(float).tolist()
        plot_values = [value * 100 for value in values] if percent else values
        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.barh(labels, plot_values, color="#0f8fa8")
        ax.set_title(title, weight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        high = max(plot_values) if plot_values else 1
        ax.set_xlim(0, max(high * 1.22, 1))
        for bar, value in zip(bars, plot_values):
            ax.text(value + max(high, 1) * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.{precision}f}{suffix}", va="center", weight="bold")
        fig.savefig(chart_dir / filename, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    barh("functional_success_rate", "CNC Tool Functional Success Rate", "Success rate (%)", "fig01_cnc_tool_success_rate.png", "%", 0, percent=True)
    barh("avg_duration_s", "CNC Tool Average Runtime", "Mean runtime (s)", "fig02_cnc_tool_runtime.png", "s", 1)
    barh("avg_tool_call_count", "CNC Tool Average System Tool Calls", "Mean tool calls", "fig03_cnc_tool_call_count.png", "", 1)
    barh("avg_loop_count", "CNC Tool Average Loop Count", "Mean loop count", "fig04_cnc_tool_loop_count.png", "", 1)
    barh("avg_estimated_total_text_tokens", "CNC Tool Estimated Token Proxy", "Estimated text tokens", "fig05_cnc_tool_token_proxy.png", "", 0)

    pivot = matrix.pivot(index="model_label", columns="cnc_tool_name", values="functional_success")
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(13, max(7, 0.45 * len(pivot) + 2)))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title("Model by CNC Tool Success Matrix", weight="bold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col).replace("_", "\n") for col in pivot.columns], rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=10, weight="bold", color="white" if value >= 0.75 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Success rate")
    fig.savefig(chart_dir / "fig06_model_cnc_tool_success_matrix.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _select_models(models: list[ModelSpec], suite: str, include_embeddings: bool) -> list[ModelSpec]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite}")
    selected = models
    if not include_embeddings:
        selected = [model for model in selected if model.model_name not in EMBEDDING_MODELS]
    if suite == "light":
        selected = [
            model for model in selected
            if model.model_name in CLOUD_MODELS or model.model_name in LIGHT_LOCAL_MODELS
        ]
    return selected


def run(
    base_url: str,
    user_id: str,
    models_file: Path,
    output_dir: Path | None,
    model_limit: int | None,
    case_ids: set[str] | None,
    repeat: int,
    timeout: int,
    include_embeddings: bool,
    suite: str,
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = output_dir or OUTPUTS_DIR / "validation" / "cnc_tool_agent" / run_id
    out.mkdir(parents=True, exist_ok=True)

    models = _select_models(_parse_model_list(models_file), suite=suite, include_embeddings=include_embeddings)
    if model_limit is not None:
        models = models[:model_limit]
    selected_cases = [case for case in CNC_TOOL_CASES if not case_ids or case.case_id in case_ids]

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for model in models:
        safe_model = _safe_name(model.model_name)
        group = _create_group(
            base_url,
            user_id,
            f"cnc-tool-{run_id}-{safe_model}",
            f"CNC tool validation for {model.model_name}; repeat={repeat}",
        )
        groups.append({"model_name": model.model_name, "model_source": model.source, **group})
        model_dir = out / safe_model
        model_dir.mkdir(parents=True, exist_ok=True)
        model_rows: list[dict[str, Any]] = []
        for case in selected_cases:
            for repeat_index in range(1, repeat + 1):
                try:
                    session_id = _create_session(base_url, user_id, group["group_id"], model, case)
                    chat = _chat(base_url, session_id, case.prompt, user_id, timeout)
                    row = _case_result(run_id, model, group, case, repeat_index, session_id, chat)
                except Exception as exc:
                    row = {
                        "run_id": run_id,
                        "model_source": model.source,
                        "model_deployment": "Cloud" if model.model_name in CLOUD_MODELS else "Local",
                        "model_name": model.model_name,
                        "model_label": f"{model.model_name} (Cloud)" if model.model_name in CLOUD_MODELS else model.model_name,
                        "group_id": group["group_id"],
                        "group_name": group["name"],
                        "case_id": case.case_id,
                        "cnc_tool_name": case.cnc_tool_name,
                        "cnc_tool_layer": case.layer,
                        "repeat_index": repeat_index,
                        "session_id": "",
                        "turn": 0,
                        "status": "error",
                        "expected_system_tool": case.expected_system_tool,
                        "system_tool_called": "",
                        "expected_system_tool_called": False,
                        "tool_output_status_ok": False,
                        "tool_output_error_detected": False,
                        "functional_success": False,
                        "tool_input_events": 0,
                        "tool_output_events": 0,
                        "duration_ms_client": 0,
                        "answer_chars_client": 0,
                        "estimated_prompt_tokens": _token_proxy(case.prompt),
                        "estimated_answer_tokens_client": 0,
                        "estimated_visible_tokens_client": _token_proxy(case.prompt),
                        "actual_prompt_tokens": "",
                        "actual_completion_tokens": "",
                        "actual_total_tokens": "",
                        "token_data_quality": "estimated_from_text_length; actual provider usage not exposed by System benchmark API",
                        "error": str(exc),
                        "answer_preview": "",
                        "prompt": case.prompt,
                    }
                rows.append(row)
                model_rows.append(row)
                _write_csv(model_dir / "cnc_tool_runs.csv", model_rows)
                _write_csv(out / "cnc_tool_runs.csv", rows)
        benchmark = _benchmark(base_url, user_id, group["group_id"])
        (model_dir / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(model_dir / "turns.csv", benchmark.get("turns", []))
        _write_csv(model_dir / "tools.csv", benchmark.get("tools", []))

    _write_csv(out / "groups.csv", groups)
    detailed = _merge_benchmark_metrics(out, rows)
    _summaries(out, detailed)
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "user_id": user_id,
        "models_tested": [model.model_name for model in models],
        "suite": suite,
        "excluded_models": [] if include_embeddings else sorted(EMBEDDING_MODELS),
        "case_ids": [case.case_id for case in selected_cases],
        "repeat": repeat,
        "output_dir": str(out),
        "actual_provider_token_usage_available": False,
        "token_estimate_method": "CJK chars * 1.15 + non-CJK chars / 4 plus benchmark answer/thinking char proxies; not billing-grade usage.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "run_id": run_id,
        "output_dir": str(out),
        "models": len(models),
        "cases": len(selected_cases),
        "repeat": repeat,
        "rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNC 8-tool agent validation against configured models.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--models-file", type=Path, default=MODEL_LIST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-limit", type=int, default=None)
    parser.add_argument("--case", action="append", choices=[case.case_id for case in CNC_TOOL_CASES])
    parser.add_argument("--suite", choices=sorted(SUITES), default="full")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--include-embeddings", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        run(
            base_url=args.base_url.rstrip("/"),
            user_id=args.user_id,
            models_file=args.models_file,
            output_dir=args.output_dir,
            model_limit=args.model_limit,
            case_ids=set(args.case) if args.case else None,
            repeat=args.repeat,
            timeout=args.timeout,
            include_embeddings=args.include_embeddings,
            suite=args.suite,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
