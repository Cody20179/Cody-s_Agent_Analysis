from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_LIST = ROOT / "模型列表.md"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "論文驗證" / "CNC_MCP單工具驗證"
MODEL_ALIASES = {"gml-5.1": "glm-5.1"}
EMBEDDING_MODELS = {"qwen3-embedding:4b", "qwen3-embedding:8b"}
VISION_MODELS = {"qwen3-vl:32b", "qwen3-vl:8b"}
CLOUD_MODELS = {"glm-5.1", "kimi-k2.6", "deepseek-v4-flash"}


@dataclass(frozen=True)
class ModelSpec:
    source: str
    deployment: str
    provider: str
    base_url: str
    api_key: str
    model_name: str


@dataclass(frozen=True)
class ToolCase:
    case_id: str
    tool_name: str
    layer: str
    prompt: str


CASES = [
    ToolCase(
        "tool_data_status",
        "tool_data_status",
        "data",
        "請直接呼叫 tool_data_status，不要改用 run_python。完成後用中文簡短回覆資料覆蓋範圍、感測器數量與是否成功。",
    ),
    ToolCase(
        "tool_update_data",
        "tool_update_data",
        "data",
        '請直接呼叫 tool_update_data，參數 sensors="Current_A"，不要改用 run_python。完成後用中文簡短回覆更新是否成功、更新感測器與輸出資料夾。',
    ),
    ToolCase(
        "tool_run_state_analysis",
        "tool_run_state_analysis",
        "state",
        '請直接呼叫 tool_run_state_analysis，參數 start="2026-05-01 00:00:00", end="2026-05-02 00:00:00"，不要改用 run_python。完成後用中文簡短回覆狀態分析是否成功、資料筆數與狀態摘要。',
    ),
    ToolCase(
        "tool_train_forecast",
        "tool_train_forecast",
        "training",
        '請直接呼叫 tool_train_forecast，參數 target="dy", models="BaselineLastWeek", months_back=1，不要改用 run_python。完成後用中文簡短回覆訓練是否成功、使用模型與輸出檔案。',
    ),
    ToolCase(
        "tool_forecast_future",
        "tool_forecast_future",
        "forecast",
        '請直接呼叫 tool_forecast_future，參數 days="7", model_names="Prophet"，不要改用 run_python。完成後用中文簡短回覆 7 天預測是否成功、預測摘要與輸出檔案。',
    ),
    ToolCase(
        "tool_train_anomaly_detection",
        "tool_train_anomaly_detection",
        "training",
        "請直接呼叫 tool_train_anomaly_detection，不要改用 run_python。完成後用中文簡短回覆異常偵測訓練是否成功、模型數與輸出檔案。",
    ),
    ToolCase(
        "tool_check_anomaly",
        "tool_check_anomaly",
        "anomaly",
        '請直接呼叫 tool_check_anomaly，參數 start="2026-05-01 00:00:00", end="2026-05-02 00:00:00", min_models=2，不要改用 run_python。完成後用中文簡短回覆是否異常、票數或 verdict、輸出檔案與是否成功。',
    ),
    ToolCase(
        "tool_run_all",
        "tool_run_all",
        "workflow",
        "請直接呼叫 tool_run_all，不要改用 run_python。完成後用中文簡短回覆完整管線各步驟是否成功，若有錯誤請列出步驟。",
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
    chunks = re.split(r"Model From ", text)
    models: list[ModelSpec] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        source = chunk[0]
        url_match = re.search(r'URL:\s*"([^"]+)"', chunk)
        key_match = re.search(r'Key:\s*"?([^\n"]+)"?', chunk)
        if not url_match:
            continue
        url = _normalize_base_url(url_match.group(1))
        key = (key_match.group(1).strip() if key_match else "")
        if source == "A":
            after = chunk.split("Model List:", 1)[-1]
            for raw in after.splitlines():
                name = raw.strip().strip('"').strip()
                if not name or name.startswith("{") or name.startswith("["):
                    continue
                if re.match(r"^[A-Za-z0-9_.:-]+$", name):
                    clean = MODEL_ALIASES.get(name, name)
                    models.append(ModelSpec("A", "Cloud", "ollama", url, key, clean))
        elif source == "B":
            try:
                raw_json = chunk.split('Model List:"', 1)[1].rsplit('"', 1)[0]
                names = [item["id"] for item in json.loads(raw_json).get("data", []) if item.get("id")]
            except Exception:
                names = re.findall(r'"id"\s*:\s*"([^"]+)"', chunk)
            for name in names:
                clean = MODEL_ALIASES.get(name.strip(), name.strip())
                models.append(ModelSpec("B", "Local", "ollama", url, "", clean))
    return models


def _select_models(models: list[ModelSpec], scope: str) -> list[ModelSpec]:
    excluded = EMBEDDING_MODELS | VISION_MODELS
    selected: list[ModelSpec] = []
    for model in models:
        if model.model_name in excluded:
            continue
        if scope == "cloud" and model.deployment != "Cloud":
            continue
        if scope == "local-language" and model.deployment != "Local":
            continue
        selected.append(model)
    return selected


def _request_json(api: str, path: str, user_id: str, method: str = "GET", body: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        api.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "X-User-Id": user_id},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _sse_chat(api: str, session_id: str, user_id: str, message: str, timeout: int) -> dict[str, Any]:
    data = json.dumps({"message": message}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        api.rstrip("/") + f"/sessions/{session_id}/chat",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "X-User-Id": user_id},
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    answer_parts: list[str] = []
    error = ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        event_name = None
        data_lines: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                if data_lines:
                    payload = _parse_sse_payload("\n".join(data_lines))
                    tag = payload.get("tag") or event_name or ""
                    chunk = payload.get("chunk", "")
                    events.append({"tag": tag, "chunk": chunk, "payload": payload})
                    if tag in {"answer", "message", "final"}:
                        answer_parts.append(str(chunk))
                    if tag == "error":
                        error = str(chunk or payload)
                    if tag == "done":
                        break
                event_name = None
                data_lines = []
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
    return {
        "events": events,
        "answer": "".join(answer_parts),
        "error": error,
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }


def _parse_sse_payload(data: str) -> dict[str, Any]:
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else {"chunk": parsed}
    except Exception:
        return {"chunk": data}


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


def _event_text(events: list[dict[str, Any]], tag: str) -> str:
    return "\n".join(str(event.get("chunk", "")) for event in events if event.get("tag") == tag)


def _case_result(run_id: str, model: ModelSpec, case: ToolCase, session_id: str, chat: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = chat.get("events", [])
    tool_input_text = _event_text(events, "tool_input")
    tool_output_text = _event_text(events, "tool_output")
    answer = chat.get("answer", "")
    error = chat.get("error", "")
    expected_called = case.tool_name in tool_input_text or case.tool_name in tool_output_text
    output_ok = (
        '"status": "ok"' in tool_output_text
        or "'status': 'ok'" in tool_output_text
        or ("status" in tool_output_text[:1500] and "ok" in tool_output_text[:1500])
    )
    tool_output_error = any(marker in tool_output_text for marker in ("Traceback", "Exception", "Error:", "[exit code"))
    functional_success = bool(chat.get("status", "success") == "success" and expected_called and output_ok and not tool_output_error and not error)
    row = {
        "run_id": run_id,
        "model_source": model.source,
        "model_deployment": model.deployment,
        "model_name": model.model_name,
        "model_label": f"{model.model_name} ({model.deployment})",
        "case_id": case.case_id,
        "cnc_mcp_tool_name": case.tool_name,
        "cnc_tool_layer": case.layer,
        "session_id": session_id,
        "status": "error" if error else "success",
        "functional_success": functional_success,
        "expected_tool_called": expected_called,
        "tool_output_status_ok": output_ok,
        "tool_output_error_detected": tool_output_error,
        "tool_input_events": sum(1 for event in events if event.get("tag") == "tool_input"),
        "tool_output_events": sum(1 for event in events if event.get("tag") == "tool_output"),
        "event_count": len(events),
        "duration_ms_client": chat.get("duration_ms", 0),
        "answer_chars": len(answer),
        "tool_output_chars": len(tool_output_text),
        "estimated_answer_tokens": max(0, round(len(answer) / 4)),
        "estimated_tool_output_tokens": max(0, round(len(tool_output_text) / 4)),
        "error": error,
        "answer_preview": answer[:500],
        "tool_input_preview": tool_input_text[:500],
        "tool_output_preview": tool_output_text[:700],
    }
    event_rows = [
        {
            "run_id": run_id,
            "model_name": model.model_name,
            "case_id": case.case_id,
            "session_id": session_id,
            "tag": event.get("tag", ""),
            "chunk": str(event.get("chunk", ""))[:2000],
        }
        for event in events
    ]
    return row, event_rows


def _summaries(out: Path, runs: pd.DataFrame) -> None:
    runs.to_csv(out / "cnc_mcp_tool_runs.csv", index=False, encoding="utf-8-sig")
    by_model = runs.groupby(["model_deployment", "model_name"], dropna=False).agg(
        runs=("case_id", "count"),
        success_rate=("functional_success", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_tool_input_events=("tool_input_events", "mean"),
        avg_tool_output_events=("tool_output_events", "mean"),
        avg_estimated_tool_output_tokens=("estimated_tool_output_tokens", "mean"),
    ).reset_index()
    by_tool = runs.groupby(["cnc_mcp_tool_name", "cnc_tool_layer"], dropna=False).agg(
        runs=("model_name", "count"),
        models=("model_name", "nunique"),
        success_rate=("functional_success", "mean"),
        avg_duration_s=("duration_ms_client", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_estimated_tool_output_tokens=("estimated_tool_output_tokens", "mean"),
    ).reset_index()
    for frame in (by_model, by_tool):
        for col in frame.columns:
            if col.endswith("_rate") or col.startswith("avg_"):
                frame[col] = frame[col].astype(float).round(3)
    by_model.to_csv(out / "summary_by_model.csv", index=False, encoding="utf-8-sig")
    by_tool.to_csv(out / "summary_by_tool.csv", index=False, encoding="utf-8-sig")
    matrix = runs.pivot_table(index="model_label", columns="cnc_mcp_tool_name", values="functional_success", aggfunc="mean")
    matrix.to_csv(out / "model_tool_success_matrix.csv", encoding="utf-8-sig")
    _plot_charts(out, by_model, by_tool, matrix)


def _plot_charts(out: Path, by_model: pd.DataFrame, by_tool: pd.DataFrame, matrix: pd.DataFrame) -> None:
    chart_dir = out / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 21,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.facecolor": "white",
    })

    def barh(data: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str, filename: str, percent: bool = False) -> None:
        data = data.sort_values(value_col, ascending=True)
        labels = [str(v).replace(":latest", "").replace("_", "\n") for v in data[label_col]]
        values = data[value_col].astype(float) * (100 if percent else 1)
        height = max(5.5, 0.48 * len(data) + 1.8)
        fig, ax = plt.subplots(figsize=(13.5, height))
        colors = ["#2f80ed" if dep == "Cloud" else "#27ae60" for dep in data.get("model_deployment", [""] * len(data))]
        ax.barh(labels, values, color=colors, alpha=0.9)
        ax.set_title(title, weight="bold", pad=14)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#d9dee7", linewidth=0.9, alpha=0.9)
        ax.set_axisbelow(True)
        max_value = max(float(values.max()) if len(values) else 1, 1)
        ax.set_xlim(0, 100 if percent else max_value * 1.22)
        for i, value in enumerate(values):
            label = f"{value:.0f}%" if percent else f"{value:.1f}"
            x = min(value + (2 if percent else max_value * 0.015), (100 if percent else max_value * 1.17))
            ax.text(x, i, label, va="center", ha="left", fontsize=12, weight="bold")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(chart_dir / filename, dpi=240, bbox_inches="tight")
        plt.close(fig)

    model_plot = by_model.copy()
    model_plot["label"] = model_plot["model_name"] + " (" + model_plot["model_deployment"] + ")"
    barh(model_plot, "label", "success_rate", "CNC MCP Tool Success Rate by Model", "Success rate (%)", "fig01_success_by_model.png", percent=True)
    barh(by_tool, "cnc_mcp_tool_name", "success_rate", "CNC MCP Tool Success Rate by Tool", "Success rate (%)", "fig02_success_by_tool.png", percent=True)
    barh(model_plot, "label", "avg_duration_s", "CNC MCP Tool Runtime by Model", "Average runtime (s)", "fig03_runtime_by_model.png")
    barh(by_tool, "cnc_mcp_tool_name", "avg_duration_s", "CNC MCP Tool Runtime by Tool", "Average runtime (s)", "fig04_runtime_by_tool.png")
    barh(by_tool, "cnc_mcp_tool_name", "avg_estimated_tool_output_tokens", "CNC MCP Tool Output Token Proxy", "Estimated output tokens", "fig05_tool_output_token_proxy.png")

    fig_w = max(11, 0.75 * len(matrix.columns) + 5)
    fig_h = max(6, 0.52 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix.fillna(0).values * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(c).replace("tool_", "").replace("_", "\n") for c in matrix.columns], rotation=0, ha="center")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(i).replace(":latest", "") for i in matrix.index])
    ax.set_title("Model x CNC MCP Tool Success Matrix", weight="bold", pad=16)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.fillna(0).values[i, j] * 100
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=11, weight="bold")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Success rate (%)")
    fig.tight_layout()
    fig.savefig(chart_dir / "fig06_model_tool_success_matrix.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    api = args.api.rstrip("/")
    user_id = args.user
    settings = _request_json(api, "/settings", user_id=user_id, timeout=60)
    mcps = _request_json(api, "/settings/mcps", user_id=user_id, timeout=60)
    if "CNC_Analysis" not in (mcps.get("mounted") or []):
        raise RuntimeError("CNC_Analysis MCP is not mounted in System settings")

    local_base_url = args.local_base_url or settings.get("model_base_url") or ""
    models = _select_models(_parse_model_list(args.models_file), args.scope)
    if args.model:
        wanted = set(args.model)
        models = [model for model in models if model.model_name in wanted]
    if not models:
        raise RuntimeError("no models selected")
    selected_cases = [case for case in CASES if not args.case or case.case_id in set(args.case)]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{args.scope}_{run_id}"
    out.mkdir(parents=True, exist_ok=True)

    group = _request_json(
        api,
        "/sessions/groups",
        user_id=user_id,
        method="POST",
        body={"name": f"cnc-mcp-tool-{run_id}", "description": "CNC MCP direct tool-call validation"},
        timeout=60,
    )["group"]
    group_id = group["group_id"]

    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    manifest = {
        "run_id": run_id,
        "scope": args.scope,
        "api": api,
        "user_id": user_id,
        "group_id": group_id,
        "mcp_names": ["CNC_Analysis"],
        "tool_names": [],
        "models": [{"source": m.source, "deployment": m.deployment, "provider": m.provider, "model_name": m.model_name} for m in models],
        "cases": [case.case_id for case in selected_cases],
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_dir={out}", flush=True)
    print(f"models={len(models)} cases={len(selected_cases)} scope={args.scope}", flush=True)

    for model_index, model in enumerate(models, 1):
        model_dir = out / _safe_name(model.model_name)
        model_dir.mkdir(exist_ok=True)
        model_base_url = model.base_url
        model_api_key = model.api_key
        if model.deployment == "Local":
            model_base_url = _normalize_base_url(local_base_url or model.base_url)
            model_api_key = ""
        print(f"[model {model_index}/{len(models)}] {model.model_name} ({model.deployment})", flush=True)
        for case_index, case in enumerate(selected_cases, 1):
            print(f"  [{case_index}/{len(selected_cases)}] {case.case_id} start", flush=True)
            try:
                session_body = {
                    "user_id": user_id,
                    "group_id": group_id,
                    "provider": model.provider,
                    "base_url": model_base_url,
                    "api_key": model_api_key,
                    "model_name": model.model_name,
                    "system_name": f"cnc-mcp-{case.case_id}",
                    "tool_names": [],
                    "mcp_names": ["CNC_Analysis"],
                }
                session = _request_json(api, "/sessions", user_id=user_id, method="POST", body=session_body, timeout=180)
                session_id = session["session_id"]
                chat = _sse_chat(api, session_id, user_id, case.prompt, timeout=args.timeout)
                row, events = _case_result(run_id, model, case, session_id, chat)
            except Exception as exc:
                if isinstance(exc, urllib.error.URLError) and "Connection refused" in str(exc):
                    _write_csv(out / "cnc_mcp_tool_runs.csv", rows)
                    _write_csv(out / "events.csv", event_rows)
                    raise RuntimeError("System API connection refused; aborting to avoid recording API outage as model failure") from exc
                row = {
                    "run_id": run_id,
                    "model_source": model.source,
                    "model_deployment": model.deployment,
                    "model_name": model.model_name,
                    "model_label": f"{model.model_name} ({model.deployment})",
                    "case_id": case.case_id,
                    "cnc_mcp_tool_name": case.tool_name,
                    "cnc_tool_layer": case.layer,
                    "session_id": "",
                    "status": "error",
                    "functional_success": False,
                    "expected_tool_called": False,
                    "tool_output_status_ok": False,
                    "tool_output_error_detected": False,
                    "tool_input_events": 0,
                    "tool_output_events": 0,
                    "event_count": 0,
                    "duration_ms_client": 0,
                    "answer_chars": 0,
                    "tool_output_chars": 0,
                    "estimated_answer_tokens": 0,
                    "estimated_tool_output_tokens": 0,
                    "error": str(exc),
                    "answer_preview": "",
                    "tool_input_preview": "",
                    "tool_output_preview": "",
                }
                events = []
            rows.append(row)
            event_rows.extend(events)
            _write_csv(out / "cnc_mcp_tool_runs.csv", rows)
            _write_csv(out / "events.csv", event_rows)
            model_rows = [item for item in rows if item["model_name"] == model.model_name]
            _write_csv(model_dir / "cnc_mcp_tool_runs.csv", model_rows)
            print(
                f"  [{case_index}/{len(selected_cases)}] {case.case_id} done "
                f"success={row['functional_success']} duration={float(row.get('duration_ms_client') or 0) / 1000:.1f}s",
                flush=True,
            )

    runs = pd.DataFrame(rows)
    _summaries(out, runs)
    manifest["ended_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["rows"] = len(rows)
    manifest["success_rate"] = float(runs["functional_success"].mean()) if len(runs) else 0
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(out), "rows": len(rows), "success_rate": manifest["success_rate"]}, ensure_ascii=False, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--user", default="Cody")
    parser.add_argument("--models-file", type=Path, default=DEFAULT_MODEL_LIST)
    parser.add_argument("--scope", choices=["cloud", "local-language", "all-language"], default="all-language")
    parser.add_argument("--model", action="append", help="Run only the named model. Can be repeated.")
    parser.add_argument("--case", action="append", choices=[case.case_id for case in CASES])
    parser.add_argument("--local-base-url", default="", help="Override local model endpoint. API key is taken from System runtime settings when omitted.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
