from __future__ import annotations

import argparse
import csv
import json
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

from src.config import OUTPUTS_DIR


MODEL_LIST = Path("模型列表.md")
DEFAULT_USER_ID = "Cody"
MODEL_ALIASES = {"gml-5.1": "glm-5.1"}
ANALYSIS_ROOT = "/Users/cody20179/Desktop/Code/Git_My_Project/Cody-s_Agent_Analysis"


def _analysis_prompt(function_name: str, code: str, summary_request: str) -> str:
    return (
        f"請只用一次 run_python 執行 Cody-s_Agent_Analysis/main.py 的 {function_name}。"
        "run_python 的 code 請使用下列程式碼，不要改用其他函式；timeout 設為 180 秒。\n\n"
        "```python\n"
        f"{code.strip()}\n"
        "```\n\n"
        f"執行後請用中文簡短回覆：{summary_request}"
    )


@dataclass(frozen=True)
class ModelSpec:
    source: str
    provider: str
    base_url: str
    api_key: str
    model_name: str


@dataclass(frozen=True)
class ToolCase:
    case_id: str
    target: str
    tool_names: tuple[str, ...]
    prompt: str
    expected_system_tool: str | None = "run_python"


CASES = [
    ToolCase(
        case_id="tool_inventory",
        target="mounted_tools",
        tool_names=(),
        expected_system_tool=None,
        prompt=(
            "請列出你目前可以使用的工具，依照工具名稱與用途整理。"
            "如果工具清單是由系統提供，請根據你實際可用的工具回答，不要臆測。"
        ),
    ),
    ToolCase(
        case_id="data_status",
        target="data_status",
        tool_names=("run_python",),
        prompt=_analysis_prompt(
            "data_status()",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import data_status
print(json.dumps(data_status(), ensure_ascii=False, indent=2))
""",
            "資料感測器數量、可用時間範圍與是否成功。",
        ),
    ),
    ToolCase(
        case_id="apply_state_model",
        target="apply_state_model",
        tool_names=("run_python",),
        prompt=_analysis_prompt(
            "apply_state_model(start='2026-05-01 00:00:00', end='2026-05-02 00:00:00')",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import apply_state_model
result = apply_state_model(start="2026-05-01 00:00:00", end="2026-05-02 00:00:00")
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "狀態分布、資料筆數與是否成功。",
        ),
    ),
    ToolCase(
        case_id="forecast_future",
        target="forecast_future",
        tool_names=("run_python",),
        prompt=_analysis_prompt(
            "forecast_future(days=[7], model_names=['Prophet'])",
            f"""
import json, os, sys
os.chdir("{ANALYSIS_ROOT}")
sys.path.insert(0, os.getcwd())
from main import forecast_future
result = forecast_future(days=[7], model_names=["Prophet"])
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
""",
            "7 天預測摘要、輸出檔案與是否成功。",
        ),
    ),
    ToolCase(
        case_id="check_anomaly",
        target="check_anomaly",
        tool_names=("run_python",),
        prompt=_analysis_prompt(
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


def _request_json(base_url: str, path: str, method: str = "GET", body: dict | None = None, user_id: str = DEFAULT_USER_ID, timeout: int = 60) -> tuple[int, Any]:
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
    duration_ms = int((time.perf_counter() - started) * 1000)
    answer = "".join(answer_parts)
    return {
        "turn": turn,
        "status": "error" if error else "success",
        "error": error,
        "duration_ms_client": duration_ms,
        "answer_chars_client": len(answer),
        "answer_preview": answer[:1000],
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


def _create_session(base_url: str, user_id: str, group_id: str, model: ModelSpec, case: ToolCase, mounted_tools: list[str]) -> str:
    tool_names = mounted_tools if case.case_id == "tool_inventory" else list(case.tool_names)
    body = {
        "user_id": user_id,
        "group_id": group_id,
        "provider": model.provider,
        "base_url": model.base_url,
        "api_key": model.api_key,
        "model_name": model.model_name,
        "system_name": f"single-tool-{case.case_id}",
        "tool_names": tool_names,
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


def _case_result(run_id: str, model: ModelSpec, group: dict[str, Any], case: ToolCase, session_id: str, chat: dict[str, Any]) -> dict[str, Any]:
    events = chat.pop("events")
    tool_inputs = [event for event in events if event.get("tag") == "tool_input"]
    tool_outputs = [event for event in events if event.get("tag") == "tool_output"]
    names = [str(event.get("chunk", "")).splitlines()[0].replace("Tool:", "").strip() for event in tool_inputs]
    output_text = "\n".join(str(event.get("chunk", "")) for event in tool_outputs)
    tool_output_error = any(marker in output_text for marker in ("Traceback", "[exit code", "ModuleNotFoundError", "Error:"))
    expected_called = case.expected_system_tool is None or case.expected_system_tool in names
    functional_success = chat.get("status", "") == "success" and expected_called and not tool_output_error
    return {
        "run_id": run_id,
        "model_source": model.source,
        "model_name": model.model_name,
        "group_id": group["group_id"],
        "group_name": group["name"],
        "case_id": case.case_id,
        "target": case.target,
        "session_id": session_id,
        "turn": chat.get("turn", 0),
        "status": chat.get("status", ""),
        "expected_system_tool": case.expected_system_tool or "",
        "system_tool_called": "|".join(names),
        "expected_system_tool_called": expected_called,
        "tool_output_error_detected": tool_output_error,
        "functional_success": functional_success,
        "tool_input_events": len(tool_inputs),
        "tool_output_events": len(tool_outputs),
        "duration_ms_client": chat.get("duration_ms_client", 0),
        "answer_chars_client": chat.get("answer_chars_client", 0),
        "error": chat.get("error", ""),
        "answer_preview": chat.get("answer_preview", ""),
        "prompt": case.prompt,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["model_name"], row["model_source"]), []).append(row)
    summary = []
    for (model_name, source), items in sorted(buckets.items()):
        total = len(items)
        success = sum(1 for row in items if row["status"] == "success")
        functional_success = sum(1 for row in items if row["functional_success"])
        expected = sum(1 for row in items if row["expected_system_tool_called"])
        summary.append({
            "model_name": model_name,
            "model_source": source,
            "cases": total,
            "success_rate": round(success / total, 4) if total else 0,
            "functional_success_rate": round(functional_success / total, 4) if total else 0,
            "expected_tool_call_rate": round(expected / total, 4) if total else 0,
            "avg_duration_ms_client": round(sum(int(row["duration_ms_client"] or 0) for row in items) / total, 2) if total else 0,
            "avg_tool_input_events": round(sum(int(row["tool_input_events"] or 0) for row in items) / total, 2) if total else 0,
            "avg_answer_chars": round(sum(int(row["answer_chars_client"] or 0) for row in items) / total, 2) if total else 0,
        })
    return summary


def run(
    base_url: str,
    user_id: str,
    models_file: Path,
    output_dir: Path | None,
    model_limit: int | None,
    case_ids: set[str] | None,
    timeout: int,
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = output_dir or OUTPUTS_DIR / "validation" / "single_tool_agent" / run_id
    out.mkdir(parents=True, exist_ok=True)

    models = _parse_model_list(models_file)
    if model_limit is not None:
        models = models[:model_limit]
    selected_cases = [case for case in CASES if not case_ids or case.case_id in case_ids]

    status, tools_payload = _request_json(base_url, "/tools", user_id=user_id)
    if status != 200:
        raise RuntimeError(f"cannot read tool list: {status} {tools_payload}")
    mounted_tools = list(tools_payload.get("mounted", []))

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for model in models:
        safe_model = _safe_name(model.model_name)
        group = _create_group(
            base_url,
            user_id,
            f"single-tool-{run_id}-{safe_model}",
            f"Single-tool validation for {model.model_name}",
        )
        groups.append({"model_name": model.model_name, "model_source": model.source, **group})
        model_dir = out / safe_model
        model_dir.mkdir(parents=True, exist_ok=True)
        model_rows = []
        for case in selected_cases:
            try:
                session_id = _create_session(base_url, user_id, group["group_id"], model, case, mounted_tools)
                chat = _chat(base_url, session_id, case.prompt, user_id, timeout)
                row = _case_result(run_id, model, group, case, session_id, chat)
            except Exception as exc:
                row = {
                    "run_id": run_id,
                    "model_source": model.source,
                    "model_name": model.model_name,
                    "group_id": group["group_id"],
                    "group_name": group["name"],
                    "case_id": case.case_id,
                    "target": case.target,
                    "session_id": "",
                    "turn": 0,
                    "status": "error",
                    "expected_system_tool": case.expected_system_tool or "",
                    "system_tool_called": "",
                    "expected_system_tool_called": False,
                    "tool_output_error_detected": False,
                    "functional_success": False,
                    "tool_input_events": 0,
                    "tool_output_events": 0,
                    "duration_ms_client": 0,
                    "answer_chars_client": 0,
                    "error": str(exc),
                    "answer_preview": "",
                    "prompt": case.prompt,
                }
            rows.append(row)
            model_rows.append(row)
            _write_csv(model_dir / "runs.csv", model_rows)
        benchmark = _benchmark(base_url, user_id, group["group_id"])
        (model_dir / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(model_dir / "turns.csv", benchmark.get("turns", []))
        _write_csv(model_dir / "tools.csv", benchmark.get("tools", []))

    _write_csv(out / "runs.csv", rows)
    _write_csv(out / "groups.csv", groups)
    _write_csv(out / "summary.csv", _summarize(rows))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at": datetime.now().isoformat(),
                "base_url": base_url,
                "user_id": user_id,
                "models_tested": [model.model_name for model in models],
                "case_ids": [case.case_id for case in selected_cases],
                "output_dir": str(out),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"status": "ok", "run_id": run_id, "output_dir": str(out), "models": len(models), "cases": len(selected_cases)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent single-tool validation against configured models.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--models-file", type=Path, default=MODEL_LIST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-limit", type=int, default=None)
    parser.add_argument("--case", action="append", choices=[case.case_id for case in CASES])
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(
        run(
            base_url=args.base_url.rstrip("/"),
            user_id=args.user_id,
            models_file=args.models_file,
            output_dir=args.output_dir,
            model_limit=args.model_limit,
            case_ids=set(args.case) if args.case else None,
            timeout=args.timeout,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
