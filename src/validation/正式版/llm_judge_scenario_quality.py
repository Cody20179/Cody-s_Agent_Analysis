from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.validation.正式版.scenario_qa_benchmark import MODEL_LIST, _parse_model_list, _token_proxy


DEFAULT_USER_ID = "Cody"
DEFAULT_JUDGE_MODEL = "glm-5.1"
RUBRIC = {
    "correctness": "數值、狀態、時間區間是否符合標準答案。",
    "completeness": "是否回答使用者問題需要的主要資訊與結論。",
    "readability": "工廠使用者是否容易讀懂，表格或條列是否清楚。",
    "practicality": "是否給出可落地的判讀、提醒或下一步。",
    "hallucination_risk": "是否編造標準答案或工具紀錄沒有支持的資訊；0 代表低風險，5 代表高風險。",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _request_json(base_url: str, path: str, user_id: str, timeout: int = 60) -> dict[str, Any] | None:
    req = urllib.request.Request(base_url.rstrip("/") + path, headers={"X-User-ID": user_id})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_full_answer(base_url: str, session_id: str, turn: str, user_id: str, timeout: int) -> tuple[str, str]:
    if not session_id:
        return "", "missing_session"
    payload = _request_json(base_url, f"/sessions/{session_id}/messages", user_id, timeout)
    if not payload:
        return "", "api_unavailable"
    target_turn = int(turn or 0)
    answers = [
        str(msg.get("content", ""))
        for msg in payload.get("messages", [])
        if msg.get("role") == "assistant" and int(msg.get("turn") or 0) == target_turn
    ]
    answer = "\n".join(part for part in answers if part.strip()).strip()
    return (answer, "system_api") if answer else ("", "api_empty")


def _messages(question: str, ground_truth: dict[str, Any], answer: str, qa_score: str) -> list[dict[str, str]]:
    system = (
        "你是工業 CNC Agent 回答品質評審。請只根據使用者問題、標準答案與系統回答評分，"
        "不要自行補充外部知識。分數必須嚴格，若回答沒有提到標準答案中的關鍵數值要扣分。"
        "請輸出 JSON，不要 Markdown。"
    )
    user = {
        "rubric": RUBRIC,
        "score_rule": {
            "correctness": "0-5",
            "completeness": "0-5",
            "readability": "0-5",
            "practicality": "0-5",
            "hallucination_risk": "0-5; lower is better",
            "overall": "0-5; consider correctness first",
        },
        "question": question,
        "ground_truth": ground_truth,
        "system_answer": answer,
        "existing_rule_based_qa_score": qa_score,
        "required_json_schema": {
            "correctness": 0,
            "completeness": 0,
            "readability": 0,
            "practicality": 0,
            "hallucination_risk": 0,
            "overall": 0,
            "reason": "short Traditional Chinese reason",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("judge response did not contain JSON")
    return json.loads(match.group(0))


def _chat_completion(model: Any, messages: list[dict[str, str]], timeout: int) -> tuple[dict[str, Any], int, str]:
    started = time.perf_counter()
    body = {
        "model": model.model_name,
        "messages": messages,
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if model.api_key:
        headers["Authorization"] = f"Bearer {model.api_key}"
    req = urllib.request.Request(
        model.base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        context = ssl.create_default_context()
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        return _extract_json(content), int((time.perf_counter() - started) * 1000), ""
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {}, int((time.perf_counter() - started) * 1000), f"HTTP {exc.code}: {detail[:500]}"
    except Exception as exc:
        return {}, int((time.perf_counter() - started) * 1000), str(exc)


def _score_value(payload: dict[str, Any], key: str) -> float:
    try:
        value = float(payload.get(key, 0))
    except Exception:
        value = 0.0
    return max(0.0, min(5.0, value))


def _plot_outputs(out: Path, judge_rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(judge_rows)
    chart_dir = out / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
    })

    def barh(data: pd.DataFrame, label_col: str, value_col: str, title: str, filename: str) -> None:
        data = data.sort_values(value_col)
        fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(data) + 2)))
        values = data[value_col].astype(float)
        bars = ax.barh(data[label_col].astype(str), values, color="#2a6f97")
        ax.set_xlim(0, 5.5)
        ax.set_xlabel("LLM judge score (0-5)")
        ax.set_title(title, weight="bold")
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", weight="bold")
        fig.savefig(chart_dir / filename, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    model_summary = df.groupby("agent_model", dropna=False).agg(overall_mean=("overall", "mean")).reset_index()
    scenario_summary = df.groupby("scenario_id", dropna=False).agg(overall_mean=("overall", "mean")).reset_index()
    barh(model_summary, "agent_model", "overall_mean", "LLM Judge Score by Agent Model", "fig01_judge_score_by_model.png")
    barh(scenario_summary, "scenario_id", "overall_mean", "LLM Judge Score by Scenario", "fig02_judge_score_by_scenario.png")

    dims = ["correctness", "completeness", "readability", "practicality"]
    dim_summary = df.groupby("agent_model", dropna=False)[dims].mean()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    dim_summary.plot(kind="bar", ax=ax, width=0.78)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (0-5)")
    ax.set_title("LLM Judge Rubric Dimensions by Model", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=4, frameon=False)
    fig.savefig(chart_dir / "fig03_rubric_dimensions_by_model.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(df["total_score"].astype(float), df["overall"].astype(float), s=58, alpha=0.75, color="#0f8fa8")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel("Rule-based QA score (0-100)")
    ax.set_ylabel("LLM judge overall (0-5)")
    ax.set_title("Rule-based QA vs LLM Judge Score", weight="bold")
    ax.grid(alpha=0.25)
    fig.savefig(chart_dir / "fig04_rule_score_vs_judge_score.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(
    scenario_dir: Path,
    output_dir: Path | None,
    judge_model_name: str,
    base_url: str,
    user_id: str,
    timeout: int,
    limit: int | None,
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = output_dir or scenario_dir / f"llm_judge_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    runs = _read_csv(scenario_dir / "scenario_qa_runs.csv")
    if limit:
        runs = runs[:limit]
    ground_truth = json.loads((scenario_dir / "ground_truth.json").read_text(encoding="utf-8"))
    models = {model.model_name: model for model in _parse_model_list(MODEL_LIST)}
    judge_model = models.get(judge_model_name)
    if not judge_model:
        raise RuntimeError(f"judge model not found in {MODEL_LIST}: {judge_model_name}")

    rows: list[dict[str, Any]] = []
    for source in runs:
        full_answer, answer_source = _fetch_full_answer(base_url, source.get("session_id", ""), source.get("turn", ""), user_id, timeout)
        answer = full_answer or source.get("answer_preview", "")
        if not full_answer:
            answer_source = "csv_preview"
        messages = _messages(
            question=source.get("question", ""),
            ground_truth=ground_truth.get(source.get("scenario_id", ""), {}),
            answer=answer,
            qa_score=source.get("total_score", ""),
        )
        judged, duration_ms, error = _chat_completion(judge_model, messages, timeout)
        row = {
            "judge_run_id": run_id,
            "scenario_run_id": source.get("run_id", ""),
            "judge_model": judge_model.model_name,
            "agent_model": source.get("model_name", ""),
            "scenario_id": source.get("scenario_id", ""),
            "category": source.get("category", ""),
            "repeat_index": source.get("repeat_index", ""),
            "session_id": source.get("session_id", ""),
            "question": source.get("question", ""),
            "answer_source": answer_source,
            "answer_chars": len(answer),
            "total_score": source.get("total_score", ""),
            "called_functions": source.get("called_functions", ""),
            "missing_expectations": source.get("missing_expectations", ""),
            "correctness": _score_value(judged, "correctness"),
            "completeness": _score_value(judged, "completeness"),
            "readability": _score_value(judged, "readability"),
            "practicality": _score_value(judged, "practicality"),
            "hallucination_risk": _score_value(judged, "hallucination_risk"),
            "overall": _score_value(judged, "overall"),
            "judge_reason": str(judged.get("reason", ""))[:1000],
            "judge_duration_ms": duration_ms,
            "estimated_judge_prompt_tokens": _token_proxy(json.dumps(messages, ensure_ascii=False)),
            "estimated_judge_completion_tokens": _token_proxy(judged),
            "actual_judge_tokens": "",
            "token_data_quality": "estimated_from_text_length; direct provider usage not normalized by this script",
            "judge_error": error,
        }
        rows.append(row)
        _write_csv(out / "llm_judge_scores.csv", rows)

    df = pd.DataFrame(rows)
    if not df.empty:
        model_summary = df.groupby("agent_model", dropna=False).agg(
            runs=("agent_model", "count"),
            avg_overall=("overall", "mean"),
            avg_correctness=("correctness", "mean"),
            avg_completeness=("completeness", "mean"),
            avg_readability=("readability", "mean"),
            avg_practicality=("practicality", "mean"),
            avg_hallucination_risk=("hallucination_risk", "mean"),
            avg_judge_duration_s=("judge_duration_ms", lambda s: float(pd.Series(s).mean() / 1000)),
        ).reset_index().round(3)
        scenario_summary = df.groupby(["scenario_id", "category"], dropna=False).agg(
            runs=("scenario_id", "count"),
            avg_overall=("overall", "mean"),
            avg_correctness=("correctness", "mean"),
            avg_completeness=("completeness", "mean"),
            avg_readability=("readability", "mean"),
            avg_practicality=("practicality", "mean"),
            avg_hallucination_risk=("hallucination_risk", "mean"),
        ).reset_index().round(3)
        model_summary.to_csv(out / "llm_judge_summary_by_model.csv", index=False, encoding="utf-8-sig")
        scenario_summary.to_csv(out / "llm_judge_summary_by_scenario.csv", index=False, encoding="utf-8-sig")
        _plot_outputs(out, rows)

    manifest = {
        "judge_run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scenario_dir": str(scenario_dir),
        "output_dir": str(out),
        "judge_model": judge_model.model_name,
        "rows": len(rows),
        "rubric": RUBRIC,
        "answer_source_priority": ["System API full assistant message", "scenario_qa_runs.csv answer_preview"],
        "actual_provider_token_usage_available": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge scenario QA answer quality with a stronger LLM.")
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(
        run(
            scenario_dir=args.scenario_dir,
            output_dir=args.output_dir,
            judge_model_name=args.judge_model,
            base_url=args.base_url,
            user_id=args.user_id,
            timeout=args.timeout,
            limit=args.limit,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
