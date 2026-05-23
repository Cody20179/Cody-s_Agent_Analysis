from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.validation.正式版.llm_judge_scenario_quality import _fetch_full_answer, _token_proxy


DEFAULT_USER_ID = "Cody"
DEFAULT_CODEX_MODEL = "gpt-5.5"
RUBRIC = {
    "correctness": "0-5: 數值、狀態、日期、時間區間是否符合標準答案。",
    "completeness": "0-5: 是否回答使用者問題需要的主要資訊與結論。",
    "readability": "0-5: 工廠使用者是否容易讀懂。",
    "practicality": "0-5: 是否有可落地的判讀、提醒或下一步。",
    "hallucination_risk": "0-5: 是否編造標準答案沒有支持的資訊；0 最低，5 最高。",
    "overall": "0-5: 以 correctness 為優先，再看完整性、可讀性與實用性。",
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


def _clamp_score(value: object) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    return max(0.0, min(5.0, score))


def _schema(path: Path) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "judge_item_id": {"type": "string"},
                        "correctness": {"type": "number"},
                        "completeness": {"type": "number"},
                        "readability": {"type": "number"},
                        "practicality": {"type": "number"},
                        "hallucination_risk": {"type": "number"},
                        "overall": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "judge_item_id",
                        "correctness",
                        "completeness",
                        "readability",
                        "practicality",
                        "hallucination_risk",
                        "overall",
                        "reason",
                    ],
                },
            }
        },
        "required": ["items"],
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def _compact_ground_truth(gt: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "sensor_count",
        "sensors",
        "rows",
        "off_hours",
        "running_high_hours",
        "running_low_hours",
        "idle_hours",
        "running_hours",
        "total_hours",
        "utilization_percent",
        "forecast_7d_increment",
        "forecast_7d_yhat",
        "forecast_30d_increment",
        "forecast_30d_yhat",
        "rate",
        "estimated_30d_cost",
        "verdict",
        "anomaly_count",
        "anomaly_rate",
        "model_votes",
        "state_summary",
    }
    return {key: value for key, value in gt.items() if key in keep}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _prompt(items: list[dict[str, Any]]) -> str:
    payload = {
        "task": "Evaluate CNC Agent answers as a strict Codex judge.",
        "rules": [
            "Use only question, ground_truth, and system_answer.",
            "Do not reward fluent wording if key numbers are wrong.",
            "If the answer contains unsupported numbers or claims, raise hallucination_risk.",
            "Return JSON only. Include one result for every judge_item_id.",
        ],
        "rubric": RUBRIC,
        "items": items,
    }
    return json.dumps(payload, ensure_ascii=False)


def _codex_batch(items: list[dict[str, Any]], model: str, timeout: int) -> tuple[list[dict[str, Any]], int, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        schema_path = tmp / "schema.json"
        output_path = tmp / "last_message.json"
        _schema(schema_path)
        command = [
            "codex",
            "exec",
            "-m",
            model,
            "--ephemeral",
            "--sandbox",
            "read-only",
            "-c",
            'approval_policy="never"',
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            "-",
        ]
        started = time.perf_counter()
        proc = subprocess.run(
            command,
            input=_prompt(items),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        if proc.returncode != 0:
            return [], duration_ms, (proc.stderr or proc.stdout)[-2000:]
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            return list(payload.get("items", [])), duration_ms, ""
        except Exception as exc:
            return [], duration_ms, f"{exc}; stdout={proc.stdout[-1000:]}; stderr={proc.stderr[-1000:]}"


def _plot_outputs(out: Path, rows: list[dict[str, Any]]) -> None:
    chart_dir = out / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ["agent_duration_ms", "tool_input_events", "rule_based_qa_score", "estimated_answer_tokens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 21,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    })

    def barh(data: pd.DataFrame, label_col: str, value_col: str, title: str, filename: str) -> None:
        data = data.sort_values(value_col)
        fig, ax = plt.subplots(figsize=(11, max(5.5, 0.55 * len(data) + 2)))
        values = data[value_col].astype(float)
        bars = ax.barh(data[label_col].astype(str), values, color="#1d6f8f")
        ax.set_xlim(0, 5.5)
        ax.set_xlabel("Codex judge score (0-5)")
        ax.set_title(title, weight="bold")
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", weight="bold")
        fig.savefig(chart_dir / filename, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    model_summary = df.groupby("agent_model", dropna=False).agg(
        overall_mean=("overall", "mean"),
        correctness_mean=("correctness", "mean"),
        avg_duration_s=("agent_duration_ms", lambda s: float(pd.Series(s).mean() / 1000)),
        avg_answer_tokens=("estimated_answer_tokens", "mean"),
        avg_tool_calls=("tool_input_events", "mean"),
    ).reset_index()
    scenario_summary = df.groupby("scenario_id", dropna=False).agg(overall_mean=("overall", "mean")).reset_index()
    barh(model_summary, "agent_model", "overall_mean", "Codex Judge Score by Agent Model", "fig01_codex_score_by_model.png")
    barh(scenario_summary, "scenario_id", "overall_mean", "Codex Judge Score by Scenario", "fig02_codex_score_by_scenario.png")

    dims = ["correctness", "completeness", "readability", "practicality"]
    dim_summary = df.groupby("agent_model", dropna=False)[dims].mean()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    dim_summary.plot(kind="bar", ax=ax, width=0.78)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (0-5)")
    ax.set_title("Codex Judge Rubric Dimensions by Model", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=4, frameon=False)
    fig.savefig(chart_dir / "fig03_codex_rubric_dimensions_by_model.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(df["rule_based_qa_score"].astype(float), df["overall"].astype(float), s=58, alpha=0.76, color="#168aad")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel("Rule-based QA score (0-100)")
    ax.set_ylabel("Codex judge overall (0-5)")
    ax.set_title("Rule-based QA vs Codex Judge Score", weight="bold")
    ax.grid(alpha=0.25)
    fig.savefig(chart_dir / "fig04_rule_score_vs_codex_score.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 7))
    model_summary["quality_pct"] = model_summary["overall_mean"] / 5 * 100
    x = model_summary["avg_duration_s"]
    y = model_summary["quality_pct"]
    ax.axhspan(80, 100, xmin=0, xmax=0.42, color="#d9f2df", alpha=0.9)
    ax.text(x.min() + 0.2, 96, "Best efficiency area\nLow runtime, high quality", color="#2d6a4f", weight="bold", fontsize=12)
    for _, row in model_summary.iterrows():
        size = 180 + float(row["avg_tool_calls"]) * 80
        ax.scatter(row["avg_duration_s"], row["quality_pct"], s=size, marker="*", color="#d62828", edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(row["avg_duration_s"] + 0.25, row["quality_pct"] + 0.5, f"{row['agent_model']}\n{row['quality_pct']:.1f}%, {row['avg_duration_s']:.1f}s", fontsize=11, weight="bold")
    ax.set_xlabel("Average response time (s, lower is better)")
    ax.set_ylabel("Codex judged answer quality (%)")
    ax.set_title("Quality-Efficiency Frontier by Agent Model", weight="bold")
    ax.set_ylim(45, 103)
    ax.set_xlim(max(0, x.min() - 2), x.max() + 6)
    ax.grid(alpha=0.25)
    fig.savefig(chart_dir / "fig05_quality_efficiency_frontier.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(
    scenario_dir: Path,
    output_dir: Path | None,
    codex_model: str,
    base_url: str,
    user_id: str,
    batch_size: int,
    timeout: int,
    limit: int | None,
) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = output_dir or scenario_dir / f"codex_judge_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    source_rows = _read_csv(scenario_dir / "scenario_qa_runs.csv")
    if limit:
        source_rows = source_rows[:limit]
    ground_truth = json.loads((scenario_dir / "ground_truth.json").read_text(encoding="utf-8"))
    prepared: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, start=1):
        full_answer, answer_source = _fetch_full_answer(base_url, source.get("session_id", ""), source.get("turn", ""), user_id, 30)
        answer = full_answer or source.get("answer_preview", "")
        prepared.append({
            "judge_item_id": f"J{index:04d}",
            "source": source,
            "prompt_item": {
                "judge_item_id": f"J{index:04d}",
                "agent_model": source.get("model_name", ""),
                "scenario_id": source.get("scenario_id", ""),
                "repeat_index": source.get("repeat_index", ""),
                "question": source.get("question", ""),
                "ground_truth": _compact_ground_truth(ground_truth.get(source.get("scenario_id", ""), {})),
                "system_answer": _truncate(answer, 4200),
                "rule_based_qa_score": source.get("total_score", ""),
                "missing_expectations": source.get("missing_expectations", ""),
            },
            "answer_source": answer_source if full_answer else "csv_preview",
            "answer_chars": len(answer),
        })

    rows: list[dict[str, Any]] = []
    for offset in range(0, len(prepared), batch_size):
        chunk = prepared[offset:offset + batch_size]
        judged_items, duration_ms, error = _codex_batch([item["prompt_item"] for item in chunk], codex_model, timeout)
        judged_map = {str(item.get("judge_item_id")): item for item in judged_items}
        for item in chunk:
            source = item["source"]
            judged = judged_map.get(item["judge_item_id"], {})
            row = {
                "codex_judge_run_id": run_id,
                "codex_model": codex_model,
                "judge_item_id": item["judge_item_id"],
                "agent_model": source.get("model_name", ""),
                "scenario_id": source.get("scenario_id", ""),
                "category": source.get("category", ""),
                "repeat_index": source.get("repeat_index", ""),
                "session_id": source.get("session_id", ""),
                "question": source.get("question", ""),
                "answer_source": item["answer_source"],
                "answer_chars": item["answer_chars"],
                "agent_duration_ms": source.get("duration_ms_client", ""),
                "tool_input_events": source.get("tool_input_events", ""),
                "rule_based_qa_score": source.get("total_score", ""),
                "called_functions": source.get("called_functions", ""),
                "missing_expectations": source.get("missing_expectations", ""),
                "correctness": _clamp_score(judged.get("correctness")),
                "completeness": _clamp_score(judged.get("completeness")),
                "readability": _clamp_score(judged.get("readability")),
                "practicality": _clamp_score(judged.get("practicality")),
                "hallucination_risk": _clamp_score(judged.get("hallucination_risk")),
                "overall": _clamp_score(judged.get("overall")),
                "codex_reason": str(judged.get("reason", ""))[:1000],
                "batch_duration_ms": duration_ms,
                "estimated_answer_tokens": _token_proxy(item["prompt_item"]["system_answer"]),
                "codex_judge_error": error,
            }
            rows.append(row)
        _write_csv(out / "codex_judge_scores.csv", rows)

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
            avg_duration_s=("agent_duration_ms", lambda s: float(pd.to_numeric(s).mean() / 1000)),
            avg_tool_calls=("tool_input_events", lambda s: float(pd.to_numeric(s).mean())),
            avg_estimated_answer_tokens=("estimated_answer_tokens", "mean"),
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
        model_summary.to_csv(out / "codex_judge_summary_by_model.csv", index=False, encoding="utf-8-sig")
        scenario_summary.to_csv(out / "codex_judge_summary_by_scenario.csv", index=False, encoding="utf-8-sig")
        _plot_outputs(out, rows)

    manifest = {
        "codex_judge_run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scenario_dir": str(scenario_dir),
        "output_dir": str(out),
        "codex_model": codex_model,
        "rows": len(rows),
        "batch_size": batch_size,
        "rubric": RUBRIC,
        "answer_source_priority": ["System API full assistant message", "scenario_qa_runs.csv answer_preview"],
        "cost_note": "No provider dollar cost is stored. Efficiency frontier uses response time as runtime cost proxy.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge scenario QA answer quality with Codex CLI.")
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(
        run(
            scenario_dir=args.scenario_dir,
            output_dir=args.output_dir,
            codex_model=args.codex_model,
            base_url=args.base_url,
            user_id=args.user_id,
            batch_size=args.batch_size,
            timeout=args.timeout,
            limit=args.limit,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
