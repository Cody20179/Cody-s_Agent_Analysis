from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "domain": "#2f5597",
    "error_recovery": "#c00000",
    "multi_step": "#548235",
    "out_of_scope": "#7f7f7f",
    "unplanned": "#7030a0",
    "passed": "#548235",
    "failed": "#c00000",
    "bar": "#2f5597",
    "grid": "#d9e2f3",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "axes.edgecolor": "#404040",
            "axes.linewidth": 0.8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _short_case(case_id: str) -> str:
    replacements = {
        "domain_today_status": "Domain: today status",
        "domain_period_status": "Domain: period status",
        "out_of_scope_weather": "Out-of-scope: weather",
        "out_of_scope_translation": "Out-of-scope: translation",
        "unplanned_sensor_compare": "Unplanned: sensor compare",
        "unplanned_operation_advice": "Unplanned: operation advice",
        "error_invalid_date": "Recovery: invalid date",
        "error_no_data_range": "Recovery: no data range",
        "multi_step_status_forecast": "Multi-step: status+forecast",
        "multi_step_guarded_forecast": "Multi-step: guarded forecast",
        "multi_step_anomaly_forecast": "Multi-step: anomaly+forecast",
    }
    return replacements.get(str(case_id), str(case_id).replace("_", " "))


def _load(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agent = pd.read_csv(report_dir / "system_test_agent_cases.csv")
    api = pd.read_csv(report_dir / "system_test_api_cases.csv")
    turns = pd.read_csv(report_dir / "system_test_turns.csv")
    tools = pd.read_csv(report_dir / "system_test_tools.csv")
    for frame in (agent, turns, tools):
        for col in frame.columns:
            if col.endswith("_ms") or col.endswith("_count") or col in {
                "duration_ms_client",
                "tool_input_events",
                "thinking_events_client",
                "thinking_chars_client",
                "estimated_text_tokens",
                "answer_chars_client",
                "loop_count",
                "turn",
                "seq",
            }:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    agent["case_label"] = agent["case_id"].map(_short_case)
    agent["runtime_s"] = agent["duration_ms_client"] / 1000
    return agent, api, turns, tools


def _table_image(df: pd.DataFrame, title: str, path: Path, col_widths: list[float] | None = None) -> None:
    rows, cols = df.shape
    fig_w = 12
    fig_h = max(2.2, 0.42 * rows + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.text(0.0, 0.99, title, transform=ax.transAxes, ha="left", va="top", fontsize=15, fontweight="bold")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.88],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#bfbfbf")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f9fb")
    _save(fig, path)


def _category_summary(agent: pd.DataFrame) -> pd.DataFrame:
    return (
        agent.groupby("category", as_index=False)
        .agg(
            cases=("case_id", "count"),
            expectation_pass=("expectation_passed", lambda s: int(pd.Series(s).astype(bool).sum())),
            avg_runtime_s=("runtime_s", "mean"),
            avg_tools=("tool_input_events", "mean"),
            avg_thinking_events=("thinking_events_client", "mean"),
            avg_token_proxy=("estimated_text_tokens", "mean"),
        )
        .assign(
            expectation_rate=lambda df: df["expectation_pass"] / df["cases"],
            avg_runtime_s=lambda df: df["avg_runtime_s"].round(1),
            avg_tools=lambda df: df["avg_tools"].round(1),
            avg_thinking_events=lambda df: df["avg_thinking_events"].round(1),
            avg_token_proxy=lambda df: df["avg_token_proxy"].round(0).astype(int),
        )
        .sort_values("category")
    )


def _plot_expectation(summary: pd.DataFrame, out: Path) -> None:
    data = summary.copy()
    failed = data["cases"] - data["expectation_pass"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(data))
    ax.barh(y, data["expectation_pass"], color=COLORS["passed"], label="Passed")
    ax.barh(y, failed, left=data["expectation_pass"], color=COLORS["failed"], label="Failed")
    ax.set_yticks(y, data["category"].str.replace("_", " ").str.title())
    ax.set_xlabel("Number of test cases")
    ax.set_title("Expectation Assessment by Scenario Type")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    for idx, passed in enumerate(data["expectation_pass"]):
        ax.text(passed / 2 if passed else 0.05, idx, str(int(passed)), va="center", ha="center", color="white", weight="bold")
        if failed.iloc[idx]:
            ax.text(passed + failed.iloc[idx] / 2, idx, str(int(failed.iloc[idx])), va="center", ha="center", color="white", weight="bold")
    ax.set_xlim(0, max(1, data["cases"].max()) * 1.25)
    _save(fig, out / "fig01_expectation_by_category.png")


def _plot_category_bars(summary: pd.DataFrame, out: Path) -> None:
    labels = summary["category"].str.replace("_", " ").str.title()
    y = np.arange(len(summary))
    metrics = [
        ("avg_runtime_s", "Average Runtime by Scenario Type", "Runtime (s)", "fig02_runtime_by_category.png"),
        ("avg_tools", "Average Tool Calls by Scenario Type", "Tool calls", "fig03_tool_calls_by_category.png"),
        ("avg_thinking_events", "Average Thinking Events by Scenario Type", "Thinking events", "fig04_thinking_events_by_category.png"),
        ("avg_token_proxy", "Estimated Token Proxy by Scenario Type", "Estimated text tokens", "fig05_token_proxy_by_category.png"),
    ]
    for col, title, xlabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        colors = [COLORS.get(category, COLORS["bar"]) for category in summary["category"]]
        bars = ax.barh(y, summary[col], color=colors)
        ax.set_yticks(y, labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x")
        high = max(float(summary[col].max()), 1)
        ax.set_xlim(0, high * 1.25)
        for bar in bars:
            value = bar.get_width()
            ax.text(value + high * 0.025, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", weight="bold")
        _save(fig, out / filename)


def _plot_case_runtime(agent: pd.DataFrame, out: Path) -> None:
    data = (
        agent.groupby(["case_id", "case_label", "category"], as_index=False)
        .agg(runtime_s=("runtime_s", "mean"), tools=("tool_input_events", "mean"))
        .sort_values("runtime_s")
    )
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    colors = [COLORS.get(category, COLORS["bar"]) for category in data["category"]]
    bars = ax.barh(np.arange(len(data)), data["runtime_s"], color=colors)
    ax.set_yticks(np.arange(len(data)), data["case_label"])
    ax.set_xlabel("Runtime (s)")
    ax.set_title("Runtime by Test Case")
    ax.grid(axis="x")
    high = max(float(data["runtime_s"].max()), 1)
    ax.set_xlim(0, high * 1.22)
    for bar in bars:
        value = bar.get_width()
        ax.text(value + high * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", weight="bold", fontsize=10)
    _save(fig, out / "fig06_runtime_by_case.png")


def _plot_tool_summary(tools: pd.DataFrame, out: Path) -> None:
    if tools.empty:
        return
    tools = tools.copy()
    tools["duration_s"] = pd.to_numeric(tools["duration_ms"], errors="coerce").fillna(0) / 1000
    summary = (
        tools.groupby("tool_name", as_index=False)
        .agg(calls=("tool_name", "size"), avg_duration_s=("duration_s", "mean"))
        .sort_values("calls")
    )
    fig, ax = plt.subplots(figsize=(8.4, max(4.8, 0.45 * len(summary) + 1.2)))
    bars = ax.barh(np.arange(len(summary)), summary["calls"], color=COLORS["bar"])
    ax.set_yticks(np.arange(len(summary)), summary["tool_name"])
    ax.set_xlabel("Call count")
    ax.set_title("Tool Call Frequency")
    ax.grid(axis="x")
    high = max(float(summary["calls"].max()), 1)
    ax.set_xlim(0, high * 1.25)
    for bar in bars:
        value = bar.get_width()
        ax.text(value + high * 0.025, bar.get_y() + bar.get_height() / 2, f"{value:.0f}", va="center", weight="bold")
    _save(fig, out / "fig07_tool_frequency.png")

    table = summary.sort_values("calls", ascending=False).head(12).copy()
    table["avg_duration_s"] = table["avg_duration_s"].round(2)
    table.columns = ["Tool", "Calls", "Avg runtime (s)"]
    _table_image(table, "Tool Usage Summary", out / "table04_tool_usage.png", [0.58, 0.18, 0.24])


def _write_index(out: Path, summary: pd.DataFrame, report_dir: Path) -> None:
    payload = {
        "source_report_dir": str(report_dir),
        "figures": sorted(path.name for path in out.glob("fig*.png")),
        "tables": sorted(path.name for path in out.glob("table*.png")),
    }
    (out / "journal_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    table_lines = ["| " + " | ".join(summary.columns) + " |"]
    table_lines.append("|" + "|".join(["---"] * len(summary.columns)) + "|")
    for row in summary.astype(str).itertuples(index=False):
        table_lines.append("| " + " | ".join(row) + " |")

    lines = [
        "# Journal Figures",
        "",
        f"- Source: `{report_dir}`",
        "- Style: Arial, high-DPI PNG, muted journal palette, English labels.",
        "",
        "## Figures",
        "",
        *[f"- `{name}`" for name in payload["figures"]],
        "",
        "## Tables",
        "",
        *[f"- `{name}`" for name in payload["tables"]],
        "",
        "## Category Summary",
        "",
        *table_lines,
    ]
    (out / "journal_figures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_journal_report(report_dir: Path, output_dir: Path | None = None) -> dict[str, str]:
    _style()
    report_dir = report_dir.resolve()
    out = output_dir or report_dir / "journal"
    out.mkdir(parents=True, exist_ok=True)
    agent, api, turns, tools = _load(report_dir)
    summary = _category_summary(agent)

    _plot_expectation(summary, out)
    _plot_category_bars(summary, out)
    _plot_case_runtime(agent, out)
    _plot_tool_summary(tools, out)

    table1 = summary.copy()
    table1["expectation_rate"] = (table1["expectation_rate"] * 100).round(1).astype(str) + "%"
    table1.columns = ["Category", "Cases", "Passed", "Avg runtime (s)", "Avg tools", "Avg thinking", "Token proxy", "Pass rate"]
    _table_image(table1, "Scenario-Type Summary", out / "table01_category_summary.png")

    case_table = agent[["case_label", "category", "status", "expectation_passed", "tool_input_events", "runtime_s"]].copy()
    case_table["runtime_s"] = case_table["runtime_s"].round(1)
    case_table["expectation_passed"] = case_table["expectation_passed"].map(lambda value: "Pass" if bool(value) else "Fail")
    case_table.columns = ["Case", "Category", "Status", "Expectation", "Tools", "Runtime (s)"]
    _table_image(case_table, "Case-Level Results", out / "table02_case_results.png", [0.34, 0.16, 0.12, 0.14, 0.1, 0.14])

    api_table = api[["case_id", "status_code", "expected_status", "passed"]].copy()
    api_table["passed"] = api_table["passed"].map(lambda value: "Pass" if bool(value) else "Fail")
    api_table.columns = ["API case", "Status", "Expected", "Result"]
    _table_image(api_table, "API Error Handling Checks", out / "table03_api_error_checks.png", [0.46, 0.18, 0.18, 0.18])

    _write_index(out, summary, report_dir)
    return {"status": "ok", "output_dir": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate journal-style figures and table images from a system validation report directory.")
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate_journal_report(args.report_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
