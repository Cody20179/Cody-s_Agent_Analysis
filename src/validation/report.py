from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR


THEME = {
    "application": "#2563eb",
    "training": "#dc2626",
    "workflow": "#16a34a",
    "single_tool": "#9333ea",
    "pass": "#16a34a",
    "fail": "#dc2626",
    "artifact": "#0f766e",
    "file": "#f59e0b",
    "neutral": "#64748b",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 260,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
            "font.family": "Arial",
            "font.size": 18,
            "axes.titlesize": 25,
            "axes.labelsize": 20,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "legend.title_fontsize": 17,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 1.1,
            "grid.color": "#cbd5e1",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
        }
    )


def _label(value: str, width: int = 24) -> str:
    return textwrap.fill(str(value).replace("_", " "), width=width)


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _barh_labels(ax: plt.Axes, bars, suffix: str = "", precision: int = 2) -> None:
    xmax = ax.get_xlim()[1]
    pad = xmax * 0.01 if xmax else 0.2
    for bar in bars:
        value = bar.get_width()
        text = f"{value:.{precision}f}{suffix}"
        ax.text(
            min(value + pad, xmax * 0.98),
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left" if value + pad < xmax * 0.98 else "right",
            fontsize=16,
            fontweight="bold",
            clip_on=True,
        )


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["kind", "category", "name"], dropna=False)
    out = grouped.agg(
        runs=("status", "size"),
        pass_count=("status", lambda s: int((s == "ok").sum())),
        fail_count=("status", lambda s: int((s != "ok").sum())),
        avg_duration_ms=("duration_ms", "mean"),
        min_duration_ms=("duration_ms", "min"),
        max_duration_ms=("duration_ms", "max"),
        artifact_count=("artifact_count", "mean"),
        key_file_count=("key_file_count", "mean"),
        tool_trace_len=("tool_trace_len", "mean"),
    ).reset_index()
    out["success_rate"] = out["pass_count"] / out["runs"]
    out["avg_duration_s"] = out["avg_duration_ms"] / 1000
    out["min_duration_s"] = out["min_duration_ms"] / 1000
    out["max_duration_s"] = out["max_duration_ms"] / 1000
    out["output_count"] = out["artifact_count"] + out["key_file_count"]
    out["outputs_per_second"] = out["output_count"] / out["avg_duration_s"].replace(0, np.nan)
    return out.sort_values(["category", "kind", "avg_duration_s", "name"]).reset_index(drop=True)


def _plot_result_counts(df: pd.DataFrame, out: Path) -> None:
    grouped = (
        df.assign(group=df["kind"] + "\n" + df["category"])
        .groupby("group")["status"]
        .agg(pass_count=lambda s: int((s == "ok").sum()), fail_count=lambda s: int((s != "ok").sum()))
        .reset_index()
    )
    y = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    pass_bars = ax.barh(y, grouped["pass_count"], color=THEME["pass"], label="Passed")
    fail_bars = ax.barh(y, grouped["fail_count"], left=grouped["pass_count"], color=THEME["fail"], label="Failed")
    ax.set_yticks(y, grouped["group"])
    ax.set_xlabel("Run count")
    ax.set_title("Validation Result Counts")
    ax.grid(axis="x")
    ax.legend(loc="lower right", ncols=2, frameon=False)
    ax.set_xlim(0, max(1, int((grouped["pass_count"] + grouped["fail_count"]).max())) * 1.25)
    for bars in (pass_bars, fail_bars):
        for bar in bars:
            if bar.get_width() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width())}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=17,
                    fontweight="bold",
                )
    _finish(fig, out / "validation_result_counts.png")


def _plot_layer_runtime(summary: pd.DataFrame, out: Path) -> None:
    grouped = summary.groupby(["kind", "category"], as_index=False).agg(avg_duration_s=("avg_duration_s", "mean"))
    grouped["group"] = grouped["kind"] + " / " + grouped["category"]
    grouped = grouped.sort_values("avg_duration_s")
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    colors = [THEME[c] for c in grouped["category"]]
    bars = ax.barh(np.arange(len(grouped)), grouped["avg_duration_s"], color=colors)
    ax.set_yticks(np.arange(len(grouped)), [_label(x, 22) for x in grouped["group"]])
    ax.set_xlabel("Average runtime (seconds)")
    ax.set_title("Runtime by Test Layer")
    ax.grid(axis="x")
    ax.set_xlim(0, grouped["avg_duration_s"].max() * 1.22)
    _barh_labels(ax, bars, "s")
    _finish(fig, out / "avg_duration_by_layer.png")


def _plot_runtime_items(summary: pd.DataFrame, category: str, filename: str, title: str, out: Path) -> None:
    data = summary[summary["category"] == category].sort_values("avg_duration_s")
    fig, ax = plt.subplots(figsize=(14.4, max(7.2, 0.72 * len(data) + 2.2)))
    colors = [THEME[k] for k in data["kind"]]
    bars = ax.barh(np.arange(len(data)), data["avg_duration_s"], color=colors)
    ax.set_yticks(np.arange(len(data)), [_label(x, 26) for x in data["name"]])
    ax.set_xlabel("Average runtime (seconds)")
    ax.set_title(title)
    ax.grid(axis="x")
    ax.set_xlim(0, data["avg_duration_s"].max() * 1.26)
    _barh_labels(ax, bars, "s")
    _finish(fig, out / filename)


def _plot_runtime_range(summary: pd.DataFrame, out: Path) -> None:
    data = summary.sort_values("avg_duration_s")
    y = np.arange(len(data))
    xerr = np.vstack(
        [
            data["avg_duration_s"] - data["min_duration_s"],
            data["max_duration_s"] - data["avg_duration_s"],
        ]
    )
    fig, ax = plt.subplots(figsize=(15, max(8, 0.66 * len(data) + 2.5)))
    colors = [THEME[c] for c in data["category"]]
    bars = ax.barh(y, data["avg_duration_s"], color=colors, alpha=0.82)
    ax.errorbar(data["avg_duration_s"], y, xerr=xerr, fmt="none", ecolor="#0f172a", elinewidth=2.2, capsize=5)
    ax.set_yticks(y, [_label(x, 27) for x in data["name"]])
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Runtime Stability by Item")
    ax.grid(axis="x")
    ax.set_xlim(0, data["max_duration_s"].max() * 1.24)
    _barh_labels(ax, bars, "s")
    _finish(fig, out / "runtime_stability_by_item.png")


def _plot_output_volume(summary: pd.DataFrame, out: Path) -> None:
    data = summary.sort_values("output_count")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(14.4, max(8, 0.66 * len(data) + 2.5)))
    ax.barh(y, data["artifact_count"], color=THEME["artifact"], label="Artifacts")
    ax.barh(y, data["key_file_count"], left=data["artifact_count"], color=THEME["file"], label="Key files")
    ax.set_yticks(y, [_label(x, 27) for x in data["name"]])
    ax.set_xlabel("Average output count")
    ax.set_title("Output Volume by Item")
    ax.grid(axis="x")
    ax.legend(loc="lower right", ncols=2, frameon=False)
    ax.set_xlim(0, max(1, data["output_count"].max()) * 1.22)
    _finish(fig, out / "output_volume_by_item.png")


def _plot_trace_length(summary: pd.DataFrame, out: Path) -> None:
    data = summary[summary["kind"] == "workflow"].sort_values("tool_trace_len")
    fig, ax = plt.subplots(figsize=(12.8, max(7.2, 0.78 * len(data) + 2)))
    bars = ax.barh(np.arange(len(data)), data["tool_trace_len"], color=THEME["workflow"])
    ax.set_yticks(np.arange(len(data)), [_label(x, 26) for x in data["name"]])
    ax.set_xlabel("Tool calls in workflow")
    ax.set_title("Workflow Tool Trace Length")
    ax.grid(axis="x")
    ax.set_xlim(0, max(1, data["tool_trace_len"].max()) * 1.35)
    _barh_labels(ax, bars, "", precision=1)
    _finish(fig, out / "workflow_tool_trace_length.png")


def _paired_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("State query", "apply_state_model_today", "today_status"),
        ("Power forecast", "forecast_future_7d_prophet", "next_week_power_forecast"),
        ("Anomaly check", "check_anomaly_period", "period_anomaly_check"),
    ]
    rows = []
    indexed = summary.set_index("name")
    for label, single, workflow in pairs:
        if single not in indexed.index or workflow not in indexed.index:
            continue
        single_s = float(indexed.loc[single, "avg_duration_s"])
        workflow_s = float(indexed.loc[workflow, "avg_duration_s"])
        rows.append(
            {
                "scenario": label,
                "single_tool_avg_s": single_s,
                "workflow_avg_s": workflow_s,
                "workflow_overhead_s": workflow_s - single_s,
                "workflow_ratio": workflow_s / single_s if single_s else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_paired_runtime(comparisons: pd.DataFrame, out: Path) -> None:
    y = np.arange(len(comparisons))
    height = 0.35
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.barh(y - height / 2, comparisons["single_tool_avg_s"], height=height, color=THEME["single_tool"], label="Single tool")
    ax.barh(y + height / 2, comparisons["workflow_avg_s"], height=height, color=THEME["workflow"], label="Workflow")
    ax.set_yticks(y, comparisons["scenario"])
    ax.set_xlabel("Average runtime (seconds)")
    ax.set_title("Single Tool vs Workflow Runtime")
    ax.grid(axis="x")
    ax.legend(loc="lower right", ncols=2, frameon=False)
    ax.set_xlim(0, comparisons[["single_tool_avg_s", "workflow_avg_s"]].max().max() * 1.24)
    _finish(fig, out / "compare_single_tool_vs_workflow_runtime.png")


def _plot_overhead(comparisons: pd.DataFrame, out: Path) -> None:
    data = comparisons.sort_values("workflow_ratio")
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    colors = [THEME["workflow"] if value >= 1 else THEME["neutral"] for value in data["workflow_ratio"]]
    bars = ax.barh(np.arange(len(data)), data["workflow_ratio"], color=colors)
    ax.axvline(1.0, color="#0f172a", linewidth=2.2)
    ax.set_yticks(np.arange(len(data)), data["scenario"])
    ax.set_xlabel("Workflow runtime / single-tool runtime")
    ax.set_title("Workflow Overhead Ratio")
    ax.grid(axis="x")
    ax.set_xlim(0, max(1.4, data["workflow_ratio"].max() * 1.24))
    _barh_labels(ax, bars, "x", precision=2)
    _finish(fig, out / "workflow_overhead_ratio.png")


def _plot_training_composition(summary: pd.DataFrame, out: Path) -> None:
    single_training_total = float(summary[(summary["kind"] == "single_tool") & (summary["category"] == "training")]["avg_duration_s"].sum())
    init = summary[summary["name"] == "initialize_project"]
    init_s = float(init["avg_duration_s"].iloc[0]) if not init.empty else np.nan
    data = pd.DataFrame(
        {
            "label": ["Standalone training tools total", "Initialize project workflow"],
            "seconds": [single_training_total, init_s],
        }
    )
    fig, ax = plt.subplots(figsize=(13.6, 7.2))
    bars = ax.barh(np.arange(len(data)), data["seconds"], color=[THEME["training"], THEME["workflow"]])
    ax.set_yticks(np.arange(len(data)), [_label(x, 26) for x in data["label"]])
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Training Runtime Comparison")
    ax.grid(axis="x")
    ax.set_xlim(0, data["seconds"].max() * 1.22)
    _barh_labels(ax, bars, "s")
    _finish(fig, out / "training_runtime_composition.png")


def _plot_output_efficiency(summary: pd.DataFrame, out: Path) -> None:
    data = summary.sort_values("outputs_per_second").replace([np.inf, -np.inf], np.nan).fillna(0)
    fig, ax = plt.subplots(figsize=(14.4, max(8, 0.66 * len(data) + 2.5)))
    bars = ax.barh(np.arange(len(data)), data["outputs_per_second"], color=THEME["artifact"])
    ax.set_yticks(np.arange(len(data)), [_label(x, 27) for x in data["name"]])
    ax.set_xlabel("Outputs per second")
    ax.set_title("Output Efficiency by Item")
    ax.grid(axis="x")
    ax.set_xlim(0, max(1, data["outputs_per_second"].max()) * 1.22)
    _barh_labels(ax, bars, "", precision=2)
    _finish(fig, out / "output_throughput_by_item.png")


def _write_report(source_csv: Path, out: Path, df: pd.DataFrame, summary: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    layer = summary.groupby(["kind", "category"]).agg(runs=("runs", "sum"), avg_duration_s=("avg_duration_s", "mean")).reset_index()
    payload = {
        "source_csv": str(source_csv),
        "run_count": int(len(df)),
        "all_success": bool((df["status"] == "ok").all()),
        "layer_summary": layer.to_dict(orient="records"),
        "item_summary": summary.to_dict(orient="records"),
        "comparisons": comparisons.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Thesis Validation Figures",
        "",
        f"- Source CSV: `{source_csv.name}`",
        f"- Test records: {len(df)}",
        f"- Status: {'all passed' if payload['all_success'] else 'has failures'}",
        "- Chart rule: readable Arial font, large labels, consistent horizontal-bar layout where suitable.",
        "- Success-rate chart is replaced by pass/fail counts because all rates are 100%.",
        "",
        "## Figures",
        "",
        "- `validation_result_counts.png`",
        "- `avg_duration_by_layer.png`",
        "- `application_runtime_by_item.png`",
        "- `training_runtime_by_item.png`",
        "- `runtime_stability_by_item.png`",
        "- `output_volume_by_item.png`",
        "- `workflow_tool_trace_length.png`",
        "- `compare_single_tool_vs_workflow_runtime.png`",
        "- `workflow_overhead_ratio.png`",
        "- `training_runtime_composition.png`",
        "- `output_throughput_by_item.png`",
    ]
    (out / "thesis_figures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    layer_rows = [
        "| Layer | Category | Runs | Avg runtime |",
        "|---|---|---:|---:|",
    ]
    for row in layer.sort_values(["category", "kind"]).itertuples(index=False):
        layer_rows.append(f"| {row.kind} | {row.category} | {int(row.runs)} | {row.avg_duration_s:.2f}s |")

    item_rows = [
        "| Item | Type | Category | Runs | Pass/Fail | Avg runtime |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary.sort_values(["category", "kind", "avg_duration_s"]).itertuples(index=False):
        item_rows.append(
            f"| `{row.name}` | {row.kind} | {row.category} | {int(row.runs)} | "
            f"{int(row.pass_count)}/{int(row.fail_count)} | {row.avg_duration_s:.2f}s |"
        )

    comparison_rows = [
        "| Scenario | Single tool | Workflow | Ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in comparisons.itertuples(index=False):
        comparison_rows.append(
            f"| {row.scenario} | {row.single_tool_avg_s:.2f}s | {row.workflow_avg_s:.2f}s | "
            f"{row.workflow_ratio:.2f}x |"
        )

    report_lines = [
        "# Cody Agent Analysis Validation Report",
        "",
        f"- Source CSV: `../{source_csv.name}`",
        f"- Test records: {len(df)}",
        "- Scope: Analysis project tool and workflow layer; LLM natural-language routing is not included.",
        "- Chart style: unified Arial font, large labels, horizontal-bar layout, legends outside data-dense areas.",
        "- Result chart: pass/fail counts are used instead of success-rate bars because this pilot run is 100% successful.",
        "",
        "## Layer Summary",
        "",
        *layer_rows,
        "",
        "## Item Summary",
        "",
        *item_rows,
        "",
        "## Figures",
        "",
        "![Validation result counts](validation_result_counts.png)",
        "",
        "![Runtime by test layer](avg_duration_by_layer.png)",
        "",
        "![Application runtime by item](application_runtime_by_item.png)",
        "",
        "![Training runtime by item](training_runtime_by_item.png)",
        "",
        "![Runtime stability by item](runtime_stability_by_item.png)",
        "",
        "![Output volume by item](output_volume_by_item.png)",
        "",
        "![Workflow tool trace length](workflow_tool_trace_length.png)",
        "",
        "## Workflow Comparison",
        "",
        *comparison_rows,
        "",
        "![Single tool vs workflow runtime](compare_single_tool_vs_workflow_runtime.png)",
        "",
        "![Workflow overhead ratio](workflow_overhead_ratio.png)",
        "",
        "![Training runtime comparison](training_runtime_composition.png)",
        "",
        "![Output efficiency by item](output_throughput_by_item.png)",
    ]
    (out / "pilot_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def generate_report(source_csv: Path, output_dir: Path | None = None) -> dict[str, str]:
    _style()
    source_csv = source_csv.resolve()
    out = output_dir or OUTPUTS_DIR / "validation" / "manual" / f"{source_csv.stem}_figures"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)
    summary = _summary(df)
    comparisons = _paired_comparisons(summary)

    summary.to_csv(out / "summary_by_test_item.csv", index=False)
    comparisons.to_csv(out / "advanced_comparisons.csv", index=False)

    _plot_result_counts(df, out)
    _plot_layer_runtime(summary, out)
    _plot_runtime_items(summary, "application", "application_runtime_by_item.png", "Application Runtime by Item", out)
    _plot_runtime_items(summary, "training", "training_runtime_by_item.png", "Training Runtime by Item", out)
    _plot_runtime_range(summary, out)
    _plot_output_volume(summary, out)
    _plot_trace_length(summary, out)
    if not comparisons.empty:
        _plot_paired_runtime(comparisons, out)
        _plot_overhead(comparisons, out)
    _plot_training_composition(summary, out)
    _plot_output_efficiency(summary, out)
    _write_report(source_csv, out, df, summary, comparisons)
    return {"status": "ok", "output_dir": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-readable validation figures from a validation CSV.")
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate_report(args.source_csv, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
