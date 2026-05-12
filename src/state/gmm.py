from __future__ import annotations

from pathlib import Path

from src.common import write_run_summary
from src.config import DATA_PROCESSED, DATA_RAW, STATE_DIR
from src.data.loaders import load_sensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["I_mean", "I_imbalance", "Power", "PF_abs", "kVAh_rate"]
STATE_NAMES = {
    2: ["Off", "Running"],
    3: ["Off", "Running_Low", "Running_High"],
    4: ["Off", "Idle", "Running_Low", "Running_High"],
    5: ["Off", "Idle", "Running_Low", "Running_High", "Running_Peak"],
}
STATE_COLORS = {
    "Off": "#9e9e9e",
    "Idle": "#64b5f6",
    "Running_Low": "#81c784",
    "Running_High": "#ef5350",
    "Running_Peak": "#b71c1c",
    "Unknown": "#bdbdbd",
}

def load_merged(start=None, end=None, freq: str = "5min", raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    sensors = {
        "IA": "Current_A",
        "IB": "Current_B",
        "IC": "Current_C",
        "Power": "Instantaneous_Total_Power",
        "PF": "PF",
        "kVAh": "kVAh",
    }
    parts = [load_sensor(name, raw_dir, start=start, end=end, freq=freq).rename(alias) for alias, name in sensors.items()]
    df = pd.concat(parts, axis=1)
    df.index.name = "time"
    return df.reset_index()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    current_cols = ["IA", "IB", "IC"]
    df["I_mean"] = df[current_cols].mean(axis=1)
    i_mean = df["I_mean"].replace(0, np.nan)
    df["I_imbalance"] = df[current_cols].std(axis=1) / i_mean * 100
    df["dI_dt"] = df["I_mean"].diff()
    df["kVAh_rate"] = df["kVAh"].diff().clip(lower=0)
    df["PF_abs"] = df["PF"].abs()
    return df

def _fit_gmm(df: pd.DataFrame, k_range=range(2, 5)):
    valid = df[FEATURE_COLS].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)
    rows, models = [], {}
    for k in k_range:
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
        model.fit(X)
        rows.append({"K": k, "BIC": float(model.bic(X)), "AIC": float(model.aic(X))})
        models[k] = model
    bic_df = pd.DataFrame(rows)
    best_k = int(bic_df.loc[bic_df["BIC"].idxmin(), "K"])
    return models[best_k], scaler, best_k, bic_df

def _label_states(df: pd.DataFrame, model: GaussianMixture, scaler: StandardScaler, best_k: int) -> pd.DataFrame:
    df = df.copy()
    valid = df[FEATURE_COLS].notna().all(axis=1)
    df.loc[valid, "cluster"] = model.predict(scaler.transform(df.loc[valid, FEATURE_COLS]))
    df["cluster"] = df["cluster"].fillna(-1).astype(int)
    order = (
        df[df["cluster"] >= 0]
        .groupby("cluster")[["I_mean", "Power", "kVAh_rate"]]
        .mean()
        .assign(score=lambda x: x["I_mean"] + x["Power"] + x["kVAh_rate"])
        .sort_values("score")
    )
    names = STATE_NAMES.get(best_k, [f"State_{i}" for i in range(best_k)])
    mapping = {int(cluster): names[i] for i, cluster in enumerate(order.index)}
    mapping[-1] = "Unknown"
    df["state"] = df["cluster"].map(mapping)
    off_mask = (df["I_mean"].fillna(0) < 0.5) & (df["Power"].fillna(0).abs() < 0.3)
    df.loc[off_mask, "state"] = "Off"
    return df

def _plot_bic(bic_df: pd.DataFrame, best_k: int, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bic_df["K"], bic_df["BIC"], "o-", label="BIC")
    ax.plot(bic_df["K"], bic_df["AIC"], "s--", label="AIC")
    ax.axvline(best_k, color="red", linestyle=":", label=f"Best K={best_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("Criterion")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "bic_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _plot_timeline(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(df["time"], df["I_mean"], color="black", lw=0.6, label="I_mean")
    axes[0].plot(df["time"], df["Power"], color="tab:red", lw=0.6, alpha=0.5, label="Power")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for state in [s for s in df["state"].dropna().unique() if s != "Unknown"]:
        axes[1].fill_between(df["time"], 0, 1, where=df["state"].eq(state), step="post", alpha=0.85, color=STATE_COLORS.get(state, "#bdbdbd"), label=state)
    axes[1].set_yticks([])
    axes[1].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path = out_dir / "state_timeline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _summary(df: pd.DataFrame, freq_min: int) -> dict:
    valid = df[df["state"].notna() & (df["state"] != "Unknown")]
    total = max(len(valid), 1)
    rows = {}
    for state, grp in valid.groupby("state"):
        rows[state] = {
            "count": int(len(grp)),
            "hours": round(len(grp) * freq_min / 60, 2),
            "pct": round(len(grp) / total * 100, 2),
            "I_mean_avg": round(float(grp["I_mean"].mean()), 3),
            "Power_avg": round(float(grp["Power"].mean()), 3),
            "kVAh_rate_avg": round(float(grp["kVAh_rate"].mean()), 4),
        }
    return rows

def _write_report(summary: dict, bic_df: pd.DataFrame, best_k: int, df: pd.DataFrame, out_dir: Path) -> Path:
    lines = [
        "CNC Machine State Analysis Report",
        f"Data range: {df['time'].min()} ~ {df['time'].max()}",
        f"GMM K: {best_k}",
        f"Features: {', '.join(FEATURE_COLS)}",
        "",
        "[BIC/AIC]",
    ]
    for _, row in bic_df.iterrows():
        mark = " best" if int(row["K"]) == best_k else ""
        lines.append(f"K={int(row['K'])} BIC={row['BIC']:.1f} AIC={row['AIC']:.1f}{mark}")
    lines += ["", "[State Summary]", "State,count,hours,pct,I_mean_avg,Power_avg,kVAh_rate_avg"]
    for state, info in sorted(summary.items(), key=lambda x: -x[1]["pct"]):
        lines.append(f"{state},{info['count']},{info['hours']},{info['pct']},{info['I_mean_avg']},{info['Power_avg']},{info['kVAh_rate_avg']}")
    path = out_dir / "state_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def run_state_analysis(start=None, end=None, freq: str = "5min", k_range=range(2, 5), raw_dir: Path = DATA_RAW, out_dir: Path = STATE_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_features(load_merged(start=start, end=end, freq=freq, raw_dir=raw_dir))
    model, scaler, best_k, bic_df = _fit_gmm(df, k_range=k_range)
    labeled = _label_states(df, model, scaler, best_k)
    labeled_csv = DATA_PROCESSED / "state_labeled.csv"
    labeled.to_csv(labeled_csv, index=False)
    state_csv = out_dir / "state_labeled.csv"
    labeled.to_csv(state_csv, index=False)
    freq_min = int(pd.Timedelta(freq).total_seconds() / 60)
    summary = _summary(labeled, freq_min)
    report = _write_report(summary, bic_df, best_k, labeled, out_dir)
    bic_plot = _plot_bic(bic_df, best_k, out_dir)
    timeline = _plot_timeline(labeled, out_dir)
    run_summary = write_run_summary(out_dir, "state", {
        "start": start,
        "end": end,
        "freq": freq,
        "best_k": best_k,
        "rows": len(labeled),
        "state_summary": summary,
    })
    return {
        "best_k": best_k,
        "rows": len(labeled),
        "state_summary": summary,
        "files": {
            "state_labeled_csv": str(state_csv),
            "processed_state_labeled_csv": str(labeled_csv),
            "state_report": str(report),
            "bic_curve": str(bic_plot),
            "state_timeline": str(timeline),
            "run_summary": str(run_summary),
        },
    }
