from __future__ import annotations

from pathlib import Path

from src.common import write_json, write_run_summary
from src.config import DATA_PROCESSED, DATA_RAW, STATE_DIR
from src.data.loaders import load_sensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
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
    X = scaler.transform(df.loc[valid, FEATURE_COLS])
    df.loc[valid, "cluster"] = model.predict(X)
    df.loc[valid, "gmm_confidence"] = model.predict_proba(X).max(axis=1)
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

def _feature_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLS:
        series = df[col]
        rows.append({
            "feature": col,
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "mean": float(series.mean()) if series.count() else np.nan,
            "std": float(series.std()) if series.count() else np.nan,
            "min": float(series.min()) if series.count() else np.nan,
            "q25": float(series.quantile(0.25)) if series.count() else np.nan,
            "median": float(series.median()) if series.count() else np.nan,
            "q75": float(series.quantile(0.75)) if series.count() else np.nan,
            "max": float(series.max()) if series.count() else np.nan,
        })
    return pd.DataFrame(rows)

def _fit_quality_scores(X: np.ndarray, labels: np.ndarray) -> dict:
    if len(X) <= len(set(labels)) or len(set(labels)) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    rng = np.random.default_rng(42)
    if len(X) > 5000:
        idx = rng.choice(len(X), size=5000, replace=False)
        X_eval, labels_eval = X[idx], labels[idx]
    else:
        X_eval, labels_eval = X, labels
    return {
        "silhouette": float(silhouette_score(X_eval, labels_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(X_eval, labels_eval)),
        "davies_bouldin": float(davies_bouldin_score(X_eval, labels_eval)),
        "evaluation_rows": int(len(X_eval)),
    }

def _plot_feature_distributions(valid: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, FEATURE_COLS):
        ax.hist(valid[col].dropna(), bins=60, color="steelblue", alpha=0.75)
        ax.set_title(col)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(FEATURE_COLS):]:
        ax.axis("off")
    fig.tight_layout()
    path = out_dir / "training_feature_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _plot_cluster_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    sample = df[df["cluster"] >= 0].copy()
    if len(sample) > 12000:
        sample = sample.sample(12000, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(sample["I_mean"], sample["Power"], c=sample["cluster"], s=8, cmap="tab10", alpha=0.55)
    ax.set_xlabel("I_mean")
    ax.set_ylabel("Power")
    ax.set_title("GMM clusters on load features")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    path = out_dir / "training_cluster_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _plot_standardized_boxplot(valid: pd.DataFrame, scaler: StandardScaler, out_dir: Path) -> Path:
    X = pd.DataFrame(scaler.transform(valid[FEATURE_COLS]), columns=FEATURE_COLS)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([X[col].dropna() for col in FEATURE_COLS], labels=FEATURE_COLS, showfliers=False)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_ylabel("Standardized value")
    ax.set_title("Standardized feature scale after preprocessing")
    fig.tight_layout()
    path = out_dir / "training_standardized_feature_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _plot_cluster_confidence(cluster_df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [
        cluster_df.loc[cluster_df["cluster"].eq(cluster), "gmm_confidence"].dropna()
        for cluster in sorted(cluster_df["cluster"].unique())
    ]
    ax.boxplot(groups, labels=[str(v) for v in sorted(cluster_df["cluster"].unique())], showfliers=False)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Maximum posterior probability")
    ax.set_title("GMM assignment confidence by cluster")
    fig.tight_layout()
    path = out_dir / "training_cluster_confidence_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _plot_confidence(df: pd.DataFrame, out_dir: Path) -> Path:
    values = df["gmm_confidence"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=50, color="seagreen", alpha=0.75)
    ax.set_xlabel("Maximum posterior probability")
    ax.set_ylabel("Rows")
    ax.set_title("GMM assignment confidence")
    fig.tight_layout()
    path = out_dir / "application_confidence_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _write_training_outputs(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    model: GaussianMixture,
    scaler: StandardScaler,
    best_k: int,
    bic_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[dict, dict]:
    train_dir = out_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid = df[FEATURE_COLS].dropna()
    X = scaler.transform(valid)
    labels = model.predict(X)
    cluster_df = valid.copy()
    cluster_df["cluster"] = labels
    cluster_df["gmm_confidence"] = model.predict_proba(X).max(axis=1)

    bic_csv = train_dir / "training_bic_aic.csv"
    bic_df.to_csv(bic_csv, index=False)
    feature_csv = train_dir / "training_feature_quality.csv"
    _feature_quality(df).to_csv(feature_csv, index=False)
    profile_csv = train_dir / "training_cluster_profile.csv"
    cluster_df.groupby("cluster")[FEATURE_COLS + ["gmm_confidence"]].agg(["count", "mean", "std", "min", "max"]).to_csv(profile_csv)

    centers = pd.DataFrame(scaler.inverse_transform(model.means_), columns=FEATURE_COLS)
    centers.insert(0, "cluster", range(len(centers)))
    centers_csv = train_dir / "training_cluster_centers.csv"
    centers.to_csv(centers_csv, index=False)

    metrics = {
        "stage": "training",
        "model": "GaussianMixture",
        "covariance_type": model.covariance_type,
        "n_init": model.n_init,
        "random_state": 42,
        "candidate_k": [int(v) for v in bic_df["K"].tolist()],
        "best_k": int(best_k),
        "rows_total": int(len(df)),
        "rows_used": int(len(valid)),
        "rows_dropped": int(len(df) - len(valid)),
        "bic_best": float(bic_df.loc[bic_df["K"].eq(best_k), "BIC"].iloc[0]),
        "aic_best": float(bic_df.loc[bic_df["K"].eq(best_k), "AIC"].iloc[0]),
        **_fit_quality_scores(X, labels),
    }
    metrics_json = write_json(train_dir / "training_metrics.json", metrics)

    files = {
        "bic_aic_csv": str(bic_csv),
        "feature_quality_csv": str(feature_csv),
        "cluster_profile_csv": str(profile_csv),
        "cluster_centers_csv": str(centers_csv),
        "metrics_json": str(metrics_json),
        "bic_curve": str(_plot_bic(bic_df, best_k, train_dir)),
        "feature_distributions": str(_plot_feature_distributions(valid, train_dir)),
        "cluster_scatter": str(_plot_cluster_scatter(labeled, train_dir)),
        "standardized_feature_boxplot": str(_plot_standardized_boxplot(valid, scaler, train_dir)),
        "cluster_confidence_boxplot": str(_plot_cluster_confidence(cluster_df, train_dir)),
    }
    return metrics, files

def _write_application_outputs(labeled: pd.DataFrame, freq_min: int, out_dir: Path) -> tuple[dict, dict]:
    app_dir = out_dir / "application"
    app_dir.mkdir(parents=True, exist_ok=True)
    valid = labeled[labeled["state"].notna() & labeled["state"].ne("Unknown")].copy()

    summary_df = pd.DataFrame.from_dict(_summary(labeled, freq_min), orient="index").reset_index(names="state")
    summary_csv = app_dir / "application_state_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    transition = pd.crosstab(valid["state"].shift(1), valid["state"], normalize="index").fillna(0)
    transition_csv = app_dir / "application_transition_matrix.csv"
    transition.to_csv(transition_csv)

    daily = valid.assign(date=valid["time"].dt.date)
    daily_hours = daily.pivot_table(index="date", columns="state", values="time", aggfunc="count", fill_value=0) * freq_min / 60
    daily_csv = app_dir / "application_daily_state_hours.csv"
    daily_hours.to_csv(daily_csv)

    profile_csv = app_dir / "application_state_feature_profile.csv"
    valid.groupby("state")[FEATURE_COLS + ["gmm_confidence"]].agg(["count", "mean", "std", "min", "max"]).to_csv(profile_csv)

    fig, ax = plt.subplots(figsize=(9, 5))
    transition.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Transition probability")
    ax.set_title("State transition probability")
    fig.tight_layout()
    transition_png = app_dir / "application_transition_matrix.png"
    fig.savefig(transition_png, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    daily_hours.plot(kind="bar", stacked=True, ax=ax, color=[STATE_COLORS.get(c, "#bdbdbd") for c in daily_hours.columns])
    ax.set_ylabel("Hours")
    ax.set_title("Daily state duration")
    fig.tight_layout()
    daily_png = app_dir / "application_daily_state_hours.png"
    fig.savefig(daily_png, dpi=150)
    plt.close(fig)

    metrics = {
        "stage": "application",
        "rows_total": int(len(labeled)),
        "rows_valid_state": int(len(valid)),
        "data_start": str(labeled["time"].min()),
        "data_end": str(labeled["time"].max()),
        "freq_min": int(freq_min),
        "total_hours": float(len(valid) * freq_min / 60),
        "mean_assignment_confidence": float(valid["gmm_confidence"].mean()),
        "low_confidence_rows_posterior_lt_0_8": int(valid["gmm_confidence"].lt(0.8).sum()),
        "state_count": int(valid["state"].nunique()),
    }
    metrics_json = write_json(app_dir / "application_metrics.json", metrics)
    files = {
        "state_summary_csv": str(summary_csv),
        "transition_matrix_csv": str(transition_csv),
        "daily_state_hours_csv": str(daily_csv),
        "state_feature_profile_csv": str(profile_csv),
        "metrics_json": str(metrics_json),
        "state_timeline": str(_plot_timeline(labeled, app_dir)),
        "transition_matrix_plot": str(transition_png),
        "daily_state_hours_plot": str(daily_png),
        "confidence_distribution": str(_plot_confidence(labeled, app_dir)),
    }
    return metrics, files

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
    training_metrics, training_files = _write_training_outputs(df, labeled, model, scaler, best_k, bic_df, out_dir)
    application_metrics, application_files = _write_application_outputs(labeled, freq_min, out_dir)
    run_summary = write_run_summary(out_dir, "state", {
        "start": start,
        "end": end,
        "freq": freq,
        "best_k": best_k,
        "rows": len(labeled),
        "state_summary": summary,
        "training": training_metrics,
        "application": application_metrics,
    })
    return {
        "best_k": best_k,
        "rows": len(labeled),
        "state_summary": summary,
        "training": training_metrics,
        "application": application_metrics,
        "files": {
            "state_labeled_csv": str(state_csv),
            "processed_state_labeled_csv": str(labeled_csv),
            "state_report": str(report),
            "bic_curve": training_files["bic_curve"],
            "state_timeline": application_files["state_timeline"],
            "run_summary": str(run_summary),
            "training": training_files,
            "application": application_files,
        },
    }
