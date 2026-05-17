from __future__ import annotations

import pickle
import shutil
from pathlib import Path

from src.common import write_json, write_run_summary
from src.config import DATA_PROCESSED, DATA_RAW, STATE_DIR, STATE_MODELS_DIR
from src.data.loaders import load_sensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
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
MODEL_FILE = "gmm_state_model.pkl"


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
    current_std = df[current_cols].std(axis=1)
    i_mean = df["I_mean"].replace(0, np.nan)
    df["I_imbalance"] = (current_std / i_mean * 100).fillna(0)
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


def _predict_clusters(df: pd.DataFrame, model: GaussianMixture, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    valid = df[FEATURE_COLS].notna().all(axis=1)
    df["cluster"] = -1
    df["gmm_confidence"] = np.nan
    if valid.any():
        X = scaler.transform(df.loc[valid, FEATURE_COLS])
        df.loc[valid, "cluster"] = model.predict(X)
        df.loc[valid, "gmm_confidence"] = model.predict_proba(X).max(axis=1)
    df["cluster"] = df["cluster"].astype(int)
    return df


def _state_mapping(df: pd.DataFrame, best_k: int) -> dict[str, str]:
    order = (
        df[df["cluster"] >= 0]
        .groupby("cluster")[["I_mean", "Power", "kVAh_rate"]]
        .mean()
        .assign(score=lambda x: x["I_mean"] + x["Power"] + x["kVAh_rate"])
        .sort_values("score")
    )
    names = STATE_NAMES.get(best_k, [f"State_{i}" for i in range(best_k)])
    mapping = {str(int(cluster)): names[i] for i, cluster in enumerate(order.index)}
    mapping["-1"] = "Unknown"
    return mapping


def _assign_states(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["state"] = df["cluster"].map(lambda value: mapping.get(str(int(value)), "Unknown"))
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
            "median": float(series.quantile(0.50)) if series.count() else np.nan,
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
    ax.set_title("GMM clusters projected on current and power")
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
    ax.set_title("Standardized feature scale")
    fig.tight_layout()
    path = out_dir / "training_standardized_feature_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_cluster_confidence(cluster_df: pd.DataFrame, out_dir: Path) -> Path:
    clusters = sorted(cluster_df["cluster"].unique())
    groups = [cluster_df.loc[cluster_df["cluster"].eq(cluster), "gmm_confidence"].dropna() for cluster in clusters]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, labels=[str(v) for v in clusters], showfliers=False)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Maximum posterior probability")
    ax.set_title("Training assignment confidence by cluster")
    fig.tight_layout()
    path = out_dir / "training_cluster_confidence_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_confidence(df: pd.DataFrame, out_dir: Path, prefix: str) -> Path:
    values = df["gmm_confidence"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=50, color="seagreen", alpha=0.75)
    ax.set_xlabel("Maximum posterior probability")
    ax.set_ylabel("Rows")
    ax.set_title("GMM assignment confidence")
    fig.tight_layout()
    path = out_dir / f"{prefix}_confidence_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_timeline(df: pd.DataFrame, out_dir: Path, prefix: str) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.15]})
    axes[0].plot(df["time"], df["I_mean"], color="black", lw=0.6, label="I_mean")
    axes[0].plot(df["time"], df["Power"], color="tab:red", lw=0.6, alpha=0.5, label="Power")
    axes[0].legend(fontsize=10, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    for state in [s for s in df["state"].dropna().unique() if s != "Unknown"]:
        axes[1].fill_between(
            df["time"],
            0,
            1,
            where=df["state"].eq(state),
            step="post",
            alpha=0.85,
            color=STATE_COLORS.get(state, "#bdbdbd"),
            label=state,
        )
    axes[1].set_yticks([])
    axes[1].set_ylim(0, 1)
    axes[1].legend(ncol=4, fontsize=11, loc="center left", bbox_to_anchor=(0.01, 0.5), frameon=True)
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
    axes[1].tick_params(axis="x", labelsize=11)
    fig.subplots_adjust(bottom=0.12, hspace=0.10)
    path = out_dir / f"{prefix}_state_timeline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_state_hours(hours_df: pd.DataFrame, out_dir: Path, prefix: str, title: str) -> Path:
    width = min(max(9, len(hours_df) * 0.45), 18)
    fig, ax = plt.subplots(figsize=(width, 5.5))
    hours_df.plot(kind="bar", stacked=True, ax=ax, color=[STATE_COLORS.get(c, "#bdbdbd") for c in hours_df.columns])
    ax.set_ylabel("Hours")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend(ncol=min(4, max(1, len(hours_df.columns))), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    fig.tight_layout()
    path = out_dir / f"{prefix}_state_hours.png"
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
            "confidence_avg": round(float(grp["gmm_confidence"].mean()), 4),
        }
    return rows


def _write_training_outputs(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    model: GaussianMixture,
    scaler: StandardScaler,
    best_k: int,
    bic_df: pd.DataFrame,
    state_mapping: dict[str, str],
    out_dir: Path,
) -> tuple[dict, dict]:
    train_dir = out_dir / "training"
    shutil.rmtree(train_dir, ignore_errors=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    valid = df[FEATURE_COLS].dropna()
    X = scaler.transform(valid)
    labels = model.predict(X)
    cluster_df = valid.copy()
    cluster_df["cluster"] = labels
    cluster_df["state_name"] = cluster_df["cluster"].map(lambda value: state_mapping.get(str(int(value)), "Unknown"))
    cluster_df["gmm_confidence"] = model.predict_proba(X).max(axis=1)

    bic_csv = train_dir / "training_bic_aic.csv"
    bic_df.to_csv(bic_csv, index=False)
    feature_csv = train_dir / "training_feature_quality.csv"
    _feature_quality(df).to_csv(feature_csv, index=False)
    profile_csv = train_dir / "training_cluster_profile.csv"
    cluster_df.groupby(["cluster", "state_name"])[FEATURE_COLS + ["gmm_confidence"]].agg(["count", "mean", "std", "min", "max"]).to_csv(profile_csv)

    centers = pd.DataFrame(scaler.inverse_transform(model.means_), columns=FEATURE_COLS)
    centers.insert(0, "cluster", range(len(centers)))
    centers["state_name"] = centers["cluster"].map(lambda value: state_mapping.get(str(int(value)), "Unknown"))
    centers_csv = train_dir / "training_cluster_centers.csv"
    centers.to_csv(centers_csv, index=False)

    metrics = {
        "stage": "training",
        "model": "GaussianMixture",
        "covariance_type": model.covariance_type,
        "n_init": model.n_init,
        "random_state": 42,
        "features": FEATURE_COLS,
        "candidate_k": [int(v) for v in bic_df["K"].tolist()],
        "best_k": int(best_k),
        "state_mapping": state_mapping,
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


def _window_frames(labeled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = labeled.dropna(subset=["time"]).copy()
    end = valid["time"].max()
    if pd.isna(end):
        return {"all": valid}
    month_start = end.normalize().replace(day=1)
    week_start = end - pd.Timedelta(days=7)
    day_start = end.normalize()
    return {
        "all": valid,
        "month": valid[valid["time"].ge(month_start)],
        "week": valid[valid["time"].ge(week_start)],
        "day": valid[valid["time"].ge(day_start)],
    }


def _state_hours(valid: pd.DataFrame, freq_min: int, window: str) -> pd.DataFrame:
    if window == "all":
        index = valid["time"].dt.to_period("M").astype(str)
    elif window == "day":
        index = valid["time"].dt.strftime("%H:00")
    else:
        index = valid["time"].dt.date.astype(str)
    return valid.assign(period=index).pivot_table(index="period", columns="state", values="time", aggfunc="count", fill_value=0) * freq_min / 60


def _write_application_window(window: str, labeled: pd.DataFrame, freq_min: int, app_dir: Path) -> tuple[dict, dict]:
    window_dir = app_dir / window
    window_dir.mkdir(parents=True, exist_ok=True)
    valid = labeled[labeled["state"].notna() & labeled["state"].ne("Unknown")].copy()

    summary_df = pd.DataFrame.from_dict(_summary(labeled, freq_min), orient="index").reset_index(names="state")
    summary_csv = window_dir / f"{window}_state_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    transition = pd.crosstab(valid["state"].shift(1), valid["state"], normalize="index").fillna(0)
    transition_csv = window_dir / f"{window}_transition_matrix.csv"
    transition.to_csv(transition_csv)

    hours = _state_hours(valid, freq_min, window) if not valid.empty else pd.DataFrame()
    hours_csv = window_dir / f"{window}_state_hours.csv"
    hours.to_csv(hours_csv)

    profile_csv = window_dir / f"{window}_state_feature_profile.csv"
    valid.groupby("state")[FEATURE_COLS + ["gmm_confidence"]].agg(["count", "mean", "std", "min", "max"]).to_csv(profile_csv)

    fig, ax = plt.subplots(figsize=(9, 5))
    transition.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Transition probability")
    ax.set_title(f"{window.title()} state transition probability")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    transition_png = window_dir / f"{window}_transition_matrix.png"
    fig.savefig(transition_png, dpi=150)
    plt.close(fig)

    title = {
        "all": "Monthly state duration across full dataset",
        "month": "Daily state duration in latest month",
        "week": "Daily state duration in latest 7 days",
        "day": "Hourly state duration in latest day",
    }.get(window, "State duration")
    hours_png = _plot_state_hours(hours, window_dir, window, title) if not hours.empty else None

    metrics = {
        "window": window,
        "rows_total": int(len(labeled)),
        "rows_valid_state": int(len(valid)),
        "data_start": str(labeled["time"].min()) if not labeled.empty else None,
        "data_end": str(labeled["time"].max()) if not labeled.empty else None,
        "freq_min": int(freq_min),
        "total_hours": float(len(valid) * freq_min / 60),
        "mean_assignment_confidence": float(valid["gmm_confidence"].mean()) if not valid.empty else np.nan,
        "low_confidence_rows_posterior_lt_0_8": int(valid["gmm_confidence"].lt(0.8).sum()) if not valid.empty else 0,
        "state_count": int(valid["state"].nunique()) if not valid.empty else 0,
    }
    metrics_json = write_json(window_dir / f"{window}_metrics.json", metrics)
    files = {
        "state_summary_csv": str(summary_csv),
        "transition_matrix_csv": str(transition_csv),
        "state_hours_csv": str(hours_csv),
        "state_feature_profile_csv": str(profile_csv),
        "metrics_json": str(metrics_json),
        "state_timeline": str(_plot_timeline(labeled, window_dir, window)),
        "transition_matrix_plot": str(transition_png),
        "confidence_distribution": str(_plot_confidence(labeled, window_dir, window)),
    }
    if hours_png:
        files["state_hours_plot"] = str(hours_png)
    return metrics, files


def _write_application_outputs(labeled: pd.DataFrame, freq_min: int, out_dir: Path) -> tuple[dict, dict]:
    app_dir = out_dir / "application"
    shutil.rmtree(app_dir, ignore_errors=True)
    app_dir.mkdir(parents=True, exist_ok=True)
    metrics, files = {}, {}
    for window, frame in _window_frames(labeled).items():
        window_metrics, window_files = _write_application_window(window, frame, freq_min, app_dir)
        metrics[window] = window_metrics
        files[window] = window_files
    write_json(app_dir / "application_metrics.json", metrics)
    return {"stage": "application", "windows": metrics}, files


def _write_report(summary: dict, bic_df: pd.DataFrame | None, best_k: int, df: pd.DataFrame, out_dir: Path) -> Path:
    lines = [
        "CNC Machine State Analysis Report",
        f"Data range: {df['time'].min()} ~ {df['time'].max()}",
        f"GMM K: {best_k}",
        f"Features: {', '.join(FEATURE_COLS)}",
        "",
    ]
    if bic_df is not None:
        lines.append("[BIC/AIC]")
        for _, row in bic_df.iterrows():
            mark = " best" if int(row["K"]) == best_k else ""
            lines.append(f"K={int(row['K'])} BIC={row['BIC']:.1f} AIC={row['AIC']:.1f}{mark}")
        lines.append("")
    lines += ["[State Summary]", "State,count,hours,pct,I_mean_avg,Power_avg,kVAh_rate_avg,confidence_avg"]
    for state, info in sorted(summary.items(), key=lambda x: -x[1]["pct"]):
        lines.append(
            f"{state},{info['count']},{info['hours']},{info['pct']},"
            f"{info['I_mean_avg']},{info['Power_avg']},{info['kVAh_rate_avg']},{info['confidence_avg']}"
        )
    path = out_dir / "state_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_model(model_dir: Path, payload: dict) -> dict[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILE
    with model_path.open("wb") as file:
        pickle.dump(payload, file)
    metadata_path = write_json(model_dir / "gmm_state_model_metadata.json", {
        "model_file": str(model_path),
        "best_k": payload["best_k"],
        "features": payload["feature_cols"],
        "state_mapping": payload["state_mapping"],
        "freq": payload["freq"],
    })
    return {"model_pickle": str(model_path), "metadata_json": str(metadata_path)}


def _load_model(model_dir: Path) -> dict:
    model_path = model_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"state model not found: {model_path}. Run train_state_model() first.")
    with model_path.open("rb") as file:
        return pickle.load(file)


def train_state_model(
    start=None,
    end=None,
    freq: str = "5min",
    k_range=range(2, 5),
    raw_dir: Path = DATA_RAW,
    out_dir: Path = STATE_DIR,
    model_dir: Path = STATE_MODELS_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_features(load_merged(start=start, end=end, freq=freq, raw_dir=raw_dir))
    model, scaler, best_k, bic_df = _fit_gmm(df, k_range=k_range)
    clustered = _predict_clusters(df, model, scaler)
    mapping = _state_mapping(clustered, best_k)
    labeled = _assign_states(clustered, mapping)
    training_metrics, training_files = _write_training_outputs(df, labeled, model, scaler, best_k, bic_df, mapping, out_dir)
    model_files = _save_model(model_dir, {
        "model": model,
        "scaler": scaler,
        "best_k": best_k,
        "bic_df": bic_df,
        "state_mapping": mapping,
        "feature_cols": FEATURE_COLS,
        "freq": freq,
        "train_start": start,
        "train_end": end,
    })
    training_files["model"] = model_files
    return {
        "best_k": best_k,
        "rows": len(df),
        "training": training_metrics,
        "files": training_files,
    }


def apply_state_model(
    start=None,
    end=None,
    freq: str = "5min",
    raw_dir: Path = DATA_RAW,
    out_dir: Path = STATE_DIR,
    model_dir: Path = STATE_MODELS_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    payload = _load_model(model_dir)
    df = build_features(load_merged(start=start, end=end, freq=freq, raw_dir=raw_dir))
    labeled = _assign_states(_predict_clusters(df, payload["model"], payload["scaler"]), payload["state_mapping"])
    labeled_csv = DATA_PROCESSED / "state_labeled.csv"
    labeled.to_csv(labeled_csv, index=False)
    state_csv = out_dir / "state_labeled.csv"
    labeled.to_csv(state_csv, index=False)
    freq_min = int(pd.Timedelta(freq).total_seconds() / 60)
    summary = _summary(labeled, freq_min)
    report = _write_report(summary, payload.get("bic_df"), payload["best_k"], labeled, out_dir)
    application_metrics, application_files = _write_application_outputs(labeled, freq_min, out_dir)
    run_summary = write_run_summary(out_dir, "state", {
        "stage": "application",
        "start": start,
        "end": end,
        "freq": freq,
        "best_k": payload["best_k"],
        "rows": len(labeled),
        "state_summary": summary,
        "application": application_metrics,
    })
    return {
        "best_k": payload["best_k"],
        "rows": len(labeled),
        "state_summary": summary,
        "application": application_metrics,
        "files": {
            "state_labeled_csv": str(state_csv),
            "processed_state_labeled_csv": str(labeled_csv),
            "state_report": str(report),
            "state_timeline": application_files["all"]["state_timeline"],
            "run_summary": str(run_summary),
            "application": application_files,
        },
    }


def run_state_analysis(
    start=None,
    end=None,
    freq: str = "5min",
    k_range=range(2, 5),
    raw_dir: Path = DATA_RAW,
    out_dir: Path = STATE_DIR,
    model_dir: Path = STATE_MODELS_DIR,
) -> dict:
    training = train_state_model(start=start, end=end, freq=freq, k_range=k_range, raw_dir=raw_dir, out_dir=out_dir, model_dir=model_dir)
    application = apply_state_model(start=start, end=end, freq=freq, raw_dir=raw_dir, out_dir=out_dir, model_dir=model_dir)
    return {
        "best_k": application["best_k"],
        "rows": application["rows"],
        "state_summary": application["state_summary"],
        "training": training["training"],
        "application": application["application"],
        "files": {
            **application["files"],
            "bic_curve": training["files"]["bic_curve"],
            "training": training["files"],
        },
    }
