from __future__ import annotations

import json
from pathlib import Path

from src.common import write_json, write_run_summary
from src.config import ANOMALY_DIR, ANOMALY_MODELS_DIR, DATA_PROCESSED, DATA_RAW
from src.data.loaders import load_sensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

FEATURE_COLS = [
    "I_mean",
    "I_imbalance",
    "dI_dt",
    "Power",
    "PF_abs",
    "kVAh_rate",
    "V_mean",
    "V_imbalance",
    "V_deviation",
    "P_error",
]
TRAIN_STATES = ["Running_Low", "Running_High"]
CONTAMINATION = 0.01

def _load_voltage(raw_dir: Path = DATA_RAW, freq: str = "5min") -> pd.DataFrame:
    parts = {
        "Vab": load_sensor("Votage_ab", raw_dir, freq=freq),
        "Vbc": load_sensor("Votage_bc", raw_dir, freq=freq),
        "Vca": load_sensor("Votage_ca", raw_dir, freq=freq),
    }
    df = pd.concat(parts.values(), axis=1, keys=parts.keys()).reset_index()
    df = df.rename(columns={"ts": "time"})
    return df

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    v_cols = ["Vab", "Vbc", "Vca"]
    df["V_mean"] = df[v_cols].mean(axis=1)
    df["V_imbalance"] = df[v_cols].std(axis=1) / df["V_mean"].replace(0, np.nan) * 100
    df["V_deviation"] = (df["V_mean"] - 220).abs() / 220 * 100
    p_calc = df["V_mean"] * df["I_mean"] * np.sqrt(3) * df["PF_abs"] / 1000
    df["P_error"] = ((p_calc - df["Power"].abs()).abs() / df["Power"].abs().replace(0, np.nan) * 100).clip(upper=200)
    df["dI_dt"] = df["dI_dt"].fillna(0)
    for col in ["V_mean", "V_imbalance", "V_deviation", "P_error"]:
        df[col] = df[col].fillna(df[col].median())
    for col in FEATURE_COLS:
        mu, sigma = df[col].mean(), df[col].std()
        if np.isfinite(sigma) and sigma > 0:
            df[col] = df[col].clip(mu - 5 * sigma, mu + 5 * sigma)
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

def load_detection_features(state_csv: Path = DATA_PROCESSED / "state_labeled.csv", raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    if not state_csv.exists():
        raise FileNotFoundError("state_labeled.csv not found; run run_state_analysis first")
    state_df = pd.read_csv(state_csv, low_memory=False)
    state_df["time"] = pd.to_datetime(state_df["time"])
    running = state_df[state_df["state"].isin(TRAIN_STATES)].copy()
    voltage = _load_voltage(raw_dir)
    return _build_features(pd.merge(running, voltage, on="time", how="left"))

def _build_ae(in_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, in_dim),
    )

def _train_ae(X: np.ndarray):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = _build_ae(X.shape[1]).to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(60):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = loss_fn(model(batch), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        losses.append(total / max(len(loader), 1))
    return model, losses

def _ae_scores(model, X: np.ndarray) -> np.ndarray:
    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return -np.mean((X - recon) ** 2, axis=1)

def _synthetic(df_normal: pd.DataFrame, n_each: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    scenarios = {
        "over_current": lambda X: X.__setitem__((slice(None), FEATURE_COLS.index("I_mean")), rng.uniform(50, 65, len(X))),
        "current_imbalance": lambda X: X.__setitem__((slice(None), FEATURE_COLS.index("I_imbalance")), rng.uniform(40, 60, len(X))),
        "voltage_sag": lambda X: X.__setitem__((slice(None), FEATURE_COLS.index("V_mean")), rng.uniform(185, 200, len(X))),
        "low_pf": lambda X: X.__setitem__((slice(None), FEATURE_COLS.index("PF_abs")), rng.uniform(0.2, 0.5, len(X))),
        "power_error": lambda X: X.__setitem__((slice(None), FEATURE_COLS.index("P_error")), rng.uniform(150, 200, len(X))),
    }
    base = df_normal[FEATURE_COLS].values
    for name, inject in scenarios.items():
        X = base[rng.integers(0, len(base), n_each)].astype(float).copy()
        inject(X)
        part = pd.DataFrame(X, columns=FEATURE_COLS)
        part["anomaly_type"] = name
        rows.append(part)
    return pd.concat(rows, ignore_index=True)

def _save_models(models: dict, scaler: StandardScaler, thresholds: dict) -> dict:
    import joblib
    import torch

    ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    joblib.dump(scaler, ANOMALY_MODELS_DIR / "scaler.pkl")
    files["scaler"] = str(ANOMALY_MODELS_DIR / "scaler.pkl")
    write_json(ANOMALY_MODELS_DIR / "thresholds.json", thresholds)
    files["thresholds"] = str(ANOMALY_MODELS_DIR / "thresholds.json")
    joblib.dump(models["IsolationForest"], ANOMALY_MODELS_DIR / "isolation_forest.pkl")
    files["IsolationForest"] = str(ANOMALY_MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(models["OneClassSVM"], ANOMALY_MODELS_DIR / "one_class_svm.pkl")
    files["OneClassSVM"] = str(ANOMALY_MODELS_DIR / "one_class_svm.pkl")
    torch.save({"state_dict": models["Autoencoder"].state_dict(), "in_dim": len(FEATURE_COLS)}, ANOMALY_MODELS_DIR / "autoencoder.pt")
    files["Autoencoder"] = str(ANOMALY_MODELS_DIR / "autoencoder.pt")
    return files

def load_models():
    import joblib
    import torch

    scaler = joblib.load(ANOMALY_MODELS_DIR / "scaler.pkl")
    thresholds = json.load((ANOMALY_MODELS_DIR / "thresholds.json").open(encoding="utf-8"))
    ckpt = torch.load(ANOMALY_MODELS_DIR / "autoencoder.pt", map_location="cpu", weights_only=True)
    ae = _build_ae(ckpt["in_dim"])
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()
    models = {
        "IsolationForest": joblib.load(ANOMALY_MODELS_DIR / "isolation_forest.pkl"),
        "OneClassSVM": joblib.load(ANOMALY_MODELS_DIR / "one_class_svm.pkl"),
        "Autoencoder": ae,
    }
    return models, scaler, thresholds

def _score(name: str, model, X: np.ndarray) -> np.ndarray:
    if name == "Autoencoder":
        return _ae_scores(model, X)
    return model.score_samples(X)

def _plot_scores(scores: dict[str, np.ndarray], thresholds: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(scores), figsize=(5 * len(scores), 4))
    if len(scores) == 1:
        axes = [axes]
    for ax, (name, values) in zip(axes, scores.items()):
        ax.hist(values, bins=60, color="steelblue", alpha=0.75)
        ax.axvline(thresholds[name], color="red", linestyle="--")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_losses(losses: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, color="darkorange", lw=1.5)
    ax.set_title("Autoencoder training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _write_metrics_table(metrics: dict, path: Path) -> Path:
    rows = []
    for name, values in metrics.items():
        rows.append({
            "model": name,
            "accuracy": values.get("accuracy"),
            "precision": values.get("precision"),
            "recall": values.get("recall"),
            "f1": values.get("f1"),
            "auc_roc": values.get("auc_roc"),
            "test_anomaly_count": values.get("test_anomaly_count"),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

def _plot_metric_bars(metrics: dict, path: Path) -> None:
    rows = []
    for name, values in metrics.items():
        for metric in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            rows.append({"model": name, "metric": metric, "value": values.get(metric)})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="metric", columns="model", values="value")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Anomaly detector validation metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_feature_distributions(combined: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()
    normal = combined[combined["label"] == 0]
    anomaly = combined[combined["label"] == 1]
    for ax, col in zip(axes, FEATURE_COLS):
        ax.hist(normal[col], bins=40, alpha=0.55, label="Normal", color="steelblue", density=True)
        ax.hist(anomaly[col], bins=40, alpha=0.55, label="Synthetic anomaly", color="crimson", density=True)
        ax.set_title(col)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].legend(loc="upper right")
    fig.suptitle("Feature distributions: normal vs synthetic anomaly")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_feature_distributions_by_type(combined: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    normal = combined[combined["label"] == 0]
    paths = []
    for anomaly_type, anomaly in combined[combined["label"] == 1].groupby("anomaly_type"):
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        axes = axes.ravel()
        for ax, col in zip(axes, FEATURE_COLS):
            ax.hist(normal[col], bins=40, alpha=0.55, label="Normal", color="steelblue", density=True)
            ax.hist(anomaly[col], bins=40, alpha=0.60, label=anomaly_type, color="crimson", density=True)
            ax.set_title(col)
            ax.grid(True, axis="y", alpha=0.2)
        axes[0].legend(loc="upper right")
        fig.suptitle(f"Feature distributions: normal vs {anomaly_type}")
        fig.tight_layout()
        path = out_dir / f"{anomaly_type}_feature_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths

def _write_training_data_profile(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, syn: pd.DataFrame, out_dir: Path) -> dict:
    profile = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "train_ratio": float(len(train_df) / max(len(df), 1)),
        "test_ratio": float(len(test_df) / max(len(df), 1)),
        "time_start": str(df["time"].min()),
        "time_end": str(df["time"].max()),
        "train_start": str(train_df["time"].min()),
        "train_end": str(train_df["time"].max()),
        "test_start": str(test_df["time"].min()),
        "test_end": str(test_df["time"].max()),
        "resample_frequency": "5min",
        "training_states": TRAIN_STATES,
        "state_counts": {k: int(v) for k, v in df["state"].value_counts().to_dict().items()},
        "feature_columns": FEATURE_COLS,
        "synthetic_validation": {
            "rows_total": int(len(syn)),
            "rows_per_type": {k: int(v) for k, v in syn["anomaly_type"].value_counts().to_dict().items()},
            "types": sorted(syn["anomaly_type"].unique().tolist()),
        },
        "normal_data_assumption": "Running_Low and Running_High records are treated as normal operating data because real fault labels are unavailable.",
    }
    write_json(out_dir / "training_data_profile.json", profile)
    df[FEATURE_COLS].describe().T.to_csv(out_dir / "feature_summary.csv")
    return profile

def _plot_score_by_label(eval_df: pd.DataFrame, thresholds: dict, path: Path, log_y: bool = False) -> None:
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.5 * len(thresholds), 4.5))
    if len(thresholds) == 1:
        axes = [axes]
    for ax, name in zip(axes, thresholds):
        normal = eval_df.loc[eval_df["label"] == 0, f"score_{name}"]
        anomaly = eval_df.loc[eval_df["label"] == 1, f"score_{name}"]
        ax.hist(normal, bins=50, alpha=0.65, label="Normal", color="steelblue", density=True)
        ax.hist(anomaly, bins=50, alpha=0.65, label="Synthetic anomaly", color="crimson", density=True)
        ax.axvline(thresholds[name], color="black", linestyle="--", lw=1.2, label="Threshold")
        ax.set_title(name)
        ax.set_xlabel("Anomaly score")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].legend(loc="upper left")
    suffix = " (log Y)" if log_y else ""
    fig.suptitle(f"Detector score distributions by label{suffix}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_score_by_label_log_xy(eval_df: pd.DataFrame, thresholds: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.5 * len(thresholds), 4.5))
    if len(thresholds) == 1:
        axes = [axes]
    for ax, name in zip(axes, thresholds):
        normal = eval_df.loc[eval_df["label"] == 0, f"score_{name}"]
        anomaly = eval_df.loc[eval_df["label"] == 1, f"score_{name}"]
        ax.hist(normal, bins=50, alpha=0.65, label="Normal", color="steelblue", density=True)
        ax.hist(anomaly, bins=50, alpha=0.65, label="Synthetic anomaly", color="crimson", density=True)
        ax.axvline(thresholds[name], color="black", linestyle="--", lw=1.2, label="Threshold")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Anomaly score (symlog)")
        ax.grid(True, axis="both", alpha=0.2)
    axes[0].legend(loc="upper left")
    fig.suptitle("Detector score distributions by label (symlog X, log Y)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_confusion_matrices(eval_df: pd.DataFrame, model_names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.5))
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(eval_df["label"], eval_df[f"pred_{name}"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], labels=["Pred normal", "Pred anomaly"])
        ax.set_yticks([0, 1], labels=["True normal", "True anomaly"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices on synthetic validation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_roc_curves(eval_df: pd.DataFrame, model_names: list[str], path: Path, log_x: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in model_names:
        fpr, tpr, _ = roc_curve(eval_df["label"], -eval_df[f"score_{name}"])
        auc_value = roc_auc_score(eval_df["label"], -eval_df[f"score_{name}"])
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} AUC={auc_value:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlim(0, 1)
        ax.set_title("ROC curves on synthetic validation (log FPR)")
    else:
        ax.set_title("ROC curves on synthetic validation")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_anomaly_type_detection(eval_df: pd.DataFrame, model_names: list[str], table_path: Path, plot_path: Path) -> tuple[Path, Path]:
    rows = []
    anomalies = eval_df[eval_df["label"] == 1]
    for anomaly_type, group in anomalies.groupby("anomaly_type"):
        for name in model_names:
            rows.append({
                "anomaly_type": anomaly_type,
                "model": name,
                "detection_rate": float(group[f"pred_{name}"].mean()),
                "rows": int(len(group)),
            })
    rate_df = pd.DataFrame(rows)
    rate_df.to_csv(table_path, index=False)
    if not rate_df.empty:
        pivot = rate_df.pivot(index="anomaly_type", columns="model", values="detection_rate")
        fig, ax = plt.subplots(figsize=(12, 5.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Detection rate by synthetic anomaly type")
        ax.set_xlabel("Synthetic anomaly type")
        ax.set_ylabel("Detection rate")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    return table_path, plot_path

def _plot_check_timeline(result: pd.DataFrame, anomaly_mask: pd.Series, path: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, col in zip(axes, ["I_mean", "V_mean", "Power"]):
        ax.plot(result["time"], result[col], color="steelblue", lw=1.1, label=col)
        flagged = result[anomaly_mask]
        if not flagged.empty:
            ax.scatter(flagged["time"], flagged[col], color="crimson", s=22, label="Anomaly vote >= threshold", zorder=5)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Anomaly check timeline")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _plot_check_scores(result: pd.DataFrame, thresholds: dict, path: Path) -> Path:
    score_cols = [c for c in result.columns if c.startswith("score_")]
    fig, axes = plt.subplots(len(score_cols), 1, figsize=(14, 3.2 * len(score_cols)), sharex=True)
    if len(score_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, score_cols):
        name = col.replace("score_", "")
        ax.plot(result["time"], result[col], color="steelblue", lw=1.1, label=name)
        ax.axhline(thresholds[name], color="crimson", linestyle="--", lw=1.0, label="Threshold")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Anomaly check model scores")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def _plot_check_votes(votes: dict[str, int], rows: int, path: Path) -> Path:
    names = list(votes)
    counts = [votes[name] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, counts, color=["tab:blue", "tab:orange", "tab:green"][:len(names)])
    ax.set_title("Anomaly check votes by model")
    ax.set_ylabel("Flagged rows")
    max_count = max(counts) if counts else 0
    if max_count == 0:
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.55, "No model flagged anomalies in this interval", transform=ax.transAxes, ha="center", va="center", fontsize=12)
    else:
        ax.set_ylim(0, max_count * 1.25)
    ax.grid(True, axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{count}/{rows}" for count in counts], padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def train_anomaly_detection(out_dir: Path = ANOMALY_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    table_dir = out_dir / "tables"
    feature_type_plot_dir = plot_dir / "feature_by_anomaly_type"
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    df = load_detection_features()
    features_csv = DATA_PROCESSED / "detection_features.csv"
    df.to_csv(features_csv, index=False)
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    scaler = StandardScaler().fit(train_df[FEATURE_COLS].values)
    X_train = scaler.transform(train_df[FEATURE_COLS].values)
    X_test = scaler.transform(test_df[FEATURE_COLS].values)

    models = {
        "IsolationForest": IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=42, n_jobs=-1).fit(X_train),
        "OneClassSVM": OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale").fit(X_train),
    }
    models["Autoencoder"], ae_losses = _train_ae(X_train)
    train_scores = {name: _score(name, model, X_train) for name, model in models.items()}
    test_scores = {name: _score(name, model, X_test) for name, model in models.items()}
    thresholds = {name: float(np.percentile(scores, CONTAMINATION * 100)) for name, scores in train_scores.items()}

    syn = _synthetic(test_df, n_each=50)
    data_profile = _write_training_data_profile(df, train_df, test_df, syn, table_dir)
    combined = pd.concat([
        test_df[FEATURE_COLS].assign(label=0, anomaly_type="normal"),
        syn[FEATURE_COLS + ["anomaly_type"]].assign(label=1),
    ], ignore_index=True)
    y_true = combined["label"].values
    X_combined = scaler.transform(combined[FEATURE_COLS].values)
    eval_df = combined[["label", "anomaly_type"] + FEATURE_COLS].copy()
    metrics = {}
    for name, model in models.items():
        scores = _score(name, model, X_combined)
        pred = (scores < thresholds[name]).astype(int)
        eval_df[f"score_{name}"] = scores
        eval_df[f"pred_{name}"] = pred
        metrics[name] = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, -scores)),
            "test_anomaly_count": int((test_scores[name] < thresholds[name]).sum()),
        }
    metrics["Autoencoder"]["loss"] = {
        "initial": float(ae_losses[0]) if ae_losses else None,
        "final": float(ae_losses[-1]) if ae_losses else None,
        "min": float(min(ae_losses)) if ae_losses else None,
        "epochs": len(ae_losses),
    }

    model_files = _save_models(models, scaler, thresholds)
    metrics_path = write_json(out_dir / "anomaly_metrics.json", metrics)
    eval_path = table_dir / "synthetic_validation_predictions.csv"
    eval_df.to_csv(eval_path, index=False)
    metrics_table_path = _write_metrics_table(metrics, table_dir / "model_metrics.csv")
    type_rates_path, type_rates_plot_path = _plot_anomaly_type_detection(
        eval_df,
        list(models),
        table_dir / "anomaly_type_detection_rate.csv",
        plot_dir / "anomaly_type_detection_rate.png",
    )
    plot_path = out_dir / "score_distribution.png"
    loss_plot_path = out_dir / "autoencoder_loss.png"
    _plot_scores(test_scores, thresholds, plot_path)
    _plot_losses(ae_losses, loss_plot_path)
    metric_bar_path = plot_dir / "model_metrics.png"
    feature_plot_path = plot_dir / "feature_distribution.png"
    feature_type_plots = _plot_feature_distributions_by_type(combined, feature_type_plot_dir)
    score_label_plot_path = plot_dir / "score_distribution_by_label.png"
    score_label_log_plot_path = plot_dir / "score_distribution_by_label_log_y.png"
    score_label_log_xy_plot_path = plot_dir / "score_distribution_by_label_log_xy.png"
    confusion_plot_path = plot_dir / "confusion_matrix.png"
    roc_plot_path = plot_dir / "roc_curve.png"
    roc_log_plot_path = plot_dir / "roc_curve_log_fpr.png"
    _plot_metric_bars(metrics, metric_bar_path)
    _plot_feature_distributions(combined, feature_plot_path)
    _plot_score_by_label(eval_df, thresholds, score_label_plot_path)
    _plot_score_by_label(eval_df, thresholds, score_label_log_plot_path, log_y=True)
    _plot_score_by_label_log_xy(eval_df, thresholds, score_label_log_xy_plot_path)
    _plot_confusion_matrices(eval_df, list(models), confusion_plot_path)
    _plot_roc_curves(eval_df, list(models), roc_plot_path)
    _plot_roc_curves(eval_df, list(models), roc_log_plot_path, log_x=True)
    run_summary = write_run_summary(out_dir, "anomaly_train", {
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": metrics,
        "synthetic_validation": True,
        "synthetic_rows": int((eval_df["label"] == 1).sum()),
        "data_profile": data_profile,
    })
    return {
        "rows": len(df),
        "metrics": metrics,
        "files": {
            "features": str(features_csv),
            "metrics": str(metrics_path),
            "metrics_table": str(metrics_table_path),
            "synthetic_validation_predictions": str(eval_path),
            "anomaly_type_detection_rate": str(type_rates_path),
            "plot": str(plot_path),
            "autoencoder_loss_plot": str(loss_plot_path),
            "metric_bar_plot": str(metric_bar_path),
            "feature_distribution_plot": str(feature_plot_path),
            "feature_distribution_by_type_plots": feature_type_plots,
            "score_distribution_by_label_plot": str(score_label_plot_path),
            "score_distribution_by_label_log_y_plot": str(score_label_log_plot_path),
            "score_distribution_by_label_log_xy_plot": str(score_label_log_xy_plot_path),
            "confusion_matrix_plot": str(confusion_plot_path),
            "roc_curve_plot": str(roc_plot_path),
            "roc_curve_log_fpr_plot": str(roc_log_plot_path),
            "anomaly_type_detection_rate_plot": str(type_rates_plot_path),
            "plots_dir": str(plot_dir),
            "tables_dir": str(table_dir),
            "training_data_profile": str(table_dir / "training_data_profile.json"),
            "feature_summary": str(table_dir / "feature_summary.csv"),
            "run_summary": str(run_summary),
            **model_files,
        },
    }

def check_anomaly(start: str, end: str, min_models: int = 2, out_dir: Path = ANOMALY_DIR) -> dict:
    models, scaler, thresholds = load_models()
    features_csv = DATA_PROCESSED / "detection_features.csv"
    if not features_csv.exists():
        raise FileNotFoundError("detection_features.csv not found; run train_anomaly_detection first")
    df = pd.read_csv(features_csv, low_memory=False)
    df["time"] = pd.to_datetime(df["time"])
    seg = df[(df["time"] >= pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(end))].copy()
    if seg.empty:
        return {"verdict": "NO_DATA", "summary": f"no running data in {start} ~ {end}"}
    X = scaler.transform(seg[FEATURE_COLS].values)
    result = seg[["time", "state"] + FEATURE_COLS].copy()
    for name, model in models.items():
        scores = _score(name, model, X)
        result[f"score_{name}"] = scores
        result[f"is_anomaly_{name}"] = scores < thresholds[name]
    flag_cols = [c for c in result if c.startswith("is_anomaly_")]
    result["n_models_flagged"] = result[flag_cols].sum(axis=1)
    anomaly_mask = result["n_models_flagged"] >= min_models
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "last_anomaly_check.csv"
    result.to_csv(detail_path, index=False)
    timeline_path = _plot_check_timeline(result, anomaly_mask, plot_dir / "last_anomaly_check_timeline.png")
    votes = {col.replace("is_anomaly_", ""): int(result[col].sum()) for col in flag_cols}
    score_plot_path = _plot_check_scores(result, thresholds, plot_dir / "last_anomaly_check_scores.png")
    vote_plot_path = _plot_check_votes(votes, len(result), plot_dir / "last_anomaly_check_votes.png")
    return {
        "verdict": "ANOMALY" if bool(anomaly_mask.any()) else "NORMAL",
        "rows": int(len(result)),
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_rate": float(anomaly_mask.mean()),
        "model_votes": votes,
        "detail_csv": str(detail_path),
        "timeline_plot": str(timeline_path),
        "score_plot": str(score_plot_path),
        "vote_plot": str(vote_plot_path),
    }
