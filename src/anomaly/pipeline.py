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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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

def train_anomaly_detection(out_dir: Path = ANOMALY_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    combined = pd.concat([
        test_df[FEATURE_COLS].assign(label=0),
        syn[FEATURE_COLS].assign(label=1),
    ], ignore_index=True)
    y_true = combined["label"].values
    X_combined = scaler.transform(combined[FEATURE_COLS].values)
    metrics = {}
    for name, model in models.items():
        scores = _score(name, model, X_combined)
        pred = (scores < thresholds[name]).astype(int)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, -scores)),
            "test_anomaly_count": int((test_scores[name] < thresholds[name]).sum()),
        }

    model_files = _save_models(models, scaler, thresholds)
    metrics_path = write_json(out_dir / "anomaly_metrics.json", metrics)
    plot_path = out_dir / "score_distribution.png"
    _plot_scores(test_scores, thresholds, plot_path)
    run_summary = write_run_summary(out_dir, "anomaly_train", {
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": metrics,
        "synthetic_validation": True,
    })
    return {
        "rows": len(df),
        "metrics": metrics,
        "files": {
            "features": str(features_csv),
            "metrics": str(metrics_path),
            "plot": str(plot_path),
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
    detail_path = out_dir / "last_anomaly_check.csv"
    result.to_csv(detail_path, index=False)
    votes = {col.replace("is_anomaly_", ""): int(result[col].sum()) for col in flag_cols}
    return {
        "verdict": "ANOMALY" if bool(anomaly_mask.any()) else "NORMAL",
        "rows": int(len(result)),
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_rate": float(anomaly_mask.mean()),
        "model_votes": votes,
        "detail_csv": str(detail_path),
    }
