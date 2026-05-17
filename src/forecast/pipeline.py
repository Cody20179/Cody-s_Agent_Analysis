from __future__ import annotations

import json
from pathlib import Path

from src.common import write_json, write_run_summary
from src.config import DATA_RAW, FORECAST_DIR, FORECAST_MODELS_DIR
from src.data.loaders import evaluate, load_consumption, split_time

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAG_STEPS = [1, 5, 10, 30, 60, 1440, 10080]
ROLLING_WINS = [5, 30, 60]
DEFAULT_MODELS = ["BaselineLastWeek", "Prophet", "XGBoost", "LightGBM"]
OPTIONAL_MODELS: list[str] = []
DEFAULT_DIRECT_HORIZON_DAYS = list(range(1, 31))
KEY_DIRECT_HORIZON_DAYS = [7, 14, 21, 30]
MODEL_COLORS = {
    "BaselineLastWeek": "tab:gray",
    "Prophet": "tab:blue",
    "XGBoost": "tab:orange",
    "LightGBM": "tab:green",
}

def _periods(days: int) -> int:
    return int(days * 24 * 60)

def _feature_cols() -> list[str]:
    return (
        ["hour", "minute", "day_of_week", "day_of_month", "month", "is_weekend"]
        + [f"lag_{lag}" for lag in LAG_STEPS]
        + [f"roll_{win}" for win in ROLLING_WINS]
    )

def _make_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["ds"].dt.hour
    out["minute"] = out["ds"].dt.minute
    out["day_of_week"] = out["ds"].dt.dayofweek
    out["day_of_month"] = out["ds"].dt.day
    out["month"] = out["ds"].dt.month
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    for lag in LAG_STEPS:
        out[f"lag_{lag}"] = out[target].shift(lag)
    for win in ROLLING_WINS:
        out[f"roll_{win}"] = out[target].shift(1).rolling(win).mean()
    return out.dropna().reset_index(drop=True)

def _recursive_forecast(predict_fn, history: pd.DataFrame, target: str, periods: int) -> pd.DataFrame:
    buffer = list(history[target].values[-(max(LAG_STEPS) + max(ROLLING_WINS)):])
    current_time = history["ds"].iloc[-1]
    current_y = float(history["y"].iloc[-1])
    rows = []
    for _ in range(periods):
        current_time += pd.Timedelta(minutes=1)
        n = len(buffer)
        feat = [
            current_time.hour,
            current_time.minute,
            current_time.dayofweek,
            current_time.day,
            current_time.month,
            int(current_time.dayofweek in (5, 6)),
        ]
        feat += [buffer[n - lag] if n - lag >= 0 else 0.0 for lag in LAG_STEPS]
        feat += [float(np.mean(buffer[n - win:])) if n - win >= 0 else 0.0 for win in ROLLING_WINS]
        X_next = pd.DataFrame([feat], columns=_feature_cols())
        pred = float(predict_fn(X_next)[0])
        if target == "dy":
            pred = max(0.0, pred)
            current_y += pred
            buffer.append(pred)
        else:
            current_y = max(current_y, pred)
            buffer.append(current_y)
        rows.append({"ds": current_time, "yhat": current_y})
    return pd.DataFrame(rows)

def _one_step_tree_forecast(predict_fn, history: pd.DataFrame, test_df: pd.DataFrame, target: str) -> pd.DataFrame:
    combined = pd.concat([history, test_df], ignore_index=True)
    feat = _make_features(combined, target)
    feat = feat[feat["ds"].ge(test_df["ds"].min())].copy()
    pred = predict_fn(feat[_feature_cols()])
    if target == "dy":
        previous_y = feat["y"].shift(1)
        previous_y = previous_y.fillna(float(history["y"].iloc[-1]))
        yhat = previous_y.values + np.clip(pred, 0, None)
    else:
        yhat = pred
    return pd.DataFrame({"ds": feat["ds"].values, "yhat": yhat})

def _baseline_last_week(history: pd.DataFrame, periods: int) -> pd.DataFrame:
    current_y = float(history["y"].iloc[-1])
    current_time = history["ds"].iloc[-1]
    dy_map = history.set_index("ds")["dy"]
    fallback = float(history["dy"].tail(7 * 24 * 60).mean())
    rows = []
    for _ in range(periods):
        current_time += pd.Timedelta(minutes=1)
        ref_time = current_time - pd.Timedelta(days=7)
        dy = float(dy_map.get(ref_time, fallback))
        current_y += max(0.0, dy)
        rows.append({"ds": current_time, "yhat": current_y})
    return pd.DataFrame(rows)

def _train_prophet(train_df: pd.DataFrame, target: str, model_path: Path):
    from prophet import Prophet
    from prophet.serialize import model_to_json

    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(train_df[["ds", target]].rename(columns={target: "y"}))
    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model_to_json(model), f)

    def forecast(history: pd.DataFrame, periods: int) -> pd.DataFrame:
        future_dates = pd.date_range(history["ds"].iloc[-1] + pd.Timedelta(minutes=1), periods=periods, freq="1min")
        fc = model.predict(pd.DataFrame({"ds": future_dates}))
        if target == "dy":
            yhat = float(history["y"].iloc[-1]) + np.cumsum(np.clip(fc["yhat"].values, 0, None))
        else:
            yhat = np.maximum.accumulate(fc["yhat"].values)
        return pd.DataFrame({"ds": future_dates, "yhat": yhat})

    return forecast

def _train_tree(name: str, train_df: pd.DataFrame, target: str, model_path: Path):
    feat = _make_features(train_df, target)
    X = feat[_feature_cols()]
    y = feat[target].values
    if name == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
        model.fit(X, y)
        model.save_model(model_path)
    else:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)
        model.fit(X, y)
        model.booster_.save_model(str(model_path))
    return model.predict

def _load_forecaster(name: str, path: Path, target: str):
    if name == "Prophet":
        from prophet.serialize import model_from_json
        model = model_from_json(json.load(path.open(encoding="utf-8")))

        def forecast(history: pd.DataFrame, periods: int) -> pd.DataFrame:
            future_dates = pd.date_range(history["ds"].iloc[-1] + pd.Timedelta(minutes=1), periods=periods, freq="1min")
            fc = model.predict(pd.DataFrame({"ds": future_dates}))
            if target == "dy":
                yhat = float(history["y"].iloc[-1]) + np.cumsum(np.clip(fc["yhat"].values, 0, None))
            else:
                yhat = np.maximum.accumulate(fc["yhat"].values)
            return pd.DataFrame({"ds": future_dates, "yhat": yhat})

        return forecast
    if name == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(path)
        return lambda history, periods: _recursive_forecast(model.predict, history, target, periods)
    if name == "LightGBM":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path))
        return lambda history, periods: _recursive_forecast(model.predict, history, target, periods)
    raise ValueError(f"unsupported model: {name}")

def _model_file(name: str, target: str) -> Path:
    suffix = {"Prophet": ".json", "XGBoost": ".json", "LightGBM": ".txt"}[name]
    return FORECAST_MODELS_DIR / f"{name}_target_{target}{suffix}"

def _direct_model_file(name: str, target: str, days: int) -> Path:
    suffix = {"XGBoost": ".json", "LightGBM": ".txt"}[name]
    model_dir = FORECAST_MODELS_DIR / "direct_trees" / target
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{name}_direct_{days}d_target_{target}{suffix}"

def _config_file(target: str) -> Path:
    return FORECAST_MODELS_DIR / f"training_config_{target}.json"

def _target_definition(target: str) -> str:
    if target == "dy":
        return "dy: per-minute electricity consumption increment; model predicts future increments and accumulates them into yhat."
    return "y: cumulative electricity consumption; model predicts the cumulative meter value directly."

def _plot_forecast(actual: pd.DataFrame, forecasts: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    if not actual.empty:
        ax.plot(actual["ds"], actual["y"], color="black", lw=1.5, label="Actual")
    for name, fc in forecasts.items():
        ax.plot(fc["ds"], fc["yhat"], lw=1, label=name)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Electricity consumption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_metric_bar(results: dict, metric: str, path: Path, title: str) -> None:
    rows = [
        (name, values.get(metric))
        for name, values in results.items()
        if values.get("status") == "ok" and np.isfinite(values.get(metric, np.nan))
    ]
    if not rows:
        return
    names, values = zip(*rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, values, color=[MODEL_COLORS.get(name, "tab:blue") for name in names])
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_horizon_grid(actual: pd.DataFrame, forecast: pd.DataFrame, path: Path, title: str) -> None:
    horizons = [3, 7, 14, 30]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharey=True)
    axes = axes.ravel()
    for ax, days in zip(axes, horizons):
        end = actual["ds"].min() + pd.Timedelta(days=days)
        actual_slice = actual[actual["ds"].le(end)]
        forecast_slice = forecast[forecast["ds"].le(end)]
        ax.plot(actual_slice["ds"], actual_slice["y"], color="black", lw=1.0, label="Actual")
        ax.plot(forecast_slice["ds"], forecast_slice["yhat"], color="tab:blue", lw=1.0, label="Predicted")
        ax.set_title(f"{days} days")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].legend(loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_future_single(history: pd.DataFrame, forecast: pd.DataFrame, name: str, path: Path, actual_ratio: int = 3) -> None:
    horizon = forecast["ds"].max() - forecast["ds"].min() + pd.Timedelta(minutes=1)
    history_start = history["ds"].max() - horizon * actual_ratio
    hist = history[history["ds"].ge(history_start)].set_index("ds")["y"].resample("1h").mean().dropna().reset_index()
    fc = forecast.set_index("ds")["yhat"].resample("1h").mean().dropna().reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist["ds"], hist["y"], color="black", lw=1.2, label="Actual")
    ax.axvline(history["ds"].max(), color="gray", linestyle="--", lw=1.0, label="Forecast start")
    ax.plot(fc["ds"], fc["yhat"], color=MODEL_COLORS.get(name, "tab:blue"), lw=1.5, label=name)
    ax.set_title(f"{name} future forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Electricity consumption")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_future_all(history: pd.DataFrame, forecasts: dict[str, pd.DataFrame], path: Path, actual_ratio: int = 3) -> None:
    if not forecasts:
        return
    max_end = max(fc["ds"].max() for fc in forecasts.values())
    min_start = min(fc["ds"].min() for fc in forecasts.values())
    horizon = max_end - min_start + pd.Timedelta(minutes=1)
    history_start = history["ds"].max() - horizon * actual_ratio
    hist = history[history["ds"].ge(history_start)].set_index("ds")["y"].resample("1h").mean().dropna().reset_index()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(hist["ds"], hist["y"], color="black", lw=1.3, label="Actual", zorder=10)
    ax.axvline(history["ds"].max(), color="gray", linestyle="--", lw=1.0, label="Forecast start")
    for name, forecast in forecasts.items():
        fc = forecast.set_index("ds")["yhat"].resample("1h").mean().dropna().reset_index()
        ax.plot(fc["ds"], fc["yhat"], lw=1.4, color=MODEL_COLORS.get(name), label=name)
    ax.set_title("Future forecast - all models")
    ax.set_xlabel("Time")
    ax.set_ylabel("Electricity consumption")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_direct_horizon_mae(metrics: dict, path: Path, title: str) -> None:
    rows = []
    for name, horizons in metrics.items():
        for day, values in horizons.items():
            rows.append({"model": name, "horizon": day, "mae": values.get("mae")})
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="horizon", columns="model", values="mae").sort_index()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    if len(pivot) > 10:
        pivot.plot(ax=ax, marker="o", color=[MODEL_COLORS.get(c, "tab:blue") for c in pivot.columns])
    else:
        pivot.plot(kind="bar", ax=ax, color=[MODEL_COLORS.get(c, "tab:blue") for c in pivot.columns])
    ax.set_ylabel("MAE")
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _write_direct_key_horizon_summary(metrics: dict, direct_dir: Path, plot_dir: Path) -> tuple[Path, Path]:
    rows = []
    for day in KEY_DIRECT_HORIZON_DAYS:
        key = f"{day}d"
        for name, horizons in metrics.items():
            values = horizons.get(key, {})
            if values.get("status") != "ok":
                continue
            rows.append({
                "horizon_days": day,
                "model": name,
                "mae": values.get("mae"),
                "rmse": values.get("rmse"),
                "r2": values.get("r2"),
                "mae_percent": values.get("mae_percent"),
                "deploy_recommendation": values.get("deploy_recommendation"),
                "validation_strategy": values.get("validation_strategy"),
            })
    summary = pd.DataFrame(rows)
    csv_path = direct_dir / "direct_horizon_key_metrics.csv"
    summary.to_csv(csv_path, index=False)

    plot_path = plot_dir / "direct_horizon_key_mae.png"
    if not summary.empty:
        pivot = summary.pivot(index="horizon_days", columns="model", values="mae").sort_index()
        fig, ax = plt.subplots(figsize=(11, 5.8))
        pivot.plot(kind="bar", ax=ax, color=[MODEL_COLORS.get(c, "tab:blue") for c in pivot.columns])
        ax.set_title("Key horizon MAE comparison")
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_ylabel("MAE")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=0)
        ax.legend(loc="upper left", ncols=2)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    return csv_path, plot_path

def _plot_direct_future(history: pd.DataFrame, forecast: pd.DataFrame, title: str, path: Path, color: str = "tab:blue", actual_ratio: int = 3) -> None:
    horizon = forecast["ds"].max() - history["ds"].max()
    history_start = history["ds"].max() - horizon * actual_ratio
    hist = history[history["ds"].ge(history_start)].set_index("ds")["y"].resample("1h").mean().dropna().reset_index()
    points = pd.concat([
        pd.DataFrame({"ds": [history["ds"].max()], "yhat": [history["y"].iloc[-1]]}),
        forecast[["ds", "yhat"]],
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist["ds"], hist["y"], color="black", lw=1.2, label="Actual")
    ax.axvline(history["ds"].max(), color="gray", linestyle="--", lw=1.0, label="Forecast start")
    ax.plot(points["ds"], points["yhat"], marker="o", color=color, lw=1.6, label="Direct horizon forecast")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Electricity consumption")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_direct_future_all(history: pd.DataFrame, forecasts: dict[str, pd.DataFrame], path: Path, actual_ratio: int = 3) -> None:
    if not forecasts:
        return
    max_end = max(fc["ds"].max() for fc in forecasts.values())
    horizon = max_end - history["ds"].max()
    history_start = history["ds"].max() - horizon * actual_ratio
    hist = history[history["ds"].ge(history_start)].set_index("ds")["y"].resample("1h").mean().dropna().reset_index()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(hist["ds"], hist["y"], color="black", lw=1.3, label="Actual", zorder=10)
    ax.axvline(history["ds"].max(), color="gray", linestyle="--", lw=1.0, label="Forecast start")
    for name, fc in forecasts.items():
        points = pd.concat([
            pd.DataFrame({"ds": [history["ds"].max()], "yhat": [history["y"].iloc[-1]]}),
            fc[["ds", "yhat"]],
        ], ignore_index=True)
        ax.plot(points["ds"], points["yhat"], marker="o", lw=1.5, color=MODEL_COLORS.get(name), label=name)
    ax.set_title("Direct horizon tree forecast - all models")
    ax.set_xlabel("Time")
    ax.set_ylabel("Electricity consumption")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_training_input_series(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, column: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.plot(df["ds"], df[column], color="black", lw=0.8)
    ax.axvspan(train_df["ds"].min(), train_df["ds"].max(), color="tab:blue", alpha=0.08, label="Train")
    ax.axvspan(test_df["ds"].min(), test_df["ds"].max(), color="tab:orange", alpha=0.10, label="Test")
    ax.set_ylabel(column)
    ax.set_xlabel("Time")
    ax.set_title(f"Forecast raw training input: {column}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _data_profile(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, months_back: int | None, target: str) -> dict:
    gaps = df["ds"].diff().dt.total_seconds().fillna(60)
    dy = df["dy"]
    return {
        "target": target,
        "months_back": months_back,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "train_ratio": round(len(train_df) / max(len(df), 1), 4),
        "test_ratio": round(len(test_df) / max(len(df), 1), 4),
        "data_start": str(df["ds"].min()),
        "data_end": str(df["ds"].max()),
        "train_start": str(train_df["ds"].min()),
        "train_end": str(train_df["ds"].max()),
        "test_start": str(test_df["ds"].min()) if len(test_df) else None,
        "test_end": str(test_df["ds"].max()) if len(test_df) else None,
        "resample_frequency": "1min",
        "outlier_rule": "dy <= Q3 + 3 * IQR after negative dy values clipped to 0",
        "time_gap_count_after_filtering": int(gaps.gt(60).sum()),
        "max_time_gap_minutes": float(gaps.max() / 60),
        "dy_min": float(dy.min()),
        "dy_mean": float(dy.mean()),
        "dy_median": float(dy.median()),
        "dy_q95": float(dy.quantile(0.95)),
        "dy_q99": float(dy.quantile(0.99)),
        "dy_max": float(dy.max()),
        "y_min": float(df["y"].min()),
        "y_max": float(df["y"].max()),
    }

def _make_direct_horizon_frame(df: pd.DataFrame, target: str, days: int) -> pd.DataFrame:
    horizon = _periods(days)
    feat = _make_features(df, target)
    feat["future_y"] = feat["y"].shift(-horizon)
    feat["future_ds"] = feat["ds"].shift(-horizon)
    if target == "dy":
        feat["label"] = feat["future_y"] - feat["y"]
    else:
        feat["label"] = feat["future_y"]
    return feat.dropna(subset=["future_y", "future_ds", "label"]).reset_index(drop=True)

def _fit_direct_tree(name: str, train_frame: pd.DataFrame, target: str, days: int):
    X = train_frame[_feature_cols()]
    y = train_frame["label"].values
    if name == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X, y)
        model.save_model(_direct_model_file(name, target, days))
    elif name == "LightGBM":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X, y)
        model.booster_.save_model(str(_direct_model_file(name, target, days)))
    else:
        raise ValueError(f"unsupported direct tree model: {name}")
    return model

def _baseline_direct_horizon(df: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    source = df.set_index("ds")["dy"].sort_index()
    full_index = pd.date_range(source.index.min(), source.index.max(), freq="1min")
    fallback = float(source.tail(7 * 24 * 60).mean())
    aligned = source.reindex(full_index).fillna(fallback)
    cumsum = aligned.cumsum()
    ref_origin = test_frame["ds"] - pd.Timedelta(days=7)
    ref_end = test_frame["future_ds"] - pd.Timedelta(days=7)
    valid = ref_origin.ge(cumsum.index.min()) & ref_end.le(cumsum.index.max())
    increments = np.full(len(test_frame), fallback * (test_frame["future_ds"] - test_frame["ds"]).dt.total_seconds().to_numpy() / 60)
    if valid.any():
        end_values = cumsum.reindex(ref_end[valid]).to_numpy()
        origin_values = cumsum.reindex(ref_origin[valid]).to_numpy()
        increments[valid.to_numpy()] = end_values - origin_values
    return test_frame["y"].to_numpy() + np.clip(increments, 0, None)

def _prophet_prediction_series(train_df: pd.DataFrame, target: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    from prophet import Prophet

    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(train_df[["ds", target]].rename(columns={target: "y"}))
    future_dates = pd.date_range(start, end, freq="1min")
    fc = model.predict(pd.DataFrame({"ds": future_dates}))[["ds", "yhat"]]
    return pd.Series(fc["yhat"].to_numpy(), index=fc["ds"])

def _prophet_direct_horizon(test_frame: pd.DataFrame, target: str, predicted: pd.Series) -> np.ndarray:
    if target == "dy":
        cumsum = pd.Series(np.clip(predicted.to_numpy(), 0, None), index=predicted.index).cumsum()
        end_values = cumsum.reindex(test_frame["future_ds"]).to_numpy()
        origin_values = cumsum.reindex(test_frame["ds"]).fillna(0).to_numpy()
        return test_frame["y"].to_numpy() + np.clip(end_values - origin_values, 0, None)
    return np.maximum(test_frame["y"].to_numpy(), predicted.reindex(test_frame["future_ds"]).to_numpy())

def train_forecast(target: str = "dy", models: list[str] | None = None, months_back: int | None = None, raw_dir: Path = DATA_RAW, out_dir: Path = FORECAST_DIR) -> dict:
    if target not in ("dy", "y"):
        raise ValueError("target must be 'dy' or 'y'")
    train_dir = out_dir / "training" / target
    train_dir.mkdir(parents=True, exist_ok=True)
    FORECAST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_consumption(raw_dir)
    if months_back:
        df = df[df["ds"] >= df["ds"].max() - pd.DateOffset(months=months_back)].reset_index(drop=True)
    train_df, test_df = split_time(df)
    model_names = models or DEFAULT_MODELS
    periods = min(len(test_df), _periods(7))
    actual = test_df.iloc[:periods].copy()
    results, forecasts = {}, {}
    data_profile = _data_profile(df, train_df, test_df, months_back, target)

    config = {
        "target": target,
        "target_definition": _target_definition(target),
        "models": model_names,
        "default_models": DEFAULT_MODELS,
        "optional_models": OPTIONAL_MODELS,
        "months_back": months_back,
        "data_profile": data_profile,
        "lag_steps": LAG_STEPS,
        "rolling_wins": ROLLING_WINS,
    }
    write_json(_config_file(target), config)
    write_json(FORECAST_MODELS_DIR / "training_config.json", config)

    for name in model_names:
        try:
            if name == "BaselineLastWeek":
                fc = _baseline_last_week(train_df, periods)
                metric_fc = fc
                strategy = "recursive_last_week_increment_baseline"
            elif name == "Prophet":
                fc = _train_prophet(train_df, target, _model_file(name, target))(train_df, periods)
                metric_fc = fc
                strategy = "direct_horizon_time_series"
            elif name in ("XGBoost", "LightGBM"):
                predict_fn = _train_tree(name, train_df, target, _model_file(name, target))
                metric_fc = _one_step_tree_forecast(predict_fn, train_df, actual, target)
                fc = _recursive_forecast(predict_fn, train_df, target, periods)
                recursive_merged = actual.merge(fc, on="ds", how="inner")
                recursive_metrics = evaluate(recursive_merged["y"], recursive_merged["yhat"])
                recursive_metrics["mae_percent"] = float(recursive_metrics["mae"] / recursive_merged["y"].mean() * 100) if len(recursive_merged) else np.nan
                fc.to_csv(train_dir / f"test_recursive_{name}.csv", index=False)
                strategy = "one_step_lag_validation_with_recursive_stress_test"
            else:
                results[name] = {"status": "skipped", "reason": "unknown model"}
                continue
            merged = actual.merge(metric_fc, on="ds", how="inner")
            metrics = evaluate(merged["y"], merged["yhat"])
            metrics["mae_percent"] = float(metrics["mae"] / merged["y"].mean() * 100) if len(merged) else np.nan
            metrics["deploy_recommendation"] = "review" if metrics["r2"] < 0 else "ok"
            result = {"status": "ok", "target": target, "validation_strategy": strategy, **metrics}
            if name in ("XGBoost", "LightGBM"):
                result["recursive_stress_test"] = {
                    **recursive_metrics,
                    "deploy_recommendation": "review" if recursive_metrics["r2"] < 0 else "ok",
                }
            results[name] = result
            forecasts[name] = metric_fc
            metric_fc.to_csv(train_dir / f"test_forecast_{name}.csv", index=False)
        except Exception as exc:
            results[name] = {"status": "error", "target": target, "reason": str(exc)}

    actual.to_csv(train_dir / "test_actual.csv", index=False)
    metrics_path = write_json(train_dir / "forecast_metrics.json", results)
    data_profile_path = write_json(train_dir / "training_data_profile.json", data_profile)
    input_y_plot_path = train_dir / "training_input_y.png"
    input_dy_plot_path = train_dir / "training_input_dy.png"
    _plot_training_input_series(df, train_df, test_df, "y", input_y_plot_path)
    _plot_training_input_series(df, train_df, test_df, "dy", input_dy_plot_path)
    plot_path = train_dir / "forecast_test_overlay.png"
    if forecasts:
        _plot_forecast(actual, forecasts, plot_path, "Forecast recursive test")
        _plot_metric_bar(results, "mae", train_dir / "compare_backtest_mae.png", f"{target} validation MAE by model")
        _plot_metric_bar(results, "r2", train_dir / "compare_backtest_r2.png", f"{target} validation R2 by model")
        for name, fc in forecasts.items():
            _plot_horizon_grid(actual, fc, train_dir / f"{name.lower()}_backtest_grid.png", f"{name} validation by horizon")
        for name in model_names:
            recursive_path = train_dir / f"test_recursive_{name}.csv"
            forecast_path = recursive_path if recursive_path.exists() else train_dir / f"test_forecast_{name}.csv"
            if forecast_path.exists():
                rec_fc = pd.read_csv(forecast_path, parse_dates=["ds"])
                _plot_horizon_grid(actual, rec_fc, train_dir / f"{name.lower()}_recursive_test.png", f"{name} recursive/direct test")
    run_summary = write_run_summary(out_dir, f"forecast_train_{target}", {"config": config, "metrics": results})
    return {
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "target": target,
        "metrics": results,
        "files": {
            "metrics": str(metrics_path),
            "training_data_profile": str(data_profile_path),
            "training_input_y": str(input_y_plot_path),
            "training_input_dy": str(input_dy_plot_path),
            "forecast_test_overlay": str(plot_path),
            "compare_backtest_mae": str(train_dir / "compare_backtest_mae.png"),
            "compare_backtest_r2": str(train_dir / "compare_backtest_r2.png"),
            "test_actual": str(train_dir / "test_actual.csv"),
            "config": str(_config_file(target)),
            "latest_config": str(FORECAST_MODELS_DIR / "training_config.json"),
            "models_dir": str(FORECAST_MODELS_DIR),
            "run_summary": str(run_summary),
        },
    }

def forecast_future(days: list[int] | None = None, model_names: list[str] | None = None, target: str | None = None, raw_dir: Path = DATA_RAW, out_dir: Path = FORECAST_DIR) -> dict:
    days = days or [3, 7, 14, 30]
    max_periods = _periods(max(days))
    if target and target not in ("dy", "y"):
        raise ValueError("target must be 'dy', 'y', or None")
    config_path = _config_file(target) if target else FORECAST_MODELS_DIR / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path.name} not found; run train_forecast first")
    config = json.load(config_path.open(encoding="utf-8"))
    target = config["target"]
    app_dir = out_dir / "application" / target
    app_dir.mkdir(parents=True, exist_ok=True)
    df = load_consumption(raw_dir)
    names = model_names or config["models"]
    forecasts, summary = {}, {}
    for name in names:
        try:
            if name == "BaselineLastWeek":
                fc = _baseline_last_week(df, max_periods)
            else:
                fc = _load_forecaster(name, _model_file(name, target), target)(df, max_periods)
            forecasts[name] = fc
            summary[name] = {}
            for day in days:
                idx = min(_periods(day) - 1, len(fc) - 1)
                summary[name][f"{day}d"] = {
                    "yhat": float(fc["yhat"].iloc[idx]),
                    "increment": float(fc["yhat"].iloc[idx] - df["y"].iloc[-1]),
                }
            fc.to_csv(app_dir / f"future_{name}.csv", index=False)
            _plot_future_single(df, fc, name, app_dir / f"{name}_forecast.png")
        except Exception as exc:
            summary[name] = {"status": "error", "target": target, "reason": str(exc)}
    plot_path = app_dir / "future_forecast.png"
    if forecasts:
        _plot_future_all(df, forecasts, app_dir / "all_models_forecast.png")
        _plot_future_all(df, forecasts, plot_path)
    report_path = write_json(app_dir / "future_forecast_report.json", {
        "target": target,
        "target_definition": _target_definition(target),
        "data_latest": str(df["ds"].iloc[-1]),
        "last_y": float(df["y"].iloc[-1]),
        "days": days,
        "models": summary,
    })
    run_summary = write_run_summary(out_dir, f"forecast_future_{target}", {"summary": summary})
    return {"target": target, "summary": summary, "files": {"report": str(report_path), "future_forecast": str(plot_path), "run_summary": str(run_summary), "application_dir": str(app_dir)}}

def train_direct_tree_forecast(
    target: str = "dy",
    days: list[int] | None = None,
    models: list[str] | None = None,
    months_back: int | None = None,
    raw_dir: Path = DATA_RAW,
    out_dir: Path = FORECAST_DIR,
) -> dict:
    if target not in ("dy", "y"):
        raise ValueError("target must be 'dy' or 'y'")
    days = days or DEFAULT_DIRECT_HORIZON_DAYS
    model_names = models or ["BaselineLastWeek", "Prophet", "XGBoost", "LightGBM"]
    direct_dir = out_dir / "direct_trees" / target
    backtest_dir = direct_dir / "backtests"
    future_dir = direct_dir / "future"
    plot_dir = direct_dir / "plots"
    for path in (direct_dir, backtest_dir, future_dir, plot_dir):
        path.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "*_direct_backtest_*d.csv",
        "future_direct_*.csv",
        "*_direct_forecast.png",
        "all_models_direct_forecast.png",
        "direct_horizon_mae.png",
    ):
        for old_file in direct_dir.glob(pattern):
            if old_file.is_file():
                old_file.unlink()

    df = load_consumption(raw_dir)
    if months_back:
        df = df[df["ds"] >= df["ds"].max() - pd.DateOffset(months=months_back)].reset_index(drop=True)
    train_df, test_df = split_time(df)
    profile = _data_profile(df, train_df, test_df, months_back, target)

    metrics: dict[str, dict] = {name: {} for name in model_names}
    future_forecasts: dict[str, pd.DataFrame] = {}
    latest_feature = _make_features(df, target).tail(1)
    latest_y = float(df["y"].iloc[-1])
    latest_time = df["ds"].iloc[-1]
    prophet_validation = None
    prophet_future = None
    if "Prophet" in model_names:
        prophet_validation = _prophet_prediction_series(
            train_df,
            target,
            test_df["ds"].min() + pd.Timedelta(minutes=1),
            df["ds"].max(),
        )
        prophet_future = _prophet_prediction_series(
            df,
            target,
            latest_time + pd.Timedelta(minutes=1),
            latest_time + pd.Timedelta(days=max(days)),
        )

    for name in model_names:
        future_rows = []
        for day in days:
            full_frame = _make_direct_horizon_frame(df, target, day)
            train_frame = full_frame[full_frame["future_ds"].le(train_df["ds"].max())].copy()
            test_frame = full_frame[full_frame["ds"].ge(test_df["ds"].min())].copy()
            if train_frame.empty or test_frame.empty:
                metrics[name][f"{day}d"] = {"status": "skipped", "reason": "not enough horizon data"}
                continue

            if name == "BaselineLastWeek":
                yhat = _baseline_direct_horizon(df, test_frame)
                validation_strategy = "rolling_origin_last_week_direct_horizon"
            elif name == "Prophet":
                yhat = _prophet_direct_horizon(test_frame, target, prophet_validation)
                validation_strategy = "rolling_origin_prophet_direct_horizon"
            elif name in ("XGBoost", "LightGBM"):
                model = _fit_direct_tree(name, train_frame, target, day)
                pred = model.predict(test_frame[_feature_cols()])
                if target == "dy":
                    yhat = test_frame["y"].values + np.clip(pred, 0, None)
                else:
                    yhat = np.maximum(test_frame["y"].values, pred)
                validation_strategy = "direct_horizon_supervised"
            else:
                metrics[name][f"{day}d"] = {"status": "skipped", "reason": "unknown model"}
                continue
            eval_metrics = evaluate(test_frame["future_y"].values, yhat)
            eval_metrics["mae_percent"] = float(eval_metrics["mae"] / test_frame["future_y"].mean() * 100)
            eval_metrics["rows_train"] = int(len(train_frame))
            eval_metrics["rows_test"] = int(len(test_frame))
            eval_metrics["validation_strategy"] = validation_strategy
            eval_metrics["deploy_recommendation"] = "review" if eval_metrics["r2"] < 0 else "ok"
            metrics[name][f"{day}d"] = {"status": "ok", **eval_metrics}

            test_out = pd.DataFrame({
                "ds": test_frame["ds"].values,
                "future_ds": test_frame["future_ds"].values,
                "actual_y": test_frame["future_y"].values,
                "yhat": yhat,
            })
            test_out.to_csv(backtest_dir / f"{name}_direct_backtest_{day}d.csv", index=False)

            if name == "BaselineLastWeek":
                future_frame = pd.DataFrame({
                    "ds": [latest_time],
                    "future_ds": [latest_time + pd.Timedelta(days=day)],
                    "y": [latest_y],
                })
                future_y = float(_baseline_direct_horizon(df, future_frame)[0])
                increment = future_y - latest_y
            elif name == "Prophet":
                future_frame = pd.DataFrame({
                    "ds": [latest_time],
                    "future_ds": [latest_time + pd.Timedelta(days=day)],
                    "y": [latest_y],
                })
                future_y = float(_prophet_direct_horizon(future_frame, target, prophet_future)[0])
                increment = future_y - latest_y
            else:
                final_frame = _make_direct_horizon_frame(df, target, day)
                final_model = _fit_direct_tree(name, final_frame, target, day)
                future_pred = float(final_model.predict(latest_feature[_feature_cols()])[0])
                if target == "dy":
                    future_y = latest_y + max(0.0, future_pred)
                    increment = max(0.0, future_pred)
                else:
                    future_y = max(latest_y, future_pred)
                    increment = future_y - latest_y
            future_rows.append({
                "horizon_days": day,
                "ds": latest_time + pd.Timedelta(days=day),
                "yhat": future_y,
                "increment": increment,
            })

        if future_rows:
            future_df = pd.DataFrame(future_rows).sort_values("horizon_days")
            future_df.to_csv(future_dir / f"future_direct_{name}.csv", index=False)
            future_forecasts[name] = future_df
            _plot_direct_future(
                df,
                future_df,
                f"{name} direct horizon forecast",
                plot_dir / f"{name}_direct_forecast.png",
                color=MODEL_COLORS.get(name, "tab:blue"),
            )

    metrics_path = write_json(direct_dir / "direct_tree_metrics.json", metrics)
    profile_path = write_json(direct_dir / "direct_tree_data_profile.json", profile)
    config_path = write_json(direct_dir / "direct_tree_config.json", {
        "target": target,
        "target_definition": _target_definition(target),
        "models": model_names,
        "horizon_days": days,
        "default_horizon_days": DEFAULT_DIRECT_HORIZON_DAYS,
        "strategy": "1 to 30 day horizon comparison; tree models use direct supervised horizons; baseline and Prophet use matching horizon validation outputs",
        "data_profile": profile,
        "lag_steps": LAG_STEPS,
        "rolling_wins": ROLLING_WINS,
    })
    mae_plot = plot_dir / "direct_horizon_mae.png"
    _plot_direct_horizon_mae(metrics, mae_plot, f"{target} direct horizon MAE")
    key_metrics_path, key_mae_plot = _write_direct_key_horizon_summary(metrics, direct_dir, plot_dir)
    all_plot = plot_dir / "all_models_direct_forecast.png"
    _plot_direct_future_all(df, future_forecasts, all_plot)
    run_summary = write_run_summary(out_dir, f"forecast_direct_trees_{target}", {"metrics": metrics, "profile": profile})
    return {
        "target": target,
        "metrics": metrics,
        "files": {
            "metrics": str(metrics_path),
            "data_profile": str(profile_path),
            "config": str(config_path),
            "direct_horizon_mae": str(mae_plot),
            "direct_horizon_key_metrics": str(key_metrics_path),
            "direct_horizon_key_mae": str(key_mae_plot),
            "all_models_direct_forecast": str(all_plot),
            "output_dir": str(direct_dir),
            "backtests_dir": str(backtest_dir),
            "future_dir": str(future_dir),
            "plots_dir": str(plot_dir),
            "run_summary": str(run_summary),
        },
    }
