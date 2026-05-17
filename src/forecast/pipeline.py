from __future__ import annotations

import json
from pathlib import Path

from src.common import write_json, write_run_summary
from src.config import DATA_RAW, FORECAST_DIR, FORECAST_MODELS_DIR
from src.data.loaders import evaluate, load_consumption, split_time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAG_STEPS = [1, 5, 10, 30, 60, 1440, 10080]
ROLLING_WINS = [5, 30, 60]
DEFAULT_MODELS = ["BaselineLastWeek", "Prophet"]
OPTIONAL_MODELS = ["XGBoost", "LightGBM"]

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
        pred = float(predict_fn(np.array([feat], dtype=float))[0])
        if target == "dy":
            pred = max(0.0, pred)
            current_y += pred
            buffer.append(pred)
        else:
            current_y = max(current_y, pred)
            buffer.append(current_y)
        rows.append({"ds": current_time, "yhat": current_y})
    return pd.DataFrame(rows)

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
    X = feat[_feature_cols()].values
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
    return lambda history, periods: _recursive_forecast(model.predict, history, target, periods)

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

    config = {
        "target": target,
        "target_definition": _target_definition(target),
        "models": model_names,
        "default_models": DEFAULT_MODELS,
        "optional_models": OPTIONAL_MODELS,
        "months_back": months_back,
        "data_start": str(df["ds"].min()),
        "data_end": str(df["ds"].max()),
        "train_start": str(train_df["ds"].min()),
        "train_end": str(train_df["ds"].max()),
        "test_start": str(test_df["ds"].min()) if len(test_df) else None,
        "test_end": str(test_df["ds"].max()) if len(test_df) else None,
        "lag_steps": LAG_STEPS,
        "rolling_wins": ROLLING_WINS,
    }
    write_json(_config_file(target), config)
    write_json(FORECAST_MODELS_DIR / "training_config.json", config)

    for name in model_names:
        try:
            if name == "BaselineLastWeek":
                fc = _baseline_last_week(train_df, periods)
            elif name == "Prophet":
                fc = _train_prophet(train_df, target, _model_file(name, target))(train_df, periods)
            elif name in ("XGBoost", "LightGBM"):
                fc = _train_tree(name, train_df, target, _model_file(name, target))(train_df, periods)
            else:
                results[name] = {"status": "skipped", "reason": "unknown model"}
                continue
            merged = actual.merge(fc, on="ds", how="inner")
            metrics = evaluate(merged["y"], merged["yhat"])
            metrics["mae_percent"] = float(metrics["mae"] / merged["y"].mean() * 100) if len(merged) else np.nan
            metrics["deploy_recommendation"] = "review" if metrics["r2"] < 0 else "ok"
            results[name] = {"status": "ok", "target": target, **metrics}
            forecasts[name] = fc
            fc.to_csv(train_dir / f"test_forecast_{name}.csv", index=False)
        except Exception as exc:
            results[name] = {"status": "error", "target": target, "reason": str(exc)}

    actual.to_csv(train_dir / "test_actual.csv", index=False)
    metrics_path = write_json(train_dir / "forecast_metrics.json", results)
    plot_path = train_dir / "forecast_test_overlay.png"
    if forecasts:
        _plot_forecast(actual, forecasts, plot_path, "Forecast recursive test")
    run_summary = write_run_summary(out_dir, f"forecast_train_{target}", {"config": config, "metrics": results})
    return {
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "target": target,
        "metrics": results,
        "files": {
            "metrics": str(metrics_path),
            "forecast_test_overlay": str(plot_path),
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
        except Exception as exc:
            summary[name] = {"status": "error", "target": target, "reason": str(exc)}
    plot_path = app_dir / "future_forecast.png"
    if forecasts:
        _plot_forecast(df.tail(7 * 24 * 60), forecasts, plot_path, "Future forecast")
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
