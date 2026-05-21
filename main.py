from __future__ import annotations

from src.common import err, ok
from src.config import ANOMALY_DIR, DATA_RAW, FORECAST_DIR, STATE_DIR, ensure_dirs
from src.data.update import status as _data_status
from src.data.update import update as _update_data
from src.scenarios.registry import list_scenarios as _list_scenarios
from src.scenarios.registry import route_scenario as _route_scenario
from src.scenarios.workflows import run_scenario as _run_scenario
from src.validation.runner import run_validation as _run_validation

ensure_dirs()

def update_data(sensors="all", verbose: bool = True) -> dict:
    try:
        result = _update_data(sensors=sensors, raw_dir=DATA_RAW, verbose=verbose)
        return ok(DATA_RAW, "data update finished", metrics=result)
    except Exception as exc:
        return err(str(exc))

def data_status() -> dict:
    try:
        rows = _data_status(DATA_RAW)
        return ok(DATA_RAW, "data status loaded", metrics={"sensors": rows})
    except Exception as exc:
        return err(str(exc))

def run_state_analysis(start=None, end=None) -> dict:
    try:
        from src.state.gmm import run_state_analysis as _run_state_analysis
        result = _run_state_analysis(start=start, end=end)
        return ok(STATE_DIR, "GMM state analysis finished", key_files=result["files"], metrics={
            "best_k": result["best_k"],
            "rows": result["rows"],
            "state_summary": result["state_summary"],
            "training": result.get("training", {}),
            "application": result.get("application", {}),
        })
    except Exception as exc:
        return err(str(exc))

def train_state_model(start=None, end=None) -> dict:
    try:
        from src.state.gmm import train_state_model as _train_state_model
        result = _train_state_model(start=start, end=end)
        return ok(STATE_DIR, "GMM state model training finished", key_files=result["files"], metrics={
            "best_k": result["best_k"],
            "rows": result["rows"],
            "training": result["training"],
        })
    except Exception as exc:
        return err(str(exc))

def apply_state_model(start=None, end=None) -> dict:
    try:
        from src.state.gmm import apply_state_model as _apply_state_model
        result = _apply_state_model(start=start, end=end)
        return ok(STATE_DIR, "GMM state model application finished", key_files=result["files"], metrics={
            "best_k": result["best_k"],
            "rows": result["rows"],
            "state_summary": result["state_summary"],
            "application": result["application"],
        })
    except Exception as exc:
        return err(str(exc))

def train_forecast(target: str = "dy", models: list[str] | None = None, months_back: int | None = None) -> dict:
    try:
        from src.forecast.pipeline import train_forecast as _train_forecast
        result = _train_forecast(target=target, models=models, months_back=months_back)
        return ok(FORECAST_DIR, "forecast training finished", key_files=result["files"], metrics=result["metrics"])
    except Exception as exc:
        return err(str(exc))

def forecast_future(days: list[int] | None = None, model_names: list[str] | None = None, target: str | None = None) -> dict:
    try:
        from src.forecast.pipeline import forecast_future as _forecast_future
        result = _forecast_future(days=days, model_names=model_names, target=target)
        return ok(FORECAST_DIR, "future forecast finished", key_files=result["files"], metrics=result["summary"])
    except Exception as exc:
        return err(str(exc))

def train_direct_tree_forecast(target: str = "dy", days: list[int] | None = None, models: list[str] | None = None, months_back: int | None = None) -> dict:
    try:
        from src.forecast.pipeline import train_direct_tree_forecast as _train_direct_tree_forecast
        result = _train_direct_tree_forecast(target=target, days=days, models=models, months_back=months_back)
        return ok(FORECAST_DIR, "direct tree forecast finished", key_files=result["files"], metrics=result["metrics"])
    except Exception as exc:
        return err(str(exc))

def train_anomaly_detection() -> dict:
    try:
        from src.anomaly.pipeline import train_anomaly_detection as _train_anomaly_detection
        result = _train_anomaly_detection()
        return ok(ANOMALY_DIR, "anomaly detection training finished", key_files=result["files"], metrics=result["metrics"])
    except Exception as exc:
        return err(str(exc))

def check_anomaly(start: str, end: str, min_models: int = 2) -> dict:
    try:
        from src.anomaly.pipeline import check_anomaly as _check_anomaly
        result = _check_anomaly(start, end, min_models=min_models)
        return ok(ANOMALY_DIR, "anomaly check finished", key_files={
            "detail_csv": result.get("detail_csv"),
            "timeline_plot": result.get("timeline_plot"),
            "score_plot": result.get("score_plot"),
            "vote_plot": result.get("vote_plot"),
        }, metrics=result)
    except Exception as exc:
        return err(str(exc))

def list_scenarios() -> dict:
    try:
        return ok(".", "scenario list loaded", metrics={"scenarios": _list_scenarios()})
    except Exception as exc:
        return err(str(exc))

def route_scenario(text: str) -> dict:
    try:
        return ok(".", "scenario routed", metrics=_route_scenario(text))
    except Exception as exc:
        return err(str(exc))

def run_scenario(scenario_id: str, **kwargs) -> dict:
    return _run_scenario(scenario_id, **kwargs)

def validate_scenarios(scenario_ids: list[str] | None = None, repeat: int = 1, model_name: str = "workflow-baseline") -> dict:
    try:
        from src.scenarios.registry import SCENARIOS
        result = _run_validation(scenario_ids or list(SCENARIOS), repeat=repeat, model_name=model_name)
        return ok(result["json"], "scenario validation finished", key_files={"json": result["json"], "csv": result["csv"]}, metrics={
            "validation_run_id": result["records"][0]["validation_run_id"] if result["records"] else None,
            "records": result["records"],
        })
    except Exception as exc:
        return err(str(exc))

def run_all() -> dict:
    steps = {
        "data_status": data_status(),
        "state": run_state_analysis(),
        "forecast": train_forecast(target="dy", models=["BaselineLastWeek", "Prophet"]),
        "forecast_future": forecast_future(days=[3, 7, 14, 30], model_names=["Prophet"]),
        "anomaly": train_anomaly_detection(),
    }
    failed = {name: result for name, result in steps.items() if result.get("status") != "ok"}
    return ok(".", "full pipeline finished" if not failed else "full pipeline finished with errors", metrics=steps)

if __name__ == "__main__":
    import json
    print(json.dumps(data_status(), ensure_ascii=False, indent=2))
