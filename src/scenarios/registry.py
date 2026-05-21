from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    display_name: str
    category: str
    workflow: str
    examples: tuple[str, ...]
    expected_tools: tuple[str, ...]
    required_outputs: tuple[str, ...]
    optional_inputs: tuple[str, ...] = field(default_factory=tuple)


SCENARIOS: dict[str, Scenario] = {
    "today_status": Scenario(
        scenario_id="today_status",
        display_name="查詢今日機台狀態",
        category="daily",
        workflow="query_today_status",
        examples=("幫我查詢今日機台狀態", "今天機台狀態如何", "今天稼動情況"),
        expected_tools=("tool_apply_state_model",),
        required_outputs=("period", "state_summary", "utilization"),
    ),
    "period_status": Scenario(
        scenario_id="period_status",
        display_name="查詢指定期間機台狀態",
        category="daily",
        workflow="query_status_period",
        examples=("幫我查詢某段期間機台狀態", "查 2026-05-01 到 2026-05-03 的狀態"),
        expected_tools=("tool_apply_state_model",),
        required_outputs=("period", "state_summary", "utilization"),
        optional_inputs=("start", "end"),
    ),
    "last_week_runtime": Scenario(
        scenario_id="last_week_runtime",
        display_name="查詢上週開關機時間",
        category="daily",
        workflow="query_last_week_runtime",
        examples=("幫我看上週機台開機關機時間", "上週什麼時候開機停機"),
        expected_tools=("tool_apply_state_model",),
        required_outputs=("period", "runtime_summary", "utilization"),
    ),
    "utilization": Scenario(
        scenario_id="utilization",
        display_name="查詢機台稼動率",
        category="daily",
        workflow="query_utilization",
        examples=("幫我看機台稼動率", "這段期間稼動率多少"),
        expected_tools=("tool_apply_state_model",),
        required_outputs=("period", "utilization"),
        optional_inputs=("start", "end"),
    ),
    "next_week_power_forecast": Scenario(
        scenario_id="next_week_power_forecast",
        display_name="預測下週耗電",
        category="daily",
        workflow="predict_next_week_power",
        examples=("幫我預測下週耗電多少", "下禮拜用電量預測"),
        expected_tools=("tool_forecast_future",),
        required_outputs=("period", "forecast_summary", "model_names"),
    ),
    "next_month_cost_forecast": Scenario(
        scenario_id="next_month_cost_forecast",
        display_name="預測下個月電費",
        category="daily",
        workflow="predict_next_month_cost",
        examples=("幫我預測下個月電費", "下個月大概電費多少"),
        expected_tools=("tool_forecast_future", "cost_calculation"),
        required_outputs=("period", "forecast_summary", "rate", "estimated_cost"),
        optional_inputs=("rate",),
    ),
    "period_anomaly_check": Scenario(
        scenario_id="period_anomaly_check",
        display_name="檢查指定期間異常",
        category="daily",
        workflow="check_period_anomaly",
        examples=("幫我看這段期間有沒有異常", "檢查 5/1 到 5/2 是否異常"),
        expected_tools=("tool_check_anomaly",),
        required_outputs=("period", "verdict"),
        optional_inputs=("start", "end", "min_models"),
    ),
    "initialize_project": Scenario(
        scenario_id="initialize_project",
        display_name="初始化專案模型",
        category="maintenance",
        workflow="initialize_project",
        examples=("初始化專案", "抓資料並建立初始模型"),
        expected_tools=("tool_update_data", "tool_train_state_model", "tool_train_forecast", "tool_train_anomaly_detection"),
        required_outputs=("initialization", "tool_trace"),
        optional_inputs=("start", "end", "sensors", "fetch_data"),
    ),
    "retrain_all_models": Scenario(
        scenario_id="retrain_all_models",
        display_name="重新訓練全部模型",
        category="maintenance",
        workflow="retrain_all_models",
        examples=("重新訓練全部模型", "更新狀態、預測、異常模型"),
        expected_tools=("tool_train_state_model", "tool_train_forecast", "tool_train_anomaly_detection"),
        required_outputs=("training", "tool_trace"),
        optional_inputs=("start", "end"),
    ),
}


def list_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": item.scenario_id,
            "display_name": item.display_name,
            "category": item.category,
            "examples": list(item.examples),
            "expected_tools": list(item.expected_tools),
            "required_outputs": list(item.required_outputs),
            "optional_inputs": list(item.optional_inputs),
        }
        for item in SCENARIOS.values()
    ]


def route_scenario(text: str) -> dict[str, Any]:
    query = text.lower()
    rules = [
        ("initialize_project", ("初始化", "建置", "初始模型")),
        ("retrain_all_models", ("重新訓練", "重訓", "訓練全部")),
        ("next_month_cost_forecast", ("下個月", "電費")),
        ("next_week_power_forecast", ("下週", "耗電", "用電")),
        ("last_week_runtime", ("上週", "開機", "關機")),
        ("period_anomaly_check", ("異常", "怪怪", "故障")),
        ("utilization", ("稼動率",)),
        ("today_status", ("今日", "今天")),
        ("period_status", ("狀態", "期間")),
    ]
    for scenario_id, keywords in rules:
        if all(keyword.lower() in query for keyword in keywords[:2]) or any(keyword.lower() in query for keyword in keywords):
            scenario = SCENARIOS[scenario_id]
            return {
                "scenario_id": scenario.scenario_id,
                "confidence": 0.75,
                "route": "rule",
                "display_name": scenario.display_name,
            }
    return {
        "scenario_id": "period_status",
        "confidence": 0.25,
        "route": "fallback",
        "display_name": SCENARIOS["period_status"].display_name,
    }
