# CNC MCP 單工具驗證目前彙整（2026-05-25）

## Purpose
彙整目前可使用的 CNC MCP 單工具驗證資料，供後續論文圖表與比較使用。

## Scope
- 納入工具：前 7 題 CNC MCP 單工具。
- 排除工具：`tool_run_all`，它屬於完整流程/工作流驗證。
- 舊資料來源：`outputs/論文驗證/ago/CNC_MCP單工具驗證/all_models_20260525/cnc_mcp_tool_runs.csv`。
- 手動 UI 來源：`Cody-s_Agent_System/data/agent.db` 中 `user_id=SingleTool` 的模型命名 group。

## Contract
- `cnc_mcp_tool_runs.csv`：逐題逐模型資料。
- `summary_by_model.csv`：依模型彙整成功率、平均時間、工具事件數與估算 token。
- `summary_by_tool.csv`：依工具彙整成功率與平均時間。
- `model_tool_success_matrix.csv`：模型 x 工具成功矩陣。
- `manual_ui_session_events.csv`：手動 UI 測試抽出的原始事件。
- `models/<model>/cnc_mcp_tool_runs.csv`：每個模型獨立資料。

## Rules
- 不重跑模型。
- 不覆蓋舊驗證來源。
- 若同一題有多個 session，代表列使用最新 session，原始事件全保留。
- 人工確認失敗項目：`nemotron-3-nano:latest` 第 5 題、`gemma4:31b` 第 3 題。

## Acceptance
- 目前模型數：11。
- 每個模型 7 題。
- 預期資料列：77。
