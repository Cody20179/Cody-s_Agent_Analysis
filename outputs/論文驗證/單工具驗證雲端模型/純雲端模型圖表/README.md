# 單工具驗證純雲端模型圖表

資料來源：`../cnc_tool_runs_detailed.csv`

篩選模型：`glm-5.1`, `kimi-k2.6`, `deepseek-v4-flash`

資料筆數：72 筆 = 3 個雲端模型 x 8 個 CNC 工具 x 3 次 repeat。

## 輸出檔案

| 檔案 | 說明 |
|---|---|
| `cnc_tool_runs_detailed_cloud_only.csv` | 純雲端模型過濾後明細 |
| `model_summary_cloud_only.csv` | 依模型彙整 |
| `tool_summary_cloud_only.csv` | 依 CNC 工具彙整 |
| `model_tool_success_matrix_cloud_only.csv` | 模型 x 工具成功率矩陣 |
| `charts/fig01_cloud_model_success_rate.png` | 雲端模型成功率 |
| `charts/fig02_cloud_model_runtime.png` | 雲端模型平均耗時 |
| `charts/fig03_cloud_tool_success_rate.png` | CNC 工具成功率 |
| `charts/fig04_cloud_tool_runtime.png` | CNC 工具平均耗時 |
| `charts/fig05_cloud_model_tool_success_matrix.png` | 模型與工具成功率矩陣 |
| `charts/fig06_cloud_quality_efficiency.png` | 成功率與耗時比較 |

注意：此資料夾只保留純雲端模型，和上一層 `單工具驗證雲端模型` 的 light combined 原始資料不同。
