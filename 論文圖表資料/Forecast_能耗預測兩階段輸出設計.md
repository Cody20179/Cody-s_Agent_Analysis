# Forecast 能耗預測兩階段輸出設計

## Purpose

說明能耗預測模組如何區分「模型訓練評估」與「實際未來預測」，並明確分開 `y` 與 `dy` 兩種預測目標。

| 階段 | 問題 | 程式入口 | 輸出位置 |
|---|---|---|---|
| 訓練 | 模型在歷史測試區間預測得好不好？ | `train_forecast()` | `outputs/forecast/training/<target>/` |
| 應用 | 用已訓練模型預測未來 3、7、14、30 天 | `forecast_future()` | `outputs/forecast/application/<target>/` |

## Scope

目前主要整理 BaselineLastWeek、Prophet、XGBoost 與 LightGBM。

| 模型 | 角色 | 是否需要訓練模型檔 |
|---|---|---|
| BaselineLastWeek | 以上週同時間用電增量作為基準線 | 否 |
| Prophet | 時間序列模型，學習日週週期性 | 是 |
| XGBoost | 使用時間、lag、rolling 特徵的梯度提升樹 | 是 |
| LightGBM | 使用時間、lag、rolling 特徵的梯度提升樹 | 是 |

## Contract

### Target 定義

| target | 意義 | 預測方式 | 適合用途 |
|---|---|---|---|
| `dy` | 每分鐘用電增量 | 先預測每分鐘增量，再累加成未來累積用電量 | 較符合耗電增量問題，建議作為主要版本 |
| `y` | 累積用電量表值 | 直接預測累積電表讀值 | 可作為對照，但需注意累積序列通常非平穩 |

### 訓練輸出

| 檔案 | 用途 |
|---|---|
| `outputs/forecast/training/dy/forecast_metrics.json` | `dy` 目標下的模型測試指標 |
| `outputs/forecast/training/dy/training_data_profile.json` | 訓練資料範圍、數量、切分比例與前處理摘要 |
| `outputs/forecast/training/dy/training_input_y.png` | 訓練前累積用電量 `y` 的時間序列樣貌 |
| `outputs/forecast/training/dy/training_input_dy.png` | 訓練前每分鐘用電增量 `dy` 的時間序列樣貌 |
| `outputs/forecast/training/dy/forecast_test_overlay.png` | 測試區間實際值與預測值比較 |
| `outputs/forecast/training/dy/test_actual.csv` | 測試區間真實資料 |
| `outputs/forecast/training/dy/test_forecast_Prophet.csv` | Prophet 測試區間預測 |
| `outputs/forecast/training/dy/test_forecast_XGBoost.csv` | XGBoost 測試區間預測 |
| `outputs/forecast/training/dy/test_forecast_LightGBM.csv` | LightGBM 測試區間預測 |
| `outputs/forecast/training/dy/test_recursive_XGBoost.csv` | XGBoost 長期遞迴壓力測試 |
| `outputs/forecast/training/dy/test_recursive_LightGBM.csv` | LightGBM 長期遞迴壓力測試 |
| `outputs/forecast/training/dy/compare_backtest_mae.png` | 各模型測試 MAE 比較 |
| `outputs/forecast/training/dy/prophet_backtest_grid.png` | Prophet 不同預測天數測試圖 |
| `outputs/forecast/training/dy/xgboost_recursive_test.png` | XGBoost 長期遞迴壓力測試圖 |
| `models/forecast/Prophet_target_dy.json` | Prophet 訓練後模型 |
| `models/forecast/training_config_dy.json` | `dy` 專用訓練設定 |

`target=y` 會輸出到 `outputs/forecast/training/y/` 與 `models/forecast/training_config_y.json`，不會覆蓋 `dy` 結果。

### 應用輸出

| 檔案 | 用途 |
|---|---|
| `outputs/forecast/application/dy/future_forecast_report.json` | 未來 3、7、14、30 天預測摘要 |
| `outputs/forecast/application/dy/future_forecast.png` | 最近 7 天實際值與未來預測曲線 |
| `outputs/forecast/application/dy/future_BaselineLastWeek.csv` | Baseline 未來預測明細 |
| `outputs/forecast/application/dy/future_Prophet.csv` | Prophet 未來預測明細 |
| `outputs/forecast/application/dy/future_XGBoost.csv` | XGBoost 未來預測明細 |
| `outputs/forecast/application/dy/future_LightGBM.csv` | LightGBM 未來預測明細 |
| `outputs/forecast/application/dy/Prophet_forecast.png` | Prophet 單模型未來預測圖 |
| `outputs/forecast/application/dy/XGBoost_forecast.png` | XGBoost 單模型未來預測圖 |
| `outputs/forecast/application/dy/all_models_forecast.png` | 全模型未來預測比較圖 |

## Rules

- `train_forecast(target="dy")` 只回答歷史資料切分後的訓練與測試表現。
- `forecast_future(target="dy")` 只回答使用既有模型對最新資料往後推估的結果。
- BaselineLastWeek 是比較基準，不是機器學習模型。
- Prophet 需要先訓練，之後可直接載入 `models/forecast/Prophet_target_<target>.json` 做未來預測。
- XGBoost 與 LightGBM 需要 lag 與 rolling 特徵，訓練驗證以 one-step lag validation 為主。
- XGBoost 與 LightGBM 若要做長期未來預測，需另看 recursive stress test，因為遞迴預測會把前一步誤差帶入下一步。
- 未來預測圖使用約 3:1 的歷史與預測時間比例；若預測 30 天，圖中約呈現最近 90 天實際資料。
- `dy` 與 `y` 必須分開報告，不能混在同一張表。
- 論文主要建議使用 `dy`，因為它代表每分鐘耗電增量，比累積表值 `y` 更接近能耗變化。

## Direct Horizon 模型比較修正版

原本的 XGBoost/LightGBM 是 recursive forecast：先預測下一分鐘，再把預測值放回 lag buffer 繼續往後推。當最新資料剛好停機或 `dy` 長時間為 0 時，模型容易掉入近零增量狀態，造成未來曲線接近平線。

為避免和既有 recursive 結果混在一起，修正版獨立輸出於：

| 項目 | 位置 |
|---|---|
| Direct horizon 結果資料夾 | `outputs/forecast/direct_trees/<target>/` |
| Direct horizon 模型資料夾 | `models/forecast/direct_trees/<target>/` |
| MCP 工具 | `tool_train_direct_tree_forecast` |
| Python 入口 | `train_direct_tree_forecast()` |

修正版以 `dy` 為主要目標，並把 Baseline、Prophet、XGBoost、LightGBM 放在同一個 1 至 30 天 horizon 比較流程中。XGBoost 與 LightGBM 不預測每一分鐘再遞迴，而是直接訓練每日 horizon 的監督式模型：

```text
目前時間 t 的 lag / rolling / time features
→ 直接預測 t+1d、t+2d、...、t+30d 的累積用電量或增量
```

每一個 horizon 都是獨立訓練的模型，因此不會把第 1 天的預測值再餵回去推第 2 天，可避免 recursive forecast 常見的誤差累積與長期平線化問題。

比較模型：

| 模型 | 比較方式 |
|---|---|
| BaselineLastWeek | 上週同時段 `dy` 累加作為基準 |
| Prophet | 以訓練集擬合 `dy` 時序，再換算 1 至 30 天累積用電 |
| XGBoost | Direct horizon supervised regression |
| LightGBM | Direct horizon supervised regression |

輸出檔案：

| 檔案 | 用途 |
|---|---|
| `direct_tree_metrics.json` | 四個模型在 1 至 30 天 horizon 的比較指標 |
| `backtests/` | 各模型、各 horizon 的驗證 CSV |
| `future/` | 各模型未來 1 至 30 天每日預測 CSV |
| `plots/direct_horizon_mae.png` | 1 至 30 天 horizon 的 MAE 變化 |
| `plots/<model>_direct_forecast.png` | 各模型 direct horizon 預測圖 |
| `plots/all_models_direct_forecast.png` | 四模型 direct horizon 預測比較圖 |

建議執行：

```python
from main import train_direct_tree_forecast

train_direct_tree_forecast(target="dy")
```

若要指定少數 horizon 作為摘要圖，可以另外傳入：

```python
train_direct_tree_forecast(target="dy", days=[3, 7, 14, 30])
```

論文應將 recursive 版本與 direct horizon 修正版分開討論。recursive 版本用於說明部署風險，direct horizon 版本才是 XGBoost/LightGBM 較合理的長期預測形式。

## 資料內容

目前訓練資料由 `Electricity_consumption.csv` 建立，處理流程如下：

| 項目 | 設定 |
|---|---|
| 原始欄位 | `svc_recv_ts_datetime`、`v1` |
| 時間欄位 | 轉為 `ds` |
| 累積用電量 | `y` |
| 每分鐘增量 | `dy = diff(y)`，負值裁為 0 |
| 重採樣 | 1 分鐘 |
| 缺值處理 | forward fill |
| 極端值處理 | `dy <= Q3 + 3 * IQR` |
| 切分方式 | 時間序列 80% 訓練、20% 測試 |

最新數量與範圍以 `outputs/forecast/training/<target>/training_data_profile.json` 為準。該檔包含：

| 欄位 | 意義 |
|---|---|
| `rows_total` | 前處理後總筆數 |
| `rows_train` | 訓練筆數 |
| `rows_test` | 測試筆數 |
| `train_ratio`、`test_ratio` | 訓練與測試比例 |
| `data_start`、`data_end` | 全資料時間範圍 |
| `train_start`、`train_end` | 訓練資料時間範圍 |
| `test_start`、`test_end` | 測試資料時間範圍 |
| `time_gap_count_after_filtering` | 極端值移除後造成的時間缺口數 |
| `dy_mean`、`dy_q95`、`dy_q99` | 每分鐘增量分布摘要 |

## 建議執行方式

首次建立模型：

```python
from main import train_forecast, forecast_future

train_forecast(target="dy")
forecast_future(target="dy")
```

更新資料後只做未來預測：

```python
from main import update_data, forecast_future

update_data()
forecast_future(target="dy")
```

若要比較 `y` 與 `dy`：

```python
train_forecast(target="dy")
forecast_future(target="dy")

train_forecast(target="y")
forecast_future(target="y")
```

## 論文寫法

方法章可寫：

> 本研究之能耗預測以累積用電量資料為基礎，並建立兩種預測目標。第一種為累積用電量 `y`，第二種為每分鐘用電增量 `dy`。其中 `dy` 由相鄰時間點累積用電量差分取得，並將負值裁切為 0，以避免電表回跳或資料雜訊造成不合理的負耗電量。模型訓練階段使用歷史資料切分為訓練集與測試集，並比較 BaselineLastWeek、Prophet、XGBoost 與 LightGBM 四種方法。

結果章可寫：

> 訓練完成後，本研究將各模型套用於測試區間以評估預測能力。BaselineLastWeek 提供可解釋的週期性基準，Prophet 用於捕捉日週季節性，而 XGBoost 與 LightGBM 則作為 lag 與 rolling 特徵模型之比較。由於樹模型屬於監督式 one-step lag 預測，結果章需同時呈現 one-step validation 與 recursive stress test，以區分模型對已知歷史 lag 的擬合能力，以及在長期未來預測時誤差累積的部署風險。

## 驗證指標

| 指標 | 意義 |
|---|---|
| MAE | 平均絕對誤差，越低越好 |
| RMSE | 對大誤差較敏感，越低越好 |
| R2 | 解釋能力，越高越好；若小於 0，代表比平均值基準還差 |
| MAE percent | MAE 相對於平均實際值的比例，方便論文解釋 |

## Acceptance

論文建議至少放：

| 編號 | 圖表名稱 | 來源 |
|---|---|---|
| 圖 3-x | 能耗預測訓練流程圖 | 本文件流程重畫 |
| 表 3-x | `y` 與 `dy` 目標定義表 | 本文件 Target 定義 |
| 圖 3-x | Forecast 原始累積用電量圖 | `outputs/forecast/training/dy/training_input_y.png` |
| 圖 3-x | Forecast 原始每分鐘耗電增量圖 | `outputs/forecast/training/dy/training_input_dy.png` |
| 表 3-x | Forecast 訓練資料摘要 | `outputs/forecast/training/dy/training_data_profile.json` |
| 表 4-x | 四模型測試指標 | `outputs/forecast/training/dy/forecast_metrics.json` |
| 圖 4-x | 測試區間預測比較圖 | `outputs/forecast/training/dy/forecast_test_overlay.png` |
| 圖 4-x | 未來能耗預測圖 | `outputs/forecast/application/dy/future_forecast.png` |
| 表 4-x | 未來 3、7、14、30 天預測摘要 | `outputs/forecast/application/dy/future_forecast_report.json` |

XGBoost / LightGBM 不是模型完全爛掉，而是之前把它們用「7 天自由遞迴」方式評估，誤差會一直累積，所以看起來非常差。修正後我把指標拆成兩種：

one-step lag validation
用真實歷史 lag 特徵驗證模型本身有沒有學到

recursive stress test
用模型自己的前一步預測繼續往後推，測長期部署風險
目前結果：

dy:
XGBoost one-step R2 = 0.9999999
LightGBM one-step R2 = 0.9999999
但 recursive stress test 還是 review

y:
XGBoost / LightGBM one-step 也不好
代表累積值 y 不適合直接給樹模型做外推

所以論文解釋應該是：

dy + XGBoost/LightGBM 可以作為短期 one-step 能耗增量預測模型；
但若要直接做長期未來預測，遞迴誤差累積風險很高。

y + XGBoost/LightGBM 不建議作為主線，因為累積用電量是非平穩序列，樹模型不擅長外推。
我也補了你說的資料內容：

新增輸出：

outputs/forecast/training/dy/training_data_profile.json
outputs/forecast/training/y/training_data_profile.json
裡面包含：

總筆數: 346149
訓練筆數: 276919
測試筆數: 69230
比例: 80% / 20%
資料範圍: 2025-09-15 15:12:00 到 2026-05-16 22:58:00
訓練範圍: 2025-09-15 15:12:00 到 2026-03-29 09:35:00
測試範圍: 2026-03-29 09:36:00 到 2026-05-16 22:58:00
重採樣: 1 min
dy 處理: diff 後負值裁 0
極端值: dy <= Q3 + 3 * IQR
