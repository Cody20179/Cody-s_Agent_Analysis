# Forecast 能耗預測兩階段輸出設計

## Purpose

說明能耗預測模組如何區分「模型訓練評估」與「實際未來預測」，並明確分開 `y` 與 `dy` 兩種預測目標。

| 階段 | 問題 | 程式入口 | 輸出位置 |
|---|---|---|---|
| 訓練 | 模型在歷史測試區間預測得好不好？ | `train_forecast()` | `outputs/forecast/training/<target>/` |
| 應用 | 用已訓練模型預測未來 3、7、14、30 天 | `forecast_future()` | `outputs/forecast/application/<target>/` |

## Scope

目前主要整理 BaselineLastWeek 與 Prophet。

| 模型 | 角色 | 是否需要訓練模型檔 |
|---|---|---|
| BaselineLastWeek | 以上週同時間用電增量作為基準線 | 否 |
| Prophet | 時間序列模型，學習日週週期性 | 是 |

XGBoost 與 LightGBM 仍保留為 optional models，但論文若目前只驗證 Baseline 與 Prophet，結果章不應把 XGBoost/LightGBM 寫成主要成果。

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
| `outputs/forecast/training/dy/forecast_test_overlay.png` | 測試區間實際值與預測值比較 |
| `outputs/forecast/training/dy/test_actual.csv` | 測試區間真實資料 |
| `outputs/forecast/training/dy/test_forecast_Prophet.csv` | Prophet 測試區間預測 |
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

## Rules

- `train_forecast(target="dy")` 只回答歷史資料切分後的訓練與測試表現。
- `forecast_future(target="dy")` 只回答使用既有模型對最新資料往後推估的結果。
- BaselineLastWeek 是比較基準，不是機器學習模型。
- Prophet 需要先訓練，之後可直接載入 `models/forecast/Prophet_target_<target>.json` 做未來預測。
- `dy` 與 `y` 必須分開報告，不能混在同一張表。
- 論文主要建議使用 `dy`，因為它代表每分鐘耗電增量，比累積表值 `y` 更接近能耗變化。

## 建議執行方式

首次建立模型：

```python
from main import train_forecast, forecast_future

train_forecast(target="dy", models=["BaselineLastWeek", "Prophet"])
forecast_future(target="dy", model_names=["BaselineLastWeek", "Prophet"])
```

更新資料後只做未來預測：

```python
from main import update_data, forecast_future

update_data()
forecast_future(target="dy", model_names=["BaselineLastWeek", "Prophet"])
```

若要比較 `y` 與 `dy`：

```python
train_forecast(target="dy", models=["BaselineLastWeek", "Prophet"])
forecast_future(target="dy", model_names=["BaselineLastWeek", "Prophet"])

train_forecast(target="y", models=["BaselineLastWeek", "Prophet"])
forecast_future(target="y", model_names=["BaselineLastWeek", "Prophet"])
```

## 論文寫法

方法章可寫：

> 本研究之能耗預測以累積用電量資料為基礎，並建立兩種預測目標。第一種為累積用電量 `y`，第二種為每分鐘用電增量 `dy`。其中 `dy` 由相鄰時間點累積用電量差分取得，並將負值裁切為 0，以避免電表回跳或資料雜訊造成不合理的負耗電量。模型訓練階段使用歷史資料切分為訓練集與測試集，並以 BaselineLastWeek 作為基準模型，Prophet 作為主要時間序列模型。

結果章可寫：

> 訓練完成後，本研究將已訓練之 Prophet 模型套用於最新資料點之後的未來時間區間，並輸出 3、7、14 與 30 天之累積用電量預測。BaselineLastWeek 則用於提供可解釋的比較基準，使模型成果不僅呈現絕對誤差，也能評估 Prophet 是否優於簡單週期性假設。

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
| 表 4-x | Baseline 與 Prophet 測試指標 | `outputs/forecast/training/dy/forecast_metrics.json` |
| 圖 4-x | 測試區間預測比較圖 | `outputs/forecast/training/dy/forecast_test_overlay.png` |
| 圖 4-x | 未來能耗預測圖 | `outputs/forecast/application/dy/future_forecast.png` |
| 表 4-x | 未來 3、7、14、30 天預測摘要 | `outputs/forecast/application/dy/future_forecast_report.json` |
