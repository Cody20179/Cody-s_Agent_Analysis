# GMM 狀態分析兩階段輸出設計

## Purpose

建立可放入論文的 GMM 狀態分析輸出架構，並嚴格區分：

- 階段一：GMM 訓練與模型選擇
- 階段二：訓練後模型應用於機台時間序列資料

## Scope

本設計對應 `src/state/gmm.py` 的 `run_state_analysis()`，不更動 MCP tool 名稱與既有主回傳欄位。既有 MCP 呼叫 `tool_run_state_analysis(start, end)` 可繼續使用。

## Contract

執行 `run_state_analysis()` 後，除原本輸出外，新增兩個子資料夾：

| 階段 | 輸出資料夾 | 用途 |
|---|---|---|
| 訓練階段 | `outputs/state/training/` | 說明資料品質、特徵分布、GMM 模型選擇與聚類品質 |
| 應用階段 | `outputs/state/application/` | 說明模型套用到機台資料後的狀態占比、轉移、每日時數與信心度 |

## Rules

- 訓練階段只回答「模型如何被訓練與選出」。
- 應用階段只回答「訓練完模型如何解釋機台狀態」。
- `best_k` 使用 BIC 最小化選擇。
- 狀態命名依 cluster 的 `I_mean + Power + kVAh_rate` 由低到高排序。
- `Off` 狀態另以低電流、低功率規則校正。
- `gmm_confidence` 為每筆資料最大後驗機率，可用於量化分類可靠度。

## Acceptance

論文至少放入下列圖表：

| 編號 | 圖表名稱 | 來源檔案 | 對應階段 |
|---|---|---|---|
| 圖 3-x | GMM 特徵分布圖 | `outputs/state/training/training_feature_distributions.png` | 訓練 |
| 圖 3-x | BIC/AIC 模型選擇圖 | `outputs/state/bic_curve.png` | 訓練 |
| 圖 3-x | GMM cluster 分布圖 | `outputs/state/training/training_cluster_scatter.png` | 訓練 |
| 表 3-x | GMM 訓練資料品質表 | `outputs/state/training/training_feature_quality.csv` | 訓練 |
| 表 3-x | GMM cluster 中心表 | `outputs/state/training/training_cluster_centers.csv` | 訓練 |
| 表 3-x | GMM 訓練品質指標表 | `outputs/state/training/training_metrics.json` | 訓練 |
| 圖 4-x | 機台狀態時間軸 | `outputs/state/state_timeline.png` | 應用 |
| 圖 4-x | 每日狀態時數堆疊圖 | `outputs/state/application/application_daily_state_hours.png` | 應用 |
| 圖 4-x | 狀態轉移機率圖 | `outputs/state/application/application_transition_matrix.png` | 應用 |
| 圖 4-x | GMM 分派信心度分布 | `outputs/state/application/application_confidence_distribution.png` | 應用 |
| 表 4-x | 機台狀態占比表 | `outputs/state/application/application_state_summary.csv` | 應用 |
| 表 4-x | 狀態轉移矩陣 | `outputs/state/application/application_transition_matrix.csv` | 應用 |
| 表 4-x | 每日狀態時數表 | `outputs/state/application/application_daily_state_hours.csv` | 應用 |

## 論文正文建議

### 階段一：GMM 訓練與模型選擇

本研究首先將三相電流、瞬時總功率、功率因數與視在電能統一重採樣為 5 分鐘時間尺度，並建立 `I_mean`、`I_imbalance`、`Power`、`PF_abs` 與 `kVAh_rate` 五項特徵。為避免不同量綱主導模型估計，訓練前以標準化轉換將各特徵轉換為零均值與單位變異數。GMM 模型候選群數以 K=2 至 K=4 進行比較，並以 BIC 作為主要模型選擇準則，AIC 作為輔助參考。

訓練階段需要報告的量化指標包括 BIC、AIC、Silhouette、Calinski-Harabasz 與 Davies-Bouldin 指標。其中 BIC/AIC 用於模型複雜度選擇，Silhouette 用於衡量群內凝聚與群間分離，Calinski-Harabasz 衡量群間離散相對於群內離散的比例，Davies-Bouldin 則以較低值代表較佳分群結構。

### 階段二：模型應用於機台狀態辨識

完成模型選擇後，本研究將最佳 GMM 模型套用於完整機台時間序列資料，並依各群集之 `I_mean`、`Power` 與 `kVAh_rate` 綜合負載分數由低至高對應為停機、待機與不同負載程度之運轉狀態。每一筆資料同時計算最大後驗機率作為狀態分派信心度，並以狀態占比、每日狀態時數與狀態轉移矩陣評估模型在實際機台資料上的解釋能力。

應用階段的結果不再討論模型如何被訓練，而是呈現訓練後模型對機台運轉行為的量化描述，包括各狀態累積時數、百分比、平均電流、平均功率、每日狀態變化與狀態轉移機率。此區分可避免方法章與結果章混淆，並使 GMM 模型同時具備訓練可驗證性與現場應用解釋性。

