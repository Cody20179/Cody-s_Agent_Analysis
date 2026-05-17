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
| 圖 3-x | BIC/AIC 模型選擇圖 | `outputs/state/training/bic_curve.png` | 訓練 |
| 圖 3-x | GMM cluster 分布圖 | `outputs/state/training/training_cluster_scatter.png` | 訓練 |
| 圖 3-x | 標準化後特徵箱型圖 | `outputs/state/training/training_standardized_feature_boxplot.png` | 訓練 |
| 圖 3-x | 各 cluster 後驗信心度箱型圖 | `outputs/state/training/training_cluster_confidence_boxplot.png` | 訓練 |
| 表 3-x | GMM 訓練資料品質表 | `outputs/state/training/training_feature_quality.csv` | 訓練 |
| 表 3-x | GMM cluster 中心表 | `outputs/state/training/training_cluster_centers.csv` | 訓練 |
| 表 3-x | GMM 訓練品質指標表 | `outputs/state/training/training_metrics.json` | 訓練 |
| 圖 4-x | 機台狀態時間軸 | `outputs/state/application/state_timeline.png` | 應用 |
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

## 圖表結果解讀

### `training_feature_distributions.png`

此圖呈現訓練用五項特徵的單變量分布。`I_mean` 與 `Power` 通常會呈現偏態與多峰結構，原因是機台同時包含停機、待機與不同負載運轉區間；若資料只屬於單一狀態，分布會較接近單峰。此圖用來佐證 GMM 適合處理多狀態混合分布。

### `bic_curve.png`

此圖比較不同 K 值下的 BIC 與 AIC。BIC 對模型複雜度懲罰較強，因此本研究以 BIC 最小值作為主要模型選擇依據。若 K 增加後 BIC 未明顯下降，代表新增群集無法抵消模型複雜度增加。

### `training_cluster_scatter.png`

此圖以 `I_mean` 與 `Power` 作為二維投影，顯示 GMM 分群結果。它看起來可能不是完全分離的圓形群集，原因有三點：

- GMM 實際訓練使用五維特徵，二維散點圖只是將五維結構投影到 `I_mean` 與 `Power`。
- 工業機台狀態是連續負載變化，不一定形成明確邊界；低負載與中負載之間可能有過渡帶。
- GMM 使用 full covariance，可描述橢圓形與相關性分布，因此群集在二維投影上可能重疊。

因此此圖不應被解讀為唯一分群依據，而是用來觀察主要負載特徵下的可分性。真正的訓練判斷仍需搭配 BIC/AIC、cluster center、後驗信心度與狀態應用結果。

### `training_standardized_feature_boxplot.png`

此圖確認標準化後各特徵位於相近尺度，避免 `Power` 或 `kVAh_rate` 因數值尺度較大而主導 GMM 估計。若標準化後仍出現極端長尾，代表該特徵存在少量高負載或資料跳變，需要在論文中說明這是機台運轉特性或資料品質問題。

### `training_cluster_confidence_boxplot.png`

此圖呈現每個 cluster 的最大後驗機率分布。後驗機率接近 1 代表 GMM 對該樣本的群集歸屬較明確；若某 cluster 的信心度偏低，表示該狀態與其他狀態存在較多重疊，應在結果章中保守解讀。

### `application_confidence_distribution.png`

此圖用於應用階段，呈現完整機台時間序列資料的狀態分派可靠度。若大多數樣本集中在高信心度區間，表示訓練後 GMM 對實際資料具有穩定解釋能力；低於 0.8 的樣本可視為過渡狀態或不確定狀態。

### `application_transition_matrix.png`

此圖呈現狀態間轉移機率，可用於討論機台運轉行為是否合理。例如 `Off` 維持在 `Off` 的機率高，代表停機期間具有時間連續性；`Running_Low` 與 `Running_High` 之間若互相轉移，則可解釋為加工負載變動。

### `application_daily_state_hours.png`

此圖將狀態結果轉為每日時數，適合放在結果章說明機台在不同日期的停機、待機與運轉負載分布。相較單純時間軸圖，此圖更容易量化比較每日稼動情形。

## 訓練流程與學理說明

GMM 假設資料由多個高斯分布混合生成，其機率密度可表示為：

```text
p(x) = sum_k pi_k N(x | mu_k, Sigma_k)
```

其中 `pi_k` 為第 k 個群集的混合權重，`mu_k` 為平均向量，`Sigma_k` 為共變異矩陣。本研究使用 full covariance，使模型可描述不同特徵之間的相關性，例如平均電流與瞬時功率通常具有正相關。

模型訓練使用 EM 演算法。E-step 計算每筆資料屬於各群集的後驗機率，M-step 根據後驗機率更新 `pi_k`、`mu_k` 與 `Sigma_k`。完成訓練後，每筆資料的狀態信心度定義為最大後驗機率：

```text
confidence_i = max_k P(z_i = k | x_i)
```

模型選擇使用 BIC：

```text
BIC = -2 log L + p log n
```

其中 `L` 為模型 likelihood，`p` 為參數數量，`n` 為樣本數。BIC 同時考慮模型配適度與複雜度，因此適合用於避免過度增加 cluster 數量。

訓練階段的驗證邏輯如下：

| 驗證項目 | 目的 |
|---|---|
| 特徵品質表 | 確認缺值比例、分布範圍與特徵尺度 |
| 特徵分布圖 | 確認資料具有多狀態混合分布 |
| 標準化箱型圖 | 確認模型不被單一量綱主導 |
| BIC/AIC | 選擇合理 K 值 |
| cluster center | 說明每群代表的負載狀態 |
| cluster scatter | 以主要負載特徵視覺化分群結構 |
| 後驗信心度 | 量化分群可靠度 |

應用階段的驗證邏輯如下：

| 驗證項目 | 目的 |
|---|---|
| 狀態占比表 | 量化機台停機、待機與運轉比例 |
| 狀態時間軸 | 檢查狀態是否隨時間連續且合理 |
| 每日狀態時數 | 評估每日稼動型態 |
| 狀態轉移矩陣 | 分析機台狀態切換行為 |
| 應用信心度分布 | 評估模型套用於完整資料的可靠性 |
