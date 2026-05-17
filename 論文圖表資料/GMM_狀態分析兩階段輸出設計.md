# GMM 狀態分析兩階段輸出設計

## Purpose

說明本研究如何使用 GMM 將 CNC 機台感測資料轉換為狀態標籤，並嚴格分成兩個階段：

| 階段 | 問題 | 程式入口 | 輸出位置 |
|---|---|---|---|
| 訓練 | GMM 模型怎麼學出狀態分群？ | `train_state_model()` | `outputs/state/training/` |
| 應用 | 訓練好的模型怎麼套到機台資料？ | `apply_state_model()` | `outputs/state/application/` |
| 一鍵執行 | 先訓練再應用 | `run_state_analysis()` | 同上 |

## Scope

本文件對應 `src/state/gmm.py`。MCP 原本的 `tool_run_state_analysis(start, end)` 保留，另外新增：

| MCP tool | 用途 |
|---|---|
| `tool_train_state_model` | 只訓練 GMM，輸出模型與訓練圖表 |
| `tool_apply_state_model` | 載入已訓練 GMM，輸出機台狀態結果 |
| `tool_run_state_analysis` | 先訓練再套用，方便完整重跑 |

## Contract

訓練階段會建立並保存：

| 檔案 | 用途 |
|---|---|
| `models/state/gmm_state_model.pkl` | GMM 模型、標準化器、狀態對應表 |
| `models/state/gmm_state_model_metadata.json` | 模型設定摘要 |
| `outputs/state/training/training_metrics.json` | BIC、AIC、Silhouette 等訓練指標 |
| `outputs/state/training/training_input_features.png` | GMM 進模型前的訓練特徵時間序列 |
| `outputs/state/training/training_bic_aic.csv` | K 值模型選擇表 |
| `outputs/state/training/training_cluster_centers.csv` | 各 cluster 中心與對應狀態 |

應用階段會建立：

| 視角 | 位置 | 用途 |
|---|---|---|
| 全資料 | `outputs/state/application/all/` | 九月到目前資料的整體狀態結果 |
| 本月 | `outputs/state/application/month/` | 最新月份每日狀態時數 |
| 本週 | `outputs/state/application/week/` | 最新 7 天狀態變化 |
| 本日 | `outputs/state/application/day/` | 最新一天每小時狀態變化 |
| 標籤資料 | `data/processed/state_labeled.csv` | 每個時間點的狀態標籤 |

## Rules

- GMM 是無監督模型，不會自己知道「真正停機」或「真正運轉」。
- 本研究先讓 GMM 依資料分群，再用負載大小替 cluster 命名。
- 使用特徵為 `I_mean`、`I_imbalance`、`Power`、`PF_abs`、`kVAh_rate`。
- 訓練前會做標準化，避免功率或用電量尺度太大而主導模型。
- 候選 K 值目前為 2 至 4，主要用 BIC 選擇最佳 K。
- cluster 依 `I_mean + Power + kVAh_rate` 由低到高命名為停機、待機、低負載運轉、高負載運轉。
- 極低電流且極低功率會用規則校正為 `Off`。
- `gmm_confidence` 是該筆資料最大後驗機率，可視為 GMM 對狀態判定的信心度。

## GMM 到底在做什麼

GMM，全名 Gaussian Mixture Model，中文可寫作高斯混合模型。它的假設是：目前看到的機台資料不是單一分布，而是由多個不同狀態混在一起產生。例如停機時電流與功率都低，運轉時電流與功率較高，待機可能介於兩者之間。

GMM 會把每一筆 5 分鐘資料看成一個特徵點：

```text
x = [I_mean, I_imbalance, Power, PF_abs, kVAh_rate]
```

模型會估計多個高斯分布：

```text
p(x) = sum_k pi_k N(x | mu_k, Sigma_k)
```

其中 `mu_k` 是第 k 群的中心，`Sigma_k` 是該群的形狀，`pi_k` 是該群出現比例。訓練完成後，每筆資料會得到：

| 欄位 | 意義 |
|---|---|
| `cluster` | GMM 判斷它屬於第幾群 |
| `state` | 將 cluster 轉成可讀的機台狀態 |
| `gmm_confidence` | 模型對這次分派的信心度 |

所以本研究確實是使用一個叫做 GMM 的模型；它的角色是從沒有人工標籤的感測資料中找出潛在機台狀態。

## 訓練階段

訓練階段只回答：「這個 GMM 模型是怎麼被建立的，分群是否有足夠依據？」

流程如下：

```text
Raw sensor CSV
→ 讀取 Current_A/B/C、Power、PF、kVAh
→ 統一重採樣為 5 分鐘
→ 建立 GMM 特徵
→ 移除缺值
→ 標準化
→ 訓練 K=2、K=3、K=4
→ 用 BIC 選最佳 K
→ 保存模型與訓練圖表
```

### 訓練圖表解讀

| 圖表 | 來源 | 怎麼看 |
|---|---|---|
| 原始訓練特徵圖 | `training_input_features.png` | 看 GMM 實際吃進去的時間序列資料，包含平均電流、功率、功率因數與 kVAh 變化率。 |
| 特徵分布圖 | `training_feature_distributions.png` | 看資料是否像多種狀態混在一起。若 `I_mean`、`Power` 有長尾或多峰，代表機台不是單一狀態。 |
| BIC/AIC 曲線 | `bic_curve.png` | BIC 最低的 K 是主要選擇。若 K 增加但 BIC 沒明顯下降，代表多分群不值得。 |
| cluster 散點圖 | `training_cluster_scatter.png` | 只把五維模型投影到 `I_mean` 與 `Power`。重疊不一定錯，因為真正分群還用了其他特徵。 |
| 標準化箱型圖 | `training_standardized_feature_boxplot.png` | 確認各特徵已在相近尺度，避免單一大數值特徵壓過其他特徵。 |
| cluster 信心度箱型圖 | `training_cluster_confidence_boxplot.png` | 信心度越接近 1，代表 GMM 越確定該筆資料屬於該群。 |

### 為什麼 scatter 可能看起來沒有完全分開

`training_cluster_scatter.png` 用的是二維圖，但模型實際使用五維特徵。若圖上 cluster 有重疊，不能直接代表 GMM 失敗，原因是：

- 工業機台負載是連續變化，不一定有完美切線。
- 低負載與高負載之間存在過渡區。
- GMM 使用 full covariance，群集可呈現橢圓形，投影到二維時會重疊。
- 分群判斷還包含 `PF_abs`、`kVAh_rate`、`I_imbalance`，不是只看電流和功率。

因此散點圖只能當視覺佐證。真正判斷要搭配 BIC、cluster center、後驗信心度、狀態時間軸與狀態轉移矩陣。

## 應用階段

應用階段只回答：「訓練好的模型套到機台資料後，得到什麼狀態結果？」

流程如下：

```text
Raw sensor CSV or new machine data
→ 使用與訓練相同的前處理
→ 載入 models/state/gmm_state_model.pkl
→ 使用同一個 scaler 標準化
→ GMM 預測 cluster 與 confidence
→ cluster 轉成 state
→ 輸出 state_labeled.csv 與各期間圖表
```

`state_labeled.csv` 是應用階段最核心的資料表。你可以把它理解成：

```text
每 5 分鐘一筆機台資料
每一筆都有 GMM 判斷出的機台狀態
```

常用欄位：

| 欄位 | 用途 |
|---|---|
| `time` | 該筆資料時間 |
| `I_mean` | 三相平均電流 |
| `Power` | 瞬時總功率 |
| `cluster` | GMM 原始群集 |
| `state` | 可解釋的機台狀態 |
| `gmm_confidence` | 狀態判斷信心度 |

## 怎麼知道 GMM 分得好不好

因為目前沒有人工標記的真實狀態，不能直接說 accuracy 幾%。本研究應使用下列量化佐證：

| 驗證指標 | 階段 | 意義 |
|---|---|---|
| BIC | 訓練 | 選擇合理 K 值，越低越好 |
| AIC | 訓練 | 輔助比較模型複雜度，越低越好 |
| Silhouette | 訓練 | 群內越近、群間越遠越好 |
| Calinski-Harabasz | 訓練 | 群間分離相對群內分散，越高越好 |
| Davies-Bouldin | 訓練 | 群間混淆程度，越低越好 |
| cluster center | 訓練 | 各群中心是否符合停機、待機、運轉的物理意義 |
| gmm_confidence | 訓練與應用 | 模型對每筆狀態判斷是否有信心 |
| state timeline | 應用 | 狀態是否隨時間連續，而不是頻繁亂跳 |
| transition matrix | 應用 | 狀態轉移是否符合機台行為 |
| state hours | 應用 | 是否能合理量化每日、每週、每月稼動型態 |

嚴格來說，若要驗證「停機或啟動是否完全正確」，需要人工標籤或現場紀錄做對照。沒有人工標籤時，本研究的論文用語應寫為「狀態辨識結果具備物理可解釋性與統計分群合理性」，不要寫成「已達到監督式分類準確率」。

## Application 圖表分層

為避免長時間圖表過密，應用階段已拆成四個視角：

| 視角 | 圖表 | 適合放在論文哪裡 |
|---|---|---|
| 全資料 | `application/all/all_state_hours.png` | 顯示九月到目前每月狀態時數趨勢 |
| 本月 | `application/month/month_state_hours.png` | 顯示最新月份每日稼動差異 |
| 本週 | `application/week/week_state_hours.png` | 顯示短期運轉模式 |
| 本日 | `application/day/day_state_hours.png` | 顯示最新一天每小時狀態 |

原本 `application_daily_state_hours.png` 太密集，是因為把所有日期都擠在同一張每日長條圖。現在全資料改用「每月彙總」，本月與本週才用每日，本日用每小時，因此比較適合閱讀與放入論文。

## 論文寫法

方法章可寫：

> 本研究在缺乏人工狀態標籤的條件下，採用 Gaussian Mixture Model 進行無監督式機台狀態辨識。首先將三相電流、瞬時總功率、功率因數與視在電能統一重採樣為 5 分鐘資料，並建立平均電流、電流不平衡率、功率、功率因數絕對值與視在電能變化率等特徵。為避免不同量綱影響模型估計，各特徵於訓練前進行標準化。模型候選群數設定為 K=2 至 K=4，並以 BIC 作為主要模型選擇準則。

結果章可寫：

> 完成 GMM 訓練後，本研究將最佳模型套用於完整機台時間序列資料，並根據各 cluster 的平均電流、瞬時功率與視在電能變化率由低至高轉換為停機、待機與運轉狀態。應用結果進一步以狀態占比、狀態時數、狀態時間軸與狀態轉移矩陣呈現，以評估模型對實際機台行為的解釋能力。由於本研究資料未包含人工標記狀態，因此驗證重點並非監督式分類準確率，而是模型選擇指標、後驗信心度、狀態連續性及物理特徵解釋性。

## Acceptance

論文建議至少放：

| 編號 | 圖表名稱 | 來源 |
|---|---|---|
| 圖 3-x | GMM 訓練流程圖 | 本文件流程重畫 |
| 圖 3-x | GMM 原始訓練特徵圖 | `outputs/state/training/training_input_features.png` |
| 圖 3-x | BIC/AIC 模型選擇圖 | `outputs/state/training/bic_curve.png` |
| 圖 3-x | GMM cluster 投影圖 | `outputs/state/training/training_cluster_scatter.png` |
| 表 3-x | GMM 訓練品質指標 | `outputs/state/training/training_metrics.json` |
| 表 3-x | GMM cluster 中心表 | `outputs/state/training/training_cluster_centers.csv` |
| 圖 4-x | 全期間每月狀態時數 | `outputs/state/application/all/all_state_hours.png` |
| 圖 4-x | 本月每日狀態時數 | `outputs/state/application/month/month_state_hours.png` |
| 圖 4-x | 本週狀態時間軸 | `outputs/state/application/week/week_state_timeline.png` |
| 圖 4-x | 本日每小時狀態時數 | `outputs/state/application/day/day_state_hours.png` |
| 表 4-x | 狀態占比表 | `outputs/state/application/all/all_state_summary.csv` |
| 表 4-x | 狀態轉移矩陣 | `outputs/state/application/all/all_transition_matrix.csv` |
