# `~/dpo/` — 第一代 DPO 專案（學長的原始版本）

> ## 👉 找現在的實驗？看 **[`TRACE_GUIDE.md`](TRACE_GUIDE.md)**
>
> 現行程式全部在 **`~/Augmented_Data/`**。`TRACE_GUIDE.md` 是從這個目錄出發的導讀：
> 檔案對照表、三個關鍵差異（**含一個會影響本目錄既有數字的評估問題**）、
> 以及「要 review 的話讀哪五個檔」。約一小時讀完。

> **狀態：已不再使用。** 現行 pipeline 在 `~/Augmented_Data/`。
> 這份留著是因為 §4 那批 log 是三代實驗裡唯一在統計上站得住的舊結果。
> 整體盤點見 `~/Agent_Workspace/docs/legacy_inventory.md`。

⚠️ **`models/` 與 `~/Augmented_Data/models/` 是硬連結**（同 inode，41 G 只佔一次磁碟）。
刪這邊不會釋放空間；兩邊都刪就毀了基礎模型。**不要碰 `models/`。**

⚠️ git 最後一次 commit 是 `42ea360 "v1"`，另有 **18 個未提交變更**。動之前先 `git status`。

---

## 這一代在做什麼

用**原生 BBQ 資料 + T5 合成資料**以不同比例混合做 DPO，跑 grid search，
看合成資料的比例對去偏見效果的影響。

評估方式與現行版**完全不同**：這一代量的是
**選擇題正確率（MCQ，a/b 二選一）**與**語義相似度**，不是 BBQ 官方三選一。

---

## 程式

| 檔案 | 內容 |
|---|---|
| `config.py` | `TrainConfig` dataclass，集中所有超參數與路徑。含 `build_lora_config()`、`print_summary()` |
| `Parser.py` | `get_args()`，命令列參數 |
| `dpo_utils.py` | `load_base_model()` / `load_tokenizer()` / `parse_save_path()` / `save_hparams()`，訓練與評估共用 |
| `train_dpo_peft.py` | **主訓練程式**。自己手寫 DPO loss（`dpo_loss()`、`compute_logprobs()`），不是用 TRL。`build_mixed_dataset()` 做原生／合成混合，`run_experiment_grid()` 跑參數網格 |
| `train_dpo_peft_v1.py` | 改成吃 `BBQ_Gender_identity_dpo.jsonl` 格式的版本，含 `BBQDPODataset` |
| `Evaluator.py` | 舊版評估。`MCQ_LOGITSPROCESSOR` 限制輸出為 a/b，`eval_multiple_choice()`、`eval_similarities()` |
| `evaluate.py` | `Evaluator.py` 的整理版（`MCQLogitsProcessor`、`save_eval_results()`）。**兩支功能重疊**，以 `evaluate.py` 為準 |
| `analysis.py` | 掃 `logs/*/accuracy/*.txt` 算平均 |
| `check.py` | 掃 `logs/` 算相似度平均，寫死只看 `data1.0` / `data0.7` |

> **注意**：這一代的 DPO loss 是手寫的；現行 `Augmented_Data/train_dpo.py` 用的是
> TRL 的 `DPOTrainer`。兩者的數字不能直接比。

---

## 資料與輸出

| 目錄 | 大小 | 內容 |
|---|---|---|
| `models/` | 41 G | ⚠️ 硬連結，見上。含 `Llama-3.2-11B-Vision-Instruct` 與 `all-MiniLM-L6-v2`（相似度用的 embedding model） |
| `DataSet/` | 81 M | 原生 BBQ |
| `Augmented_DataSet/` | 16 M | T5 合成資料 |
| `save/` | 625 M | LoRA adapter |
| `tmp/` | 3.2 G | 7/30 的備份 checkpoint。**已被完全取代，是這個目錄唯一乾淨的可回收項** |
| `logs/` | 708 K | ⭐ 見下 |
| `preprocess/` | 16 K | 前處理 |

### ⭐ `logs/` —— 唯一還有價值的東西

```
logs/base/  data0.5/  data0.7/  data1.0/  syn0.5/  syn0.7/  syn1.0/
     orig1.0_syn0.0/  orig1.0_syn0.25/   dpo_peft_log.jsonl
```

每個實驗目錄下有 `accuracy/`、`similarity/`、`summary/`。

**三代實驗下來，只有這裡的相似度結果在統計上站得住**：
n=40 配對，+0.0164，t=12.01，p<1e-6。分析見
`~/Agent_Workspace/reports/report_0902.md` 附錄 A。

正確率則什麼都沒測到（base 90.48±1.56，所有比較 p>0.3）。

`logs/syn1.0/similarity/22.pt[0,1]` = **0.6969**，就是曾經被誤傳成
「ICAT 目標 0.75」的那個數字。它實際上是 `sim(yt, gen_theta)`，跟 ICAT 無關。

---

## 如果要重跑（不建議）

沒有留下可直接執行的入口腳本。要跑的話從 `train_dpo_peft.py --help` 開始，
路徑都在 `config.py` 裡。**但現行版在 `Augmented_Data/`，指標也換過了，
重跑這一代得到的數字無法跟現在的報告對接。**
