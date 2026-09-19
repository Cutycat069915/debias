# 從 `~/dpo/` 走到現行程式的導讀

寫給熟悉這個目錄、但還沒看過 `~/Augmented_Data/` 的人。

**一句話**：現行的實驗全部在 `~/Augmented_Data/`。這個目錄（`~/dpo/`）是第一代，
程式還在、log 還在，但已經不跑了。兩邊的 **DPO 實作與評估方式都不一樣**，
所以數字不能直接互比——下面說明差在哪、為什麼改。

---

## 一、為什麼有三個目錄

| 目錄 | 時期 | 用途 | 現況 |
|---|---|---|---|
| `~/dpo/` | 第一代 | 原生／合成資料混合比例的 grid search | 停用。`logs/` 還有值（見 §五） |
| `~/BiasUnlearn/` | 第二代 | NPO / unlearning | 停用。是 EMNLP 2025 論文的**第三方 repo**，不是我們寫的 |
| `~/workspace_2/` | 第二代後期 | SFT → DPO 兩階段 | 停用。`New_Datasets/` 有還沒用的評測資料 |
| **`~/Augmented_Data/`** | **現行** | **BBQ 去偏見，9/15 報告的全部數字** | **在用** |

盤點：`~/Agent_Workspace/docs/legacy_inventory.md`

⚠️ `dpo/models/` 與 `Augmented_Data/models/` 是**硬連結**（同 inode，41 G 只佔一次）。
刪這邊不會釋放空間，兩邊都刪就毀了基礎模型。

---

## 二、檔案對照表

| `~/dpo/` | `~/Augmented_Data/` | 差在哪 |
|---|---|---|
| `config.py` + `Parser.py` | `train_dpo.py` 的 `parse_args()` | 沒有集中式 config，參數走命令列 |
| `dpo_utils.py` | （併進 `train_dpo.py`） | |
| **`train_dpo_peft.py`** | **`train_dpo.py`** | **手寫 loss → TRL `DPOTrainer`**（§三①） |
| `Evaluator.py` / `evaluate.py` | `eval_seed.py`＋`eval_control.py`（打分二選一）<br>`eval_bbq_official.py`（**BBQ 官方三選一**）<br>`eval_mcq3.py`（選擇題，含輪替） | **一支變三支**（§三②） |
| `analysis.py` / `check.py` | `aggregate_official.py`、`aggregate_clean.py`、`analyse_2x2.py`、`show_mcq3.py` | |
| `preprocess/synthetic_data.py` | `src/GenerateData.py` + `run_pipeline*.py` | |
| `preprocess/data_splitter.py` | `src/BBQ_Dataset.py` 的 `create_Data()` | |
| `logs/`（`.pt` / `.txt`） | `results/`（`.json`，304 個） | 索引見 `results/MANIFEST.md` |
| `DataSet/`、`Augmented_DataSet/` | `Augmented_BBQ/` | |
| `save/` | `save/` | 同樣是 LoRA adapter |

目錄總覽：`~/Augmented_Data/README_layout.md`

---

## 三、三個關鍵差異

### ① DPO 實作換成 TRL

`dpo/train_dpo_peft.py` 自己實作了 `compute_logprobs()` 與 `dpo_loss()`。
`Augmented_Data/train_dpo.py` 改用 `trl.DPOTrainer`。

**為什麼**：陳老師組（宇辰）用的就是 TRL。要跟他們對數字，得先用同一套訓練器，
否則差異永遠說不清是方法還是實作。對齊之後，明確題兩邊差 1.9 點以內。

`train_dpo.py` 只有 115 行，因為 loss 交給 TRL 了。

### ② ⚠️ 評估方式：正解位置

這一項最重要，因為它會影響**這個目錄裡既有的數字**。

`dpo/evaluate.py:74-122`：

```python
def format_mcq_prompt(sample):
    return (... f"A) {sample['chosen']}\n"        # 正解永遠在 A
                f"B) {sample['rejected']}\n" ...)
...
if pred_text == "A":
    correct += 1
```

**正解固定在 A，選項不輪替。** 所以一個永遠回答 A 的模型會拿到 100%。

實測這個偏差有多大（`Augmented_Data/eval_mcq3.py`，三種擺法輪替）：

| 模型 | 正解固定 A 會評為 | 輪替後的實際值 | 誤差 |
|---|---|---|---|
| Baseline | 74.85% | 84.79% | 低估 9.93 |
| DPO | 99.04% | 94.56% | 高估 4.48 |
| **NPO** | **95.90%** | **52.45%** | **高估 43.45** |

**NPO 看起來跟 DPO 一樣好，實際已經崩潰成幾乎永遠選 A**（82% 的作答選 A，選 C 只有 1%）。

現行程式因此改成三種評估並行，每個結論都用三種問法交叉檢查：

| | 模型輸出什麼 | 主要偏差 |
|---|---|---|
| **打分** `eval_seed.py` | 什麼都不輸出，把答案黏在 prompt 後讀 logprob | 受答案**長度**影響 |
| **官方三選一** `eval_bbq_official.py` | 三個選項各算 logprob 取 argmax | 同上，但至少讓 Unknown 參與競爭 |
| **選擇題** `eval_mcq3.py` | 選項全給，限制輸出一個字母，**三種擺法輪替** | 受**位置**影響，故輪替 |

同一批模型、同一批題目，只換問法，分數可以差 40 個百分點。

### ③ 指標換成 BBQ 官方

第一代量的是「MCQ 正確率 + 語義相似度」。現行版加上 BBQ 原論文（Parrish et al. 2022 §5）
的三選一正確率與 bias score，模糊題／明確題分開算。

**為什麼**：BBQ 的整套設計就建立在「模糊題正解是 Unknown、明確題正解是人名」上。
二選一把明確題的候選限制成「正確人名 vs 錯誤人名」，**從來不讓 Unknown 跟正解競爭**——
所以「模型該答卻拒答」這個錯誤在舊指標下完全看不見。這就是 §四① 那個 bug 藏了那麼久的原因。

---

## 四、要 review 的話，讀這五個地方就夠

按這個順序，約一小時。

### 1. `Augmented_Data/run_pipeline.py:37-38` —— 核心主張

```python
if item['context_condition'] == 'disambig' and rejected_ans.lower() in unknown_variants:
    continue
```

明確題中「錯誤答案是 Unknown」的訓練配對一律刪掉，共 **2,836 筆**。
等於移除了「明確題上說不知道是錯的」這個訊號，一筆不剩。

**要檢查的**：`unknown_variants` 的定義（第 10–13 行）涵蓋得對不對；
以及這兩行是不是真的會刪掉那麼多。

拿掉這兩行的版本是 `run_pipeline_nofilter.py`。

### 2. `Augmented_Data/src/BBQ_Dataset.py` 的 `transform_data()` —— 資料怎麼來

一列 BBQ 變成幾個偏好配對、哪些答案算 rejected，都在這裡決定。
這支是從這個目錄（`~/dpo/`）的邏輯抄過去再改的，應該最眼熟。

### 3. `Augmented_Data/train_dpo.py` —— 訓練

115 行。要看的是 `LoraConfig`（r=16, α=32，`--target_modules` 可切 q,v 或 q,k,v,o）
與 `DPOConfig`（epochs 2、lr 5e-5、batch 2、grad_accum 8、beta）。

### 4. `Augmented_Data/eval_bbq_official.py` —— 官方指標

兩個地方：

- `biased_index()`（第 90 行附近）：判定哪個答案是刻板印象方向。
  依 `question_polarity`：`neg`（問壞事）→ 刻板印象群體；`nonneg`（問好事）→ 另一方。
  **這裡 9/3 才修過一個 bug**：群體標籤對照表原本只涵蓋性別，
  Religion 的 `Muslim`、Age 的 `old` 都對不上，跨類別的 bias score 一律是 0——
  那些 0 是「沒算」不是「沒偏見」。修正後 coverage 由 0% → 100%，
  且性別的判定經回歸檢查完全未變（5,672 列 0 筆差異）。
- `score()`（第 156 行附近）：logprob 的算法，同時輸出加總與每 token 平均兩種。

### 5. `Augmented_Data/eval_mcq3.py:64-70` —— 為什麼舊的位置偏差數字是錯的

```python
for token_str, token_id in tokenizer.get_vocab().items():
    clean = token_str.replace("Ġ", "").replace("▁", "").strip()
    if clean in ("A","a","B","b","C","c"):
        letter_ids["ABC".index(clean.upper())].append(token_id)
```

早期版本讀的是裸 `"A"` 的 token id（32），但模型實際輸出的是帶空格的 `" A"`（362）。
裸 token 的機率是 0.0000——當時等於在把數值雜訊正規化。
修法是掃描整個 vocabulary 收集所有變體。所有 `mcq3_*` 結果都是修正後重跑的。

---

## 五、這個目錄裡還有價值的東西

`logs/`（708 K）。三代實驗下來，**只有這裡的相似度結果在統計上站得住**：

```
n=40 配對    +0.0164    t=12.01    p<1e-6
```

正確率則什麼都沒測到（base 90.48±1.56，所有比較 p>0.3）。
分析寫在 `~/Agent_Workspace/reports/report_0902.md` 附錄 A。

順帶澄清一件事：曾經流傳「ICAT 目標是 0.75」。已查證是誤傳——
`logs/syn1.0/similarity/22.pt[0,1]` 的 **0.6969** 是 `sim(yt, gen_theta)`，
一個語義相似度，跟 ICAT 無關。ICAT 的實作在
`~/BiasUnlearn/Evaluator.py:249-268`，三代下來一次都沒真正跑過。

---

## 六、想親手重現一個數字

例如簡報第 4 頁的 `DPO 明確題 90.92% ± 2.17`：

```bash
docker exec -it Augmented_Data bash
cd /workspace

# 單一 seed，約 4 分鐘
python eval_bbq_official.py \
    Augmented_BBQ/22_clean_mixed \
    save/dpo_clean_22 \
    "DPO" \
    /tmp/check.json \
    Gender_identity

# 五個 seed 的彙整
python aggregate_official.py
```

`results/MANIFEST.md` 有全部 304 個結果檔的對照：
哪個前綴＝哪個 adapter＋哪份訓練資料＋哪個 β＋哪組 LoRA 目標層＋對應報告哪一節。

---

## 七、相關文件

| 文件 | 內容 |
|---|---|
| `~/Agent_Workspace/reports/report_0902.md` | **完整研究報告**（9/15 底稿） |
| `~/Agent_Workspace/reports/xcat_analysis.md` | 跨類別泛化的組成分分析 |
| `~/Agent_Workspace/slides/slides_0915.md` | 簡報（15 主頁 + B1–B8 備用） |
| `~/Augmented_Data/README_layout.md` | 現行目錄與程式總覽 |
| `~/Augmented_Data/results/MANIFEST.md` | 304 個結果檔索引 |
| `~/Agent_Workspace/docs/legacy_inventory.md` | 四個舊目錄的盤點 |
| `~/dpo/README.md` | 這個目錄本身的說明 |
