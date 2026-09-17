# DPO_SRC — BBQ 去偏見 DPO 訓練（孫老師實驗室）

用 DPO + LoRA 訓練 Llama-3.2-11B-Vision-Instruct，減少 BBQ Gender_identity 上的偏見。
下面的設定就是我們報告裡的「最終配置」。

---

## 1. 環境

我們用的套件版本如下，其他版本沒有測過：

```
python 3.12    torch 2.9    transformers 5.14.0    trl 1.12.0    peft 0.20.0
datasets 5.0.0    accelerate 1.14.0
```

```bash
pip install -r requirements.txt
```

硬體：我們是在實驗室的 DGX Spark（GB10，119 GB 統一記憶體）上用單卡、bf16 跑的。訓練時會同時載入 11B 模型和一份 reference model，所以 24 GB 的卡放不下。

## 2. 放模型

模型不進 git，請自己放到 `models/` 底下：

```
DPO_SRC/models/Llama-3.2-11B-Vision-Instruct/
```

BBQ 資料讀的是 repo 裡的 `../dataset/Gender_identity.jsonl`，要換位置的話設環境變數 `BBQ_DIR`。

## 3. 執行

以下指令都在 `DPO_SRC/` 底下執行。

```bash
python make_data.py            # 產生 data/<seed>_pure916、data/<seed>_chat916，五個 seed
bash run_final.sh              # 訓練 + 評估，五個 seed（每個約 30 分鐘）
bash run_final.sh 22           # 只跑一個 seed
```

未訓練的 baseline：

```bash
python eval_bbq_official.py data/22_pure916 models/Llama-3.2-11B-Vision-Instruct base results/official_base_22.json Gender_identity --chat
python eval_mcq3.py         data/22_pure916 models/Llama-3.2-11B-Vision-Instruct base results/mcq3_base_22.json --chat
```

## 4. 檔案

| 檔案 | 做什麼 |
|---|---|
| `make_data.py` | BBQ → 偏好配對（chosen / rejected） |
| `train_dpo.py` | DPO + LoRA 訓練，預設值就是最終配置 |
| `eval_bbq_official.py` | **打分**：三個選項接在 prompt 後面，比較長度正規化的 logprob，取最高分。算正確率與 BBQ bias score |
| `eval_mcq3.py` | **選擇題**：讀 A/B/C 三個字母的機率，選項順序輪替三次。算正確率、ECE、位置偏差 |
| `run_final.sh` | 把上面串起來 |

## 5. 設定

| 參數 | 值 |
|---|---|
| base model | Llama-3.2-11B-Vision-Instruct |
| β | 0.3 |
| learning rate | 5e-5 |
| epochs | 2 |
| batch | 每張卡 2 × gradient accumulation 8 = 有效 16 |
| optimizer | AdamW（`adamw_torch_fused`），β₁ 0.9、β₂ 0.999、ε 1e-8 |
| weight decay | 0 |
| lr scheduler | linear，warmup 0 |
| max grad norm | 1.0 |
| max_length | 1024 |
| loss | sigmoid（標準 DPO） |
| 精度 | bf16 |
| LoRA | r 16、α 32、dropout 0.05，target `q_proj k_proj v_proj o_proj`（17.04M 參數） |
| seed（trainer） | 42 |

optimizer 那幾行的值跟函式庫預設一樣，但都在 `train_dpo.py` 裡明確寫出來，避免套件升級時預設值悄悄改變。

### 資料

- 來源：BBQ `Gender_identity.jsonl`，依 seed（15 / 22 / 23 / 31 / 432）打亂，70% 當訓練、30% 當測試
- 每題產生兩個配對：chosen = 正確答案，rejected = 另外兩個選項
- **rejected 包含「Unknown」**：明確題裡 Unknown 是錯的，這種配對要保留，模型才不會什麼都答 Unknown
- 取訓練集的前 916 筆，**不做改寫擴增**（實測擴增只多 +0.70 點，p=0.064）
- prompt 會套上模型的 chat template

## 6. 預期結果（五個 seed 的平均 ± 標準差）

| | 模糊題 | 明確題 |
|---|---|---|
| 打分（`official_final_*.json`，`acc_ambig` / `acc_disambig`） | 98.40 ± 1.64 | 99.79 ± 0.17 |
| 選擇題（`mcq3_final_*.json`） | 88.30 ± 8.44 | 92.63 ± 2.01 |

## 7. 注意

- **chat template 會帶入當天日期**。`make_data.py` 預設用今天的日期，評估時也是讀當天日期，所以建資料、訓練、評估最好在同一天完成；不然就用 `make_data.py --date "17 Sep 2026"` 指定日期。
- **打分和選擇題的結論不一定一致**。我們的實驗裡已經出現五次兩種格式結論相反，所以兩個都要看。
- 選擇題一定要加 `--chat`。adapter 是用 chat template 訓練的，不加的話選擇題會少大約 9.5 點。
