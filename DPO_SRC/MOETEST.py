import os



import gc
import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import transformers.utils.hub


from peft import (
    LoraConfig,
    get_peft_model,
)

from trl import DPOConfig, DPOTrainer
from huggingface_hub import login


# ==========================================
# 🌟 全局測試與優化開關
# ==========================================
TEST_MODE = False

# 💡 強烈建議：DPO 顯存消耗極大，若出現 OOM，請改為 True
USE_GRADIENT_CHECKPOINTING = False 

# ==========================================
# 0. 強制顯存清理與環境設定
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



# ==========================================
# 1. 模型與輸出路徑設定
# ==========================================
base_model_id = "google/gemma-4-26B-A4B-it"
dpo_final_path = "./gemma-4-26B-A4B-it-dpo-final-BBQ-CROSS"

DTYPE = torch.bfloat16

if TEST_MODE:
    print("\n" + "⚠️" * 20)
    print("⚠️ 測試模式 TEST_MODE 已開啟！將以極少資料與參數進行端到端測試。")
    print("⚠️" * 20 + "\n")


# ==========================================
# 2. 工具函式
# ==========================================
def free_memory():
    """強制清空 GPU 記憶體，確保不殘留 Cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_gpu_info():
    if not torch.cuda.is_available():
        print("⚠️ 未偵測到 CUDA GPU")
        return
    n = torch.cuda.device_count()
    print(f"✅ 偵測到 {n} 張 GPU")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}, VRAM={total_gb:.2f} GB")


def print_vram(step_name=""):
    if not torch.cuda.is_available():
        return
    print(f"\n--- 🖥️ VRAM [{step_name}] ---")
    total_alloc = 0.0
    total_reserved = 0.0
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total_alloc += alloc
        total_reserved += reserved
        print(f"  GPU {i}: allocated={alloc:.2f} GB | reserved={reserved:.2f} GB")
    print(f"🔥 total allocated={total_alloc:.2f} GB | total reserved={total_reserved:.2f} GB")
    print("-" * 60)


def load_tokenizer(model_id_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_bf16_model(model_id_or_path, for_training=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        device_map="auto",
        dtype=DTYPE,
        trust_remote_code=True,
    )

    if for_training:
        if hasattr(model, "config"):
            model.config.use_cache = False
        
        if USE_GRADIENT_CHECKPOINTING:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model


def maybe_print_device_map(model):
    if hasattr(model, "hf_device_map"):
        print("\n🧭 hf_device_map:")
        print(model.hf_device_map)


# ==========================================
# 3. Gemma 4 LoRA Target 獲取
# ==========================================
def build_gemma4_lora_targets(model):
    wanted_keywords = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "model.language_model.layers" not in name:
            continue
        if any(key in name for key in wanted_keywords):
            targets.append(name)

    targets = sorted(set(targets))

    if len(targets) == 0:
        raise ValueError("❌ No LoRA target modules found for Gemma 4.")

    print(f"\n✅ 找到 {len(targets)} 個 LoRA target Linear modules")
    return targets


def get_lora_config(model):
    target_modules = build_gemma4_lora_targets(model)
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def print_trainable(model, title="LoRA 可訓練參數"):
    print(f"\n📌 {title}:")
    try:
        model.print_trainable_parameters()
    except Exception as e:
        print(f"⚠️ 無法列印 trainable parameters: {e}")


# ==========================================
# 🔧 Text-only DPOTrainer Patch
# ==========================================
class TextOnlyDPOTrainer(DPOTrainer):
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        self.is_vision_model = False
        return super()._prepare_dataset(
            dataset, processing_class, args, dataset_name,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_vision_model = False


# ==========================================
# 🚀 主訓練階段：DPO
# ==========================================
def run_dpo_phase():
    print("\n" + "=" * 60)
    print("🚀 啟動 Gemma 4 26B A4B-it BF16 LoRA DPO 訓練")
    print("=" * 60)

    print_gpu_info()
    free_memory()
    print_vram("DPO 開始前")

    print(f"✅ 將直接從基礎模型進行 DPO: {base_model_id}")
    tokenizer = load_tokenizer(base_model_id)
    model = load_bf16_model(base_model_id, for_training=True)

    maybe_print_device_map(model)
    print_vram("DPO 模型載入完成")

    lora_config = get_lora_config(model)
    model = get_peft_model(model, lora_config)
    print_trainable(model, "DPO 階段 LoRA 可訓練參數")
    print_vram("DPO LoRA 注入完成")

    # ------------------------------------------
    # 處理 DPO 資料集
    # ------------------------------------------
    dpo_data = []
    with open("bbq_cross_dpo_zh.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dpo_data.append(json.loads(line))

    if TEST_MODE:
        dpo_data = dpo_data[:4]
        print(f"🔬 DPO 測試資料量：{len(dpo_data)} 筆")
    else:
        print(f"📚 DPO 資料量：{len(dpo_data)} 筆")

    dataset = Dataset.from_list(dpo_data)

    def format_dpo(example):
        messages = [
            {"role": "system", "content": "你是一個客觀且不具偏見的 AI 助手。"},
            {"role": "user", "content": example["prompt"]},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        chosen = example["chosen"]
        rejected = example["rejected"]

        if tokenizer.eos_token is not None:
            chosen = chosen + tokenizer.eos_token
            rejected = rejected + tokenizer.eos_token

        return {
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
        }

    dpo_dataset = dataset.map(
        format_dpo,
        remove_columns=dataset.column_names,
    )
    
    keep_cols = {"prompt", "chosen", "rejected"}
    remove_cols = [c for c in dpo_dataset.column_names if c not in keep_cols]
    if len(remove_cols) > 0:
        dpo_dataset = dpo_dataset.remove_columns(remove_cols)

    # ------------------------------------------
    # DPO 訓練設定
    # ------------------------------------------
    dpo_config = DPOConfig(
        output_dir=dpo_final_path,
        beta=0.1,
        max_length=768,
        max_prompt_length=384,

        per_device_train_batch_size=1 if TEST_MODE else 2,
        gradient_accumulation_steps=1 if TEST_MODE else 8,

        learning_rate=5e-6,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=1 if TEST_MODE else 20,
        num_train_epochs=1 if TEST_MODE else 2,

        bf16=True,
        fp16=False,

        logging_steps=1 if TEST_MODE else 10,
        report_to="none",
        save_strategy="epoch",

        # 這裡會吃 USE_GRADIENT_CHECKPOINTING 的全域設定
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if USE_GRADIENT_CHECKPOINTING else None,

        remove_unused_columns=False,
    )

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = TextOnlyDPOTrainer(
        model=model,
        ref_model=None, # PEFT 模式下設為 None 是標準做法
        train_dataset=dpo_dataset,
        args=dpo_config,
        processing_class=tokenizer,
    )

    print("🔥 開始 DPO 訓練...")
    trainer.train()

    print(f"💾 儲存最終 DPO LoRA 權重至 {dpo_final_path}")
    trainer.save_model(dpo_final_path)
    tokenizer.save_pretrained(dpo_final_path)

    print_vram("DPO 訓練結束後")

    del trainer, model
    free_memory()

    print("🎉 純 DPO 對齊訓練完成！")
    print(f"✅ 最終 DPO LoRA adapter 儲存於: {dpo_final_path}")


if __name__ == "__main__":
    free_memory()
    run_dpo_phase()