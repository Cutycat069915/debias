import os
from glob import glob
import yaml

import gc
import json
import torch
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor
)

import transformers.utils.hub
from transformers import MllamaForConditionalGeneration

from peft import (
    LoraConfig,
    get_peft_model,
)

import argparse
from trl import DPOConfig, DPOTrainer
from huggingface_hub import login
from utils import *
from dataclass import *
from peft import PeftModel
from PIL import Image
from data_collator import *
# ==========================================
# 🌟 全局測試與優化開關
# ==========================================
TEST_MODE = False
VISION_MODE = False
SEEDS = [15, 22, 23, 32, 432]
# 💡 強烈建議：DPO 顯存消耗極大，若出現 OOM，請改為 True
USE_GRADIENT_CHECKPOINTING = False 

# ==========================================
# 0. 強制顯存清理與環境設定
# ==========================================
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


CONFIG_DIR = "./DPO_SRC/configs"
DATASET_DIR = "./split_dataset"
VISION_DATASET_DIR = "./split_dataset_vision"
SAVE_DIR = "./save"

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

def load_processor(model_id_or_path:str):
    processor = AutoProcessor.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
    )

    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({
                "pad_token": "<|pad|>"
            })

    tokenizer.padding_side = "right"

    return processor 


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
def print_trainable(model, title="LoRA 可訓練參數"):
    print(f"\n📌 {title}:")
    try:
        model.print_trainable_parameters()
    except Exception as e:
        print(f"⚠️ 無法列印 trainable parameters: {e}")

def get_lora_config(config, model):
    rank = config["lora"]["rank"]
    alpha = config["lora"]["alpha"]
    dropout = config["lora"]["dropout"]

    target_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    target_modules = []

    for name, _ in model.named_modules():
        if not name.startswith("model.language_model."):
            continue

        if name.split(".")[-1] in target_suffixes:
            target_modules.append(name)

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
def load_bf16_model_v1(model_id_or_path, for_training=True):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id_or_path,
        torch_dtype=DTYPE,
        device_map={"":0},
        trust_remote_code=True,
    )

    if for_training:
        model.config.use_cache = False

        if USE_GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }
            )

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model

def run_dpo_phase_v1(config,seed,testcase_name):
    base_model_id = config["base_model_id"]
   

    beta = config["dpo"]["beta"]
    epochs = config["dpo"]["epochs"]
    learning_rate = float(config["dpo"]["learning_rate"])
    max_length = config["dpo"]["max_length"]
    max_prompt_length = config["dpo"]["max_prompt_length"]
    
    batch_size = config["training"]["batch_size"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    gradient_checkpointing = config["training"]["gradient_checkpointing"]
    
    train_path = os.path.join(
        "./split_dataset_vision" if VISION_MODE else"./split_dataset",
        f"train_{seed}.json"
    )
    output_dir = os.path.join(
        "./vision_save" if VISION_MODE else "text_save" ,
        testcase_name,
        f"seed_{seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print_gpu_info()
    free_memory()
    #load_model
    processor = load_processor(base_model_id)
    model = load_bf16_model_v1(base_model_id, for_training=True)
    if VISION_MODE:
        text_lora_path =  os.path.join("./text_save",testcase_name,f"seed_{seed}")
        model  = PeftModel.from_pretrained(model,  text_lora_path,is_trainable=True)
        dpo_dataset = load_dpo_dataset(train_path, processor,format_dpo_vision, TEST_MODE) 
    else:
        lora_config = get_lora_config(config,  model)           
        model =  get_peft_model(model,  lora_config)
        dpo_dataset = load_dpo_dataset(train_path, processor,format_dpo, TEST_MODE)     
    collator = MllamaVisionDPOCollator(processor) 
    dpo_config = DPOConfig(
        output_dir=output_dir,

        beta=beta,
        max_length=None,

        per_device_train_batch_size=(
            1 if TEST_MODE else batch_size
        ),

        gradient_accumulation_steps=(
            1
            if TEST_MODE
            else gradient_accumulation_steps
        ),

        learning_rate=learning_rate,

        num_train_epochs=(
            1 if TEST_MODE else epochs
        ),

        optim="adamw_torch",
        lr_scheduler_type="cosine",

        warmup_steps=(
            1 if TEST_MODE else 20
        ),

        bf16=True,
        fp16=False,

        logging_steps=(
            1 if TEST_MODE else 10
        ),

        report_to="none",
        save_strategy="epoch",

        seed=seed,
        data_seed=seed,

        gradient_checkpointing=(
            gradient_checkpointing
        ),

        gradient_checkpointing_kwargs=(
            {"use_reentrant": False}
            if gradient_checkpointing
            else None
        ),

        remove_unused_columns=False,
        
    )
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
  
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=dpo_dataset,
        args=dpo_config,
        processing_class=processor if VISION_MODE else processor.tokenizer,
        data_collator = collator,
    )

    trainer.train()

    trainer.save_model(output_dir)
    processor_saver = processor if VISION_MODE else processor.tokenizer
    processor_saver.save_pretrained(output_dir)
    print_vram("DPO 訓練結束後")
    del trainer
    del model
    del dpo_dataset

    free_memory()

    print(
        f"✅ Finished testcase={testcase_name}, "
        f"seed={seed}"
    )
def run_testcase(config_path):
    config = get_config(config_path)
    testcase_name = Path(config_path).stem
    print("\n" + "#" * 80)
    print(f"Starting testcase: {testcase_name}")
    print("#" * 80)
    for seed in SEEDS:
        try:
            run_dpo_phase_v1(
                config=config,
                seed=seed,
                testcase_name=testcase_name,
            )

        except Exception as e:
            print(
                f"Failed testcase={testcase_name}, "
                f"seed={seed}"
            )
            print(e)

        finally:
            free_memory()
def run_all_experiments():
    config_files = sorted(
        glob("./DPO_SRC/configs/*.yaml")
    )

    print(f"Found {len(config_files)} testcases")

    for config_path in config_files:
        run_testcase(config_path)    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--vision",
        action="store_true",
        help="Enable vision DPO training",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="Enable TestMode",
            
    )

    args = ap.parse_args()
    VISION_MODE = args.vision
    TEST_MODE = args.test
    run_all_experiments()
