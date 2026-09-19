#!/usr/bin/env python3
"""
train_dpo_peft.py
─────────────────────────────────────────────────────────────────────
DPO fine-tuning of Llama-3.2-11B-Vision-Instruct using PEFT (LoRA).
Adapted for BBQ_Gender_identity_dpo.jsonl format:
  { "prompt": "str", "chosen": "str", "rejected": "str" }

Dataset stats:
  - 3,015 preference pairs
  - Traditional Chinese, text-only (no images)
  - avg prompt ~200 chars, avg chosen ~111 chars, avg rejected ~25 chars

Key design:
  1. Single-model trick: disable_adapter_layers() for reference logprobs
     → avoids loading two 11B models (~44 GB); fits V100 32 GB
  2. fp16 throughout  — V100 has no BF16 hardware support
  3. attn_implementation="eager"  — required for V100 + Llama 3.x
  4. Regex LoRA targets  — language model self-attention only
  5. Gradient checkpointing  — saves ~40% VRAM on activations

VRAM budget (approx):
  Model fp16       22 GB
  LoRA params      ~50 MB
  AdamW (LoRA)    ~120 MB
  Activations       ~4 GB  (grad ckpt, BS=1)
  ─────────────────────────
  Total           ~27 GB  ✓ fits V100 32 GB

Usage:
  python train_dpo_peft.py
"""

import os, json, math
from pathlib import Path
from functools import partial


import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from  Parser import get_args
from  datasets import load_from_disk
from  glob  import glob
import gc
from  tqdm import tqdm
# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
MODEL_ID      = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DATA_PATH     = os.path.expanduser("~/bbq_lora/BBQ_Gender_identity_dpo.jsonl")
OUTPUT_DIR    = os.path.expanduser("./save")
LOG_PATH      = os.path.expanduser("~/bbq_lora/dpo_peft_log.jsonl")

# LoRA
LORA_R        = 16
LORA_ALPHA    = 32          # effective scale = alpha / r = 2
LORA_DROPOUT  = 0.05

# Training
MAX_LEN       = 256         # max tokens (prompts avg ~200 chars → 256 tok is safe)
BETA          = 0.1         # DPO temperature; higher → more conservative
BATCH_SIZE    = 1           # per-GPU
GRAD_ACCUM    = 8           # effective batch = 8
LR            = 5e-5
WEIGHT_DECAY  = 0.01
EPOCHS        = 2
WARMUP_RATIO  = 0.1
MAX_GRAD_NORM = 1.0
SAVE_STEPS    = 100         # every N optimiser steps
LOG_STEPS     = 10

DEVICE = torch.device("cuda")
DTYPE  = torch.float16      # V100: no BF16 hw


# ═══════════════════════════════════════════════════════════════════
#  LORA CONFIG
# ═══════════════════════════════════════════════════════════════════
# Regex targets language-model self-attention only.
# Path: language_model.model.layers.<i>.self_attn.{q,k,v,o}_proj
# Skips: vision_model encoder + cross_attn projections
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=r".*language_model.*self_attn\.(q|k|v|o)_proj",
    lora_dropout=LORA_DROPOUT,
    bias="none",
)


# ═══════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════
class BBQDPODataset(Dataset):
    """
    Loads BBQ_Gender_identity_dpo.jsonl.
    Format per line (all plain strings):
      {
        "prompt":   "閱讀以下情境並回答問題：...",
        "chosen":   "正確答案是...",
        "rejected": "答案是...。根據..."
      }
    """
    def __init__(self, path: str):
        self.data = [json.loads(l) for l in open(path) if l.strip()]
        print(f"Loaded {len(self.data)} DPO pairs from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ═══════════════════════════════════════════════════════════════════
#  TOKENISATION
# ═══════════════════════════════════════════════════════════════════
def _to_messages(prompt, response):
    """
    Normalise plain strings into chat-template message lists.
    Accepts both str and already-formatted list inputs (flexible).
    """
    if isinstance(prompt, str):
        prompt_msgs = [{"role": "user", "content": prompt}]
    else:
        prompt_msgs = prompt  # already a list of dicts

    if isinstance(response, str):
        resp_msgs = [{"role": "assistant", "content": response}]
    else:
        resp_msgs = response  # already a list of dicts

    return prompt_msgs, resp_msgs


def _tokenize_pair(processor, prompt, response, max_len):
    """
    Returns (input_ids, attention_mask, labels) where:
      - labels == -100 for prompt positions  (masked from DPO loss)
      - labels == token_id for response positions (supervised)
    """
    prompt_msgs, resp_msgs = _to_messages(prompt, response)

    # ── Full conversation ──────────────────────────────────────────
    full_text = processor.apply_chat_template(
        prompt_msgs + resp_msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_enc = processor(
        text=full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )

    # ── Prompt-only length  (to know where to mask labels) ────────
    prompt_text = processor.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,   # adds the assistant turn header
    )
    prompt_len = processor(
        text=prompt_text, return_tensors="pt",
        truncation=False, padding=False,
    )["input_ids"].shape[1]

    ids  = full_enc["input_ids"].squeeze(0)           # [T]
    mask = full_enc["attention_mask"].squeeze(0)      # [T]
    lbls = ids.clone()
    lbls[:prompt_len] = -100                          # mask prompt from loss

    return ids, mask, lbls


def collate_fn(batch, processor):
    """Tokenise a batch and right-pad chosen / rejected to their own max length."""

    def _pad_group(items):
        max_l  = max(x[0].shape[0] for x in items)
        pad_id = processor.tokenizer.pad_token_id
        ids_l, mask_l, lbl_l = [], [], []
        for ids, mask, lbls in items:
            p = max_l - ids.shape[0]
            ids_l.append(F.pad(ids,  (0, p), value=pad_id))
            mask_l.append(F.pad(mask, (0, p), value=0))
            lbl_l.append(F.pad(lbls,  (0, p), value=-100))
        return torch.stack(ids_l), torch.stack(mask_l), torch.stack(lbl_l)

    c_list, r_list = [], []
    for sample in batch:
        c_list.append(_tokenize_pair(processor, sample["prompt"], sample["chosen"],   MAX_LEN))
        r_list.append(_tokenize_pair(processor, sample["prompt"], sample["rejected"], MAX_LEN))

    c_ids, c_mask, c_lbls = _pad_group(c_list)
    r_ids, r_mask, r_lbls = _pad_group(r_list)

    return {
        "chosen_input_ids":        c_ids,
        "chosen_attention_mask":   c_mask,
        "chosen_labels":           c_lbls,
        "rejected_input_ids":      r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels":         r_lbls,
    }


# ═══════════════════════════════════════════════════════════════════
#  LOG-PROBABILITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════
def compute_logprobs(model, input_ids, attention_mask, labels):
    """
    Returns sum_{t ∈ response} log P(y_t | y_{<t}, x) for each sample.
      - shift by 1 for standard CLM next-token prediction
      - positions with labels == -100 (prompt) are excluded
    Returns: [B] float tensor
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits  = outputs.logits                         # [B, T, V]

    shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)

    # clamp -100 → 0 as dummy gather index (masked out below)
    target_ids = shift_labels.clamp(min=0)
    token_logp = log_probs.gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)                                    # [B, T-1]

    response_mask = (shift_labels != -100).float()
    return (token_logp * response_mask).sum(-1)      # [B]


# ═══════════════════════════════════════════════════════════════════
#  DPO LOSS  (Rafailov et al. 2023)
# ═══════════════════════════════════════════════════════════════════
def dpo_loss(π_c, π_r, ref_c, ref_r, beta=BETA):
    """  
                  - β*(log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

    Returns: (loss, chosen_reward, rejected_reward)
    reward_acc = (chosen_reward > rejected_reward) should → 1 during training
    """
    chosen_r   = beta * (π_c - ref_c)
    rejected_r = beta * (π_r - ref_r)
    loss = -F.logsigmoid(chosen_r - rejected_r).mean()
    return loss, chosen_r.detach().mean(), rejected_r.detach().mean()


# ═══════════════════════════════════════════════════════════════════
#  TRAINING EPOCH
# ═══════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, epoch, log_file):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0

    for step, batch in enumerate(tqdm(loader,desc = "Training")):
        c_ids  = batch["chosen_input_ids"].to(DEVICE)
        c_mask = batch["chosen_attention_mask"].to(DEVICE)
        c_lbls = batch["chosen_labels"].to(DEVICE)
        r_ids  = batch["rejected_input_ids"].to(DEVICE)
        r_mask = batch["rejected_attention_mask"].to(DEVICE)
        r_lbls = batch["rejected_labels"].to(DEVICE)

        # ── 1. Reference logprobs  (LoRA disabled → base weights) ─────────
        model.disable_adapter_layers()
        with torch.no_grad():
            ref_c = compute_logprobs(model, c_ids, c_mask, c_lbls)
            ref_r = compute_logprobs(model, r_ids, r_mask, r_lbls)
        model.enable_adapter_layers()

        # ── 2. Policy logprobs  (LoRA active) ─────────────────────────────
        π_c = compute_logprobs(model, c_ids, c_mask, c_lbls)
        π_r = compute_logprobs(model, r_ids, r_mask, r_lbls)

        # ── 3. DPO loss ────────────────────────────────────────────────────
        loss, chosen_rwd, rejected_rwd = dpo_loss(π_c, π_r, ref_c, ref_r)
        (loss / GRAD_ACCUM).backward()
        running_loss += loss.item()

        # ── 4. Optimiser step ──────────────────────────────────────────────
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                MAX_GRAD_NORM,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step = epoch * len(loader) + step
            opt_step    = (global_step + 1) // GRAD_ACCUM

            if opt_step % LOG_STEPS == 0:
                avg_loss = running_loss / GRAD_ACCUM
                running_loss = 0.0
                entry = {
                    "epoch":           epoch + 1,
                    "opt_step":        opt_step,
                    "loss":            round(avg_loss, 5),
                    "chosen_reward":   round(chosen_rwd.item(), 4),
                    "rejected_reward": round(rejected_rwd.item(), 4),
                    "reward_acc":      int(chosen_rwd > rejected_rwd),
                    "lr":              scheduler.get_last_lr()[0],
                }
                print(
                    f"[E{entry['epoch']} S{opt_step:04d}] "
                    f"loss={entry['loss']:.4f}  "
                    f"r_acc={entry['reward_acc']}  "
                    f"chosen_r={entry['chosen_reward']:+.3f}  "
                    f"rejected_r={entry['rejected_reward']:+.3f}  "
                    f"lr={entry['lr']:.2e}"
                )
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

            if opt_step % SAVE_STEPS == 0:
                ckpt = Path(OUTPUT_DIR) / f"step_{opt_step:05d}"
                model.save_pretrained(str(ckpt))
                print(f"  → checkpoint: {ckpt}")



def dataset_train(data_size_list, dataset_paths, base, processor):
    for d  in data_size_list:
        for path in dataset_paths:

            split_name = os.path.basename(path.rstrip("/"))

            print(split_name)
            model =  get_peft_model(base, LORA_CONFIG)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.print_trainable_parameters()
            dataset =  load_from_disk(path)
            train_dataset =  dataset["train"]

            num_samples = int(len(train_dataset) * d)
            train_dataset = train_dataset.select(range(num_samples))
            loader =  DataLoader(
                train_dataset,
                batch_size =  BATCH_SIZE,
                shuffle=True,
                collate_fn=partial(collate_fn,  processor=processor),
                num_workers=4,
                pin_memory=True
            )
            trainable = [p for p in model.parameters() if  p.requires_grad]
            optimizer =  AdamW(trainable, lr = LR,  weight_decay = WEIGHT_DECAY)
            total_opt_steps = math.ceil(len(loader) / GRAD_ACCUM) * EPOCHS
            warmup_steps    = int(total_opt_steps * WARMUP_RATIO)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_opt_steps
            )

            with open(LOG_PATH, "w") as log_file:
                for epoch in range(EPOCHS):
                    print(f"\n{'═'*60}")
                    print(f"  EPOCH {epoch + 1} / {EPOCHS}")
                    print(f"{'═'*60}")
                    train_epoch(model, loader, optimizer, scheduler, epoch, log_file)
            model.save_pretrained(f"./save/data{d}/{split_name}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
    ...
def syn_dataset_train(dataset_paths,base, processor):
    for path in dataset_paths:
        clean_path =  path.rstrip("/")
        split_name = os.path.basename(clean_path)#should be the randomseed
        parent_dir =  os.path.dirname(clean_path)
        portion =  os.path.basename(parent_dir) # should be portion

        print(f"{portion} + {split_name}")
        model =  get_peft_model(base, LORA_CONFIG)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()
        dataset =  load_from_disk(path)
        train_dataset =  dataset["train"]
        
        loader =  DataLoader(
            train_dataset,
            batch_size =  BATCH_SIZE,
            shuffle=True,
            collate_fn=partial(collate_fn,  processor=processor),
            num_workers=4,
            pin_memory=True
        )
        trainable = [p for p in model.parameters() if  p.requires_grad]
        optimizer =  AdamW(trainable, lr = LR,  weight_decay = WEIGHT_DECAY)
        total_opt_steps = math.ceil(len(loader) / GRAD_ACCUM) * EPOCHS
        warmup_steps    = int(total_opt_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_opt_steps
        )

        with open(LOG_PATH, "w") as log_file:
            for epoch in range(EPOCHS):
                print(f"\n{'═'*60}")
                print(f"  EPOCH {epoch + 1} / {EPOCHS}")
                print(f"{'═'*60}")
                train_epoch(model, loader, optimizer, scheduler, epoch, log_file)
        model.save_pretrained(f"./save/syn{portion}/{split_name}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    ...

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════



def main():
    args = get_args()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    processor =  AutoProcessor.from_pretrained(args.model_name)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    base = MllamaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=DTYPE,
        attn_implementation="eager",   # V100: no FlashAttention
        device_map="auto",
    )
    base.config.use_cache = False      # required for gradient checkpointing
     
    data_size_list = [0.5,0.7,1.0] #add 0.7,0.5 back for data set 
    
    original_dataset_paths =  glob(r"./DataSet/DPO_SPLIT/*")
    syn_dataset_paths =  glob(r"./DataSet/SYN_DPO_SPLIT/*/*")
    if args.syn == True :
        dataset_paths = syn_dataset_paths
        syn_dataset_train(dataset_paths,  base, processor)
    else:
        dataset_paths = original_dataset_paths
        datatset_train(data_size_list, dataset_paths, base,  processor)
    print(dataset_paths)


if __name__ == "__main__":
    main()
