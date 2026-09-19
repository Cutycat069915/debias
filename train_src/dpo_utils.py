"""
dpo_utils.py
─────────────────────────────────────────────────────────────────────
Shared utilities for DPO training and evaluation pipelines.

Provides common functions used by both `train_dpo_peft.py` and
`evaluate.py`, eliminating code duplication for model loading,
tokenizer initialization, and checkpoint path parsing.
"""

import os
import json
from typing import Optional

import torch
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)

from config import TrainConfig


# ═══════════════════════════════════════════════════════════════════
#  MODEL & TOKENIZER LOADING
# ═══════════════════════════════════════════════════════════════════

def load_base_model(config: TrainConfig, *, use_cache: bool = False,
                    attn_impl: str = "eager", device_map="auto"):
    """
    Load the base Llama-3.2-11B-Vision-Instruct model.

    Args:
        config: Training configuration containing model_id and dtype.
        use_cache: Whether to enable KV-cache.
                   - False for training (required for gradient checkpointing)
                   - True for evaluation (speeds up generation)
        attn_impl: Attention implementation.
                   - "eager" for V100 (no FlashAttention)
                   - "sdpa" for newer GPUs
        device_map: Device placement strategy.
                    - {"": 0} for single-GPU training
                    - "auto" for evaluation

    Returns:
        The loaded model.
    """
    model = MllamaForConditionalGeneration.from_pretrained(
        config.model_id,
        torch_dtype=config.torch_dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    model.config.use_cache = use_cache
    return model


def load_tokenizer(config: TrainConfig):
    """
    Load the processor and tokenizer, setting pad_token if needed.

    Returns:
        (processor, tokenizer) tuple
    """
    processor = AutoProcessor.from_pretrained(config.model_id)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return processor, tokenizer


# ═══════════════════════════════════════════════════════════════════
#  CHECKPOINT PATH PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_save_path(save_path: str) -> dict:
    """
    Parse a checkpoint save_path to extract metadata.

    Supports both old format (e.g., ./save/syn1.0/31) and
    new mixed format (e.g., ./save/orig1.0_syn0.5/31_31).

    Args:
        save_path: Path to the checkpoint directory.

    Returns:
        dict with keys: parent_dir, seed, eval_split, hparams
    """
    clean_path = save_path.rstrip("/")
    seed = os.path.basename(clean_path)
    parent_dir = os.path.basename(os.path.dirname(clean_path))

    # Try to read hparams.json for the eval_split
    hparams_path = os.path.join(clean_path, "hparams.json")
    hparams = None
    if os.path.exists(hparams_path):
        with open(hparams_path, "r") as f:
            hparams = json.load(f)

    # Determine eval_split
    if hparams and "eval_split" in hparams:
        eval_split = hparams["eval_split"]
    else:
        # Old format: seed is the split name directly (e.g., "31")
        # New format without hparams: try first part before underscore
        if "_" in seed and not seed.isdigit():
            eval_split = seed.split("_")[0]
        else:
            eval_split = seed

    return {
        "parent_dir": parent_dir,
        "seed": seed,
        "eval_split": eval_split,
        "hparams": hparams,
    }


def save_hparams(save_path, config: TrainConfig, *,
                 data_size: float, syn_data_size: float,
                 original_split: str, syn_split: str,
                 num_orig_samples: int, num_syn_samples: int,
                 total_samples: int):
    """
    Save a JSON file with the hyperparameters used for a training run.

    Args:
        save_path: Directory to save hparams.json into.
        config: Training configuration.
        data_size: Fraction of original data used.
        syn_data_size: Fraction of synthetic data used.
        original_split: Name of the original data split.
        syn_split: Name of the synthetic data split.
        num_orig_samples: Number of original samples used.
        num_syn_samples: Number of synthetic samples used.
        total_samples: Total training samples.
    """
    hparams = {
        "data_size": data_size,
        "syn_data_size": syn_data_size,
        "original_split": original_split,
        "syn_split": syn_split,
        "eval_split": original_split,
        "beta": config.beta,
        "epochs": config.epochs,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "max_len": config.max_len,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "training_steps": config.training_steps,
        "num_orig_samples": num_orig_samples,
        "num_syn_samples": num_syn_samples,
        "total_samples": total_samples,
    }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)
