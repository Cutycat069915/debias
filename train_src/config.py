"""
config.py
─────────────────────────────────────────────────────────────────────
Centralized configuration for DPO training and evaluation.

All hyperparameters and paths are managed through a single TrainConfig
dataclass, eliminating the need for global variables scattered across
multiple files.

Usage:
    from config import TrainConfig
    config = TrainConfig.from_args(args)
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from peft import LoraConfig, TaskType


@dataclass
class TrainConfig:
    """All training and evaluation hyperparameters in one place."""

    # ── Paths ──────────────────────────────────────────────────────
    model_id: str = "./models/Llama-3.2-11B-Vision-Instruct"
    output_dir: str = "./save"
    log_path: str = "./logs/dpo_peft_log.jsonl"
    dataset_root: str = "./DataSet/DPO_SPLIT"
    augmented_root: str = "./Augmented_DataSet"

    # ── LoRA ───────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ── Training ───────────────────────────────────────────────────
    max_len: int = 256
    beta: float = 0.1
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    save_steps: int = 100
    log_steps: int = 10
    training_steps: Optional[int] = None

    # ── Hardware ───────────────────────────────────────────────────
    device: str = "cuda"
    dtype: str = "float16"

    # ── Experiment Grid ────────────────────────────────────────────
    data_size: list = field(default_factory=lambda: [1.0])
    syn_data_size: list = field(default_factory=lambda: [1.0])
    syn_dataset_names: list = field(default_factory=lambda: ["31"])
    seeds: Optional[list] = None  # None = use all available seeds

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        """Build a TrainConfig from argparse Namespace, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in vars(args).items() if k in valid_keys}
        return cls(**filtered)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def build_lora_config(self) -> LoraConfig:
        """Create a LoRA configuration from the current hyperparameters."""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=r".*language_model.*self_attn\.(q|k|v|o)_proj",
            lora_dropout=self.lora_dropout,
            bias="none",
        )

    def to_dict(self) -> dict:
        """Serialize config to a JSON-friendly dictionary."""
        return asdict(self)

    def print_summary(self):
        """Print a human-readable summary of the current configuration."""
        print("Runtime training config:")
        for key, val in self.to_dict().items():
            print(f"  {key}={val}")
