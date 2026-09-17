"""DPO + LoRA training for BBQ bias unlearning.

Defaults are the configuration the report settled on (report_0902 §11):
no filter + LoRA q,k,v,o + beta 0.3 + chat-template data, lr 5e-5, 2 epochs,
effective batch 2 x 8 = 16.

Every optimizer setting is written out below even where it equals the library
default, so the two labs can compare configs line by line and a library upgrade
cannot silently change them.

Usage (from DPO_SRC/):
    python train_dpo.py --dataset_path data/22_chat916 --output_dir save/dpo_final_22
"""
import argparse
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training Script for BBQ Bias Unlearning")
    parser.add_argument("--model_name_or_path", type=str, default="models/Llama-3.2-11B-Vision-Instruct", 
                        help="Path to the pre-trained SFT model to use as reference and target (e.g., CU7B).")
    parser.add_argument("--dataset_path", type=str, default="data/22_chat916", 
                        help="Path to the fixed BBQ dataset.")
    parser.add_argument("--output_dir", type=str, default="save/dpo_final_22", 
                        help="Directory to save the trained model.")
    parser.add_argument("--beta", type=float, default=0.3, 
                        help="KL divergence penalty coefficient (beta) for DPO.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate for DPO training.")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size per device.")
    parser.add_argument("--epochs", type=int, default=2, 
                        help="Number of training epochs.")
    # q,k,v,o = 17.04M trainable params (Chen-lab's choice); q,v = 8.52M.
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="LoRA target modules.")
    parser.add_argument("--eval_strategy", type=str, default="no",
                        help="Per-epoch eval during training; 'no' skips it (final scoring is done by eval_seed.py).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🚀 Loading dataset from {args.dataset_path}...")
    # 嚴格遵守路徑規範：基於根目錄 Augmented_Data/ 讀取
    dataset = load_from_disk(args.dataset_path)
    
    # DPOTrainer 預設需要 'prompt', 'chosen', 'rejected' 欄位，我們的修復版資料集完美符合！
    # 全自動化偏好最佳化：完全不依賴人工審查 (Human Consumption)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if "test" in dataset else None

    print(f"🤖 Loading tokenizer and models from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # 實體一：Target Model (要被訓練的模型 P_target)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype # 避免 AMP fp16/bf16 混用導致梯度爆炸或 logps 損壞
    )
    
    # 實體二：Reference Model (凍結的 SFT 模型 P_ref)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype
    )

    # 評估指標擴充性：預留 evaluation_strategy，未來可彈性掛載自訂指標，不寫死 B-score
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        eval_strategy=args.eval_strategy if eval_dataset else "no",
        save_strategy="epoch",
        logging_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        beta=args.beta,
        # Written out explicitly; these equal the trl 1.12 / transformers 5.14 defaults
        # the reported adapters were trained with.
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_grad_norm=1.0,
        max_length=1024,
        seed=42,
    )

    print(f"⚙️ Initializing DPOTrainer with beta={args.beta}...")
    # DPOTrainer 會自動在底層計算 Implicit Reward 並最佳化 DPO Loss：
    # L_DPO = - E [ log(sigma(beta * log(P_target(y_w|x)/P_ref(y_w|x)) - beta * log(P_target(y_l|x)/P_ref(y_l|x)))) ]
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules
    )
    print(f"⚙️ LoRA target_modules: {args.target_modules}")

    dpo_trainer = DPOTrainer(
        peft_config=peft_config,
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("🔥 Starting DPO training...")
    dpo_trainer.train()
    
    print(f"💾 Training complete. Saving final model to {args.output_dir}...")
    dpo_trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    main()
