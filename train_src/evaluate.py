#!/usr/bin/env python3
"""
evaluate.py
─────────────────────────────────────────────────────────────────────
Evaluation pipeline for DPO fine-tuned models.

Performs two metrics per checkpoint:
  1. Multiple-Choice Accuracy (MCQ) — forced single-token A/B generation
  2. Semantic Similarity Matrix — 3x3 cosine similarity between
     [ground truth, LoRA generation, base generation]

Features:
  - Auto-discovers all checkpoints in ./save/*/* that have hparams.json
  - Skip logic: won't re-evaluate checkpoints that already have results
  - Safe PeftModel reuse: uses load_adapter() instead of from_pretrained()
  - Base accuracy caching: evaluated once per test split, not per checkpoint

Usage:
  python evaluate.py
"""

import os
import glob

import torch
from sentence_transformers import SentenceTransformer
from transformers import LogitsProcessor, LogitsProcessorList
from datasets import load_from_disk
from tqdm import tqdm
from peft import PeftModel

from config import TrainConfig
from dpo_utils import load_base_model, load_tokenizer, parse_save_path


# ═══════════════════════════════════════════════════════════════════
#  MCQ LOGITS PROCESSOR
# ═══════════════════════════════════════════════════════════════════

class MCQLogitsProcessor(LogitsProcessor):
    """
    Restricts model generation to only output tokens corresponding
    to the given choices (e.g., 'A' or 'B').
    """
    def __init__(self, tokenizer, choices=("a", "b")):
        self.allowed_token_ids = set()
        vocab = tokenizer.get_vocab()
        for choice in choices:
            target_lower = choice.lower()
            target_upper = choice.upper()
            for token_str, token_id in vocab.items():
                clean_token = token_str.replace("\u0120", "").replace("\u2581", "").strip()
                if clean_token in [target_lower, target_upper]:
                    self.allowed_token_ids.add(token_id)

        self.allowed_token_ids = list(self.allowed_token_ids)

        # Fallback: direct token lookup if vocabulary scan found nothing
        if not self.allowed_token_ids:
            for choice in choices:
                self.allowed_token_ids.append(tokenizer.convert_tokens_to_ids(choice.lower()))
                self.allowed_token_ids.append(tokenizer.convert_tokens_to_ids(choice.upper()))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = 0.0
        return scores + mask


# ═══════════════════════════════════════════════════════════════════
#  PROMPT FORMATTING
# ═══════════════════════════════════════════════════════════════════

def format_mcq_prompt(sample: dict) -> str:
    """Format a DPO sample into an A/B multiple-choice prompt."""
    return (
        f"Context/Question: {sample['prompt']}\n\n"
        f"Which of the following responses is better?\n"
        f"A) {sample['chosen']}\n"
        f"B) {sample['rejected']}\n\n"
        f"Answer:"
    )


# ═══════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def eval_multiple_choice(model, dataset, logits_processor, tokenizer) -> float:
    """
    Evaluate model on forced A/B multiple-choice selection.

    Returns:
        Classification accuracy (0.0 to 1.0).
    """
    model.eval()
    correct = 0
    total = len(dataset)

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Evaluating Multiple Choice"):
            prompt = format_mcq_prompt(sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

            pred_token_id = outputs.sequences[0][-1].item()
            pred_text = tokenizer.decode([pred_token_id]).strip().upper()

            if pred_text == "A":
                correct += 1

    return correct / total if total > 0 else 0.0


def eval_similarities(model, dataset, tokenizer, semantic_model) -> torch.Tensor:
    """
    Compute average 3x3 similarity matrix comparing:
    [yt (chosen), pi_theta (LoRA generation), pi_base (base generation)]

    Returns:
        3x3 torch.Tensor of average cosine similarities.
    """
    model.eval()
    total = len(dataset)
    final_matrix = torch.zeros(3, 3, device=model.device)

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Eval similarities"):
            inputs = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[-1]

            # Generate with LoRA active (pi_theta)
            out_theta = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_theta = tokenizer.decode(out_theta[0][input_length:], skip_special_tokens=True)

            # Generate with LoRA disabled (pi_base)
            with model.disable_adapter():
                out_base = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                gen_base = tokenizer.decode(out_base[0][input_length:], skip_special_tokens=True)

            # Compute 3x3 similarity matrix
            sentences = [sample["chosen"], gen_theta, gen_base]
            embeddings = semantic_model.encode(sentences)
            matrix = semantic_model.similarity(embeddings, embeddings)

            if not isinstance(matrix, torch.Tensor):
                matrix = torch.tensor(matrix, device=model.device)
            else:
                matrix = matrix.to(model.device)

            final_matrix += matrix

    final_matrix /= total
    return final_matrix


# ═══════════════════════════════════════════════════════════════════
#  RESULT SAVING
# ═══════════════════════════════════════════════════════════════════

def save_eval_results(base_log_root, parent_dir, seed, eval_split,
                      base_accuracy, accuracy, final_matrix, hparams):
    """
    Save evaluation results to disk:
      - accuracy/{seed}.txt
      - similarity/{seed}.pt
      - summary/{seed}.txt
    """
    target_sim_dir = os.path.join(base_log_root, parent_dir, "similarity")
    target_acc_dir = os.path.join(base_log_root, parent_dir, "accuracy")
    base_acc_dir = os.path.join(base_log_root, "base", "accuracy")
    target_summary_dir = os.path.join(base_log_root, parent_dir, "summary")

    for d in [target_sim_dir, target_acc_dir, base_acc_dir, target_summary_dir]:
        os.makedirs(d, exist_ok=True)

    # Save similarity matrix tensor
    torch.save(final_matrix, os.path.join(target_sim_dir, f"{seed}.pt"))

    # Save accuracy values
    with open(os.path.join(target_acc_dir, f"{seed}.txt"), "w") as f:
        f.write(f"{accuracy}\n")
    with open(os.path.join(base_acc_dir, f"{eval_split}.txt"), "w") as f:
        f.write(f"{base_accuracy}\n")

    # Save comprehensive summary
    with open(os.path.join(target_summary_dir, f"{seed}.txt"), "w") as f:
        f.write(f"Evaluation Summary: {parent_dir} | Seed: {seed}\n")
        f.write(f"{'='*50}\n")
        if hparams:
            f.write(f"\n--- Training Hyperparameters ---\n")
            for key in ["data_size", "syn_data_size", "original_split", "syn_split",
                         "beta", "epochs", "lr", "batch_size", "grad_accum",
                         "num_orig_samples", "num_syn_samples", "total_samples"]:
                f.write(f"{key:20s}{hparams.get(key)}\n")
        f.write(f"\n--- Evaluation Results ---\n")
        f.write(f"Base Acc:           {base_accuracy:.4f}\n")
        f.write(f"LoRA Acc:           {accuracy:.4f}\n")
        f.write(f"\n")
        f.write(f"Similarity Matrix (yt / gen_theta / gen_base):\n")
        f.write(f"{final_matrix}\n")
        f.write(f"\n")
        f.write(f"sim(yt, gen_theta): {final_matrix[0][1]:.4f}\n")
        f.write(f"sim(yt, gen_base):  {final_matrix[0][2]:.4f}\n")


# ═══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    config = TrainConfig()  # Use defaults for evaluation
    base_log_root = "./logs"

    # Discover all checkpoints with hparams.json
    raw_paths = glob.glob("./save/*/*")
    save_paths = sorted([
        p for p in raw_paths
        if os.path.exists(os.path.join(p, "hparams.json"))
    ])

    if not save_paths:
        print("No checkpoints found in ./save/*/*")
        return

    # Load models and tokenizer
    semantic_model = SentenceTransformer("./models/all-MiniLM-L6-v2")
    base = load_base_model(config, use_cache=True, attn_impl="sdpa")
    _, tokenizer = load_tokenizer(config)

    logits_processor = LogitsProcessorList([MCQLogitsProcessor(tokenizer)])

    # State for safe PeftModel reuse
    base_accuracy_cache = {}
    model = None

    for save_path in save_paths:
        info = parse_save_path(save_path)
        parent_dir = info["parent_dir"]
        seed = info["seed"]
        eval_split = info["eval_split"]
        hparams = info["hparams"]

        # Skip if already evaluated
        target_sim_dir = os.path.join(base_log_root, parent_dir, "similarity")
        target_acc_dir = os.path.join(base_log_root, parent_dir, "accuracy")
        if (os.path.exists(os.path.join(target_acc_dir, f"{seed}.txt"))
                and os.path.exists(os.path.join(target_sim_dir, f"{seed}.pt"))):
            print(f"Skipping {save_path}, already evaluated.")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {parent_dir} | Seed: {seed}")
        if hparams:
            print(f"  data_size={hparams.get('data_size')}, "
                  f"syn_data_size={hparams.get('syn_data_size')}, "
                  f"beta={hparams.get('beta')}, epochs={hparams.get('epochs')}")
        print(f"{'='*50}")

        # Load test set
        dataset = load_from_disk(f"./DataSet/DPO_SPLIT/{eval_split}")
        test_set = dataset["test"]

        # Compute base accuracy (cached per eval_split)
        if eval_split not in base_accuracy_cache:
            if model is not None:
                with model.disable_adapter():
                    base_accuracy_cache[eval_split] = eval_multiple_choice(
                        model, test_set, logits_processor, tokenizer
                    )
            else:
                base_accuracy_cache[eval_split] = eval_multiple_choice(
                    base, test_set, logits_processor, tokenizer
                )
        base_accuracy = base_accuracy_cache[eval_split]

        # Load LoRA adapter (reuse PeftModel wrapper)
        if model is None:
            model = PeftModel.from_pretrained(base, save_path, adapter_name="default")
        else:
            model.load_adapter(save_path, adapter_name="default")
            model.set_adapter("default")

        # Run evaluations
        final_matrix = eval_similarities(model, test_set, tokenizer, semantic_model)
        accuracy = eval_multiple_choice(model, test_set, logits_processor, tokenizer)

        # Save results
        save_eval_results(
            base_log_root, parent_dir, seed, eval_split,
            base_accuracy, accuracy, final_matrix, hparams
        )

        print(f"--> Saved to: {os.path.join(base_log_root, parent_dir)}")
        print(f"    [Base Acc: {base_accuracy:.4f}]")
        print(f"    [LoRA Acc: {accuracy:.4f}]")
        print(f"    [sim(yt, gen_theta): {final_matrix[0][1]:.4f}]")
        print(f"    [sim(yt, gen_base):  {final_matrix[0][2]:.4f}]")


if __name__ == "__main__":
    main()
