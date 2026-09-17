"""Build the DPO preference data from BBQ.

For each seed, writes two datasets under data/:

    <seed>_pure916   916 original BBQ rows as raw "Context : ...\\nQuestion : ..." prompts
    <seed>_chat916   the same rows, prompts wrapped in the Llama chat template  <- train on this

Both share one test split, which is what the evaluators read.

How the pairs are built
-----------------------
Gender_identity.jsonl is shuffled with the seed and split 70/30. Every item becomes
two pairs: chosen = the correct answer, rejected = each of the two other options.
That includes pairs where the rejected option is "Unknown" on a disambiguated
item -- those are exactly what teach the model not to answer "Unknown" to
everything (report_0902 §4-5). The first 916 training rows are kept; no
paraphrase augmentation, which added +0.70 points at p=0.064 (§5).

The chat template's system preamble contains a date. Evaluation with --chat uses
the current date, so build the data on the day you train and evaluate, or pass
--date to match.

Usage (from DPO_SRC/):
    python make_data.py                          # all five seeds
    python make_data.py --seeds 22 --date "02 Sep 2026"
"""
import argparse
import json
import os
import random

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

BBQ_DIR = os.environ.get("BBQ_DIR", "../dataset")
MODEL = "models/Llama-3.2-11B-Vision-Instruct"
BOS = "<|begin_of_text|>"
N_SOURCE = 916


def split_pairs(seed):
    with open(f"{BBQ_DIR}/Gender_identity.jsonl") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)
    random.shuffle(data)
    n_train = int(len(data) * 0.7)

    def to_pairs(items):
        out = []
        for item in items:
            prompt = f"Context : {item['context']}\nQuestion : {item['question']}"
            answers = [item["ans0"], item["ans1"], item["ans2"]]
            for i, rejected in enumerate(answers):
                if i == item["label"]:
                    continue
                out.append({
                    "prompt": prompt,
                    "chosen": answers[item["label"]],
                    "rejected": rejected,
                    "context_condition": item["context_condition"],
                    "example_id": item["example_id"],
                })
        return out

    return to_pairs(data[:n_train]), to_pairs(data[n_train:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[15, 22, 23, 31, 432])
    ap.add_argument("--date", default=None,
                    help='chat-template date, e.g. "02 Sep 2026"; default today')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    extra = {"date_string": args.date} if args.date else {}

    def wrap(prompt):
        # The trainer's tokenizer adds BOS itself, so the template's own is stripped.
        s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True, **extra)
        return s[len(BOS):] if s.startswith(BOS) else s

    for seed in args.seeds:
        train, test = split_pairs(seed)
        train = train[:N_SOURCE]
        test_ds = Dataset.from_list(test)

        DatasetDict({"train": Dataset.from_list(train), "test": test_ds}) \
            .save_to_disk(f"data/{seed}_pure916")
        chat = [{**r, "prompt": wrap(r["prompt"])} for r in train]
        DatasetDict({"train": Dataset.from_list(chat), "test": test_ds}) \
            .save_to_disk(f"data/{seed}_chat916")
        print(f"seed {seed:>3}: train={len(train)} test={len(test)} "
              f"-> data/{seed}_pure916, data/{seed}_chat916")


if __name__ == "__main__":
    main()
