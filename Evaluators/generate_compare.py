"""史實亂編檢查：同一批測試題，讓未訓練和訓練後的模型各自寫回答，並排存成 log 給人看。

在 repo 根目錄執行：
    python Evaluators/generate_compare.py save/R16_B01_E2/seed_22
    python Evaluators/generate_compare.py save/R16_B01_E2/seed_22 --n 30 --split ./split_dataset

不自動判斷對錯——要人自己看。每題放：題目、資料集的去偏見回答（參考）、未訓練的回答、訓練後的回答，
另外標出這題在訓練集有沒有出現過（同一題有好幾組回答，一筆一筆切的話測試題可能訓練時看過）。
生成用 greedy（每次跑結果都一樣），題目套 chat template（跟訓練、evaluate.py 一樣）。
輸出：Evaluators/compare_logs/<adapter 名>_seed<seed>.md（給人看）和同名 .json。
"""
import argparse
import json
import os
import random
import re

import torch
from peft import PeftModel
from transformers import AutoTokenizer, MllamaForConditionalGeneration

from evaluate import BASE, load_json


def generate(model, tok, prompts, max_new_tokens, batch=8):
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", help="訓練後的 adapter 資料夾，例如 save/R16_B01_E2/seed_22")
    ap.add_argument("--split", default="./split_dataset")
    ap.add_argument("--n", type=int, default=25, help="抽幾題（不同的題目）")
    ap.add_argument("--sample-seed", type=int, default=0, help="抽題用的亂數種子，固定才能重現")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out-dir", default="./Evaluators/compare_logs")
    args = ap.parse_args()

    seed = int(re.search(r"seed_(\d+)", args.adapter).group(1))
    test = load_json(f"{args.split}/test_{seed}.json")
    seen = {r["question"].strip() for r in load_json(f"{args.split}/train_{seed}.json")}
    by_q = {}
    for r in test:  # 同一題只取第一筆，參考答案用它的 answer_debiased
        by_q.setdefault(r["question"].strip(), r)
    picked = random.Random(args.sample_seed).sample(sorted(by_q), min(args.n, len(by_q)))
    rows = [by_q[q] for q in picked]

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = MllamaForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model = PeftModel.from_pretrained(model, args.adapter)
    assert any("lora_B" in k and v.abs().sum() > 0 for k, v in model.state_dict().items()), f"{args.adapter} 沒掛上"

    prompts = [tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    with model.disable_adapter():
        base_out = generate(model, tok, prompts, args.max_new_tokens)
    tuned_out = generate(model, tok, prompts, args.max_new_tokens)

    name = re.sub(r"[^\w.-]+", "_", args.adapter.strip("./").replace("/seed_" + str(seed), ""))
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.out_dir}/{name}_seed{seed}"
    def hit_limit(s):  # 寫到上限才停（常是同一段一直重複），標出來
        return len(tok(s, add_special_tokens=False)["input_ids"]) >= args.max_new_tokens - 5
    items = [{"id": r["id"], "question": r["question"], "seen_in_train": r["question"].strip() in seen,
              "reference_debiased": r["answer_debiased"], "base": b, "tuned": t,
              "base_hit_limit": hit_limit(b), "tuned_hit_limit": hit_limit(t)}
             for r, b, t in zip(rows, base_out, tuned_out)]
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"adapter": args.adapter, "split": args.split, "sample_seed": args.sample_seed,
                   "max_new_tokens": args.max_new_tokens, "items": items}, f, ensure_ascii=False, indent=1)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(f"# 未訓練 vs 訓練後：{args.adapter}\n\n測試集 `{args.split}/test_{seed}.json`，抽 {len(items)} 題"
                f"（sample-seed {args.sample_seed}），greedy，最多 {args.max_new_tokens} token。\n"
                "⚠️ 回答被截在上限的話，結尾會不完整。\n")
        for i, it in enumerate(items, 1):
            f.write(f"\n---\n\n## {i}. [{it['id']}] {it['question']}\n\n"
                    f"{'⚠️ 這題訓練時看過' if it['seen_in_train'] else '訓練時沒看過這題'}\n\n"
                    f"**參考（資料集的去偏見回答）**\n\n{it['reference_debiased']}\n\n"
                    f"**未訓練**{' ⚠️ 寫到上限被截斷' if it['base_hit_limit'] else ''}\n\n{it['base'].strip()}\n\n"
                    f"**訓練後**{' ⚠️ 寫到上限被截斷' if it['tuned_hit_limit'] else ''}\n\n{it['tuned'].strip()}\n")
    print(f"✅ {stem}.md（{len(items)} 題；寫到上限：未訓練 {sum(i['base_hit_limit'] for i in items)}、"
          f"訓練後 {sum(i['tuned_hit_limit'] for i in items)}）")


if __name__ == "__main__":
    main()
