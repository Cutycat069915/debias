"""評估 DPO adapter：模型比較可能說出去偏見回答，還是偏見回答？

在 repo 根目錄執行：
    python Evaluators/evaluate.py save/R16_B01_E2/seed_22 save/R16_B03_E3/seed_22 ...
    python Evaluators/evaluate.py save/R16_B01_E2/seed_*      # 一組五個 seed
    python Evaluators/evaluate.py base/seed_22                 # 未訓練的底模

每個 adapter 用它自己 seed 的測試集（split_dataset/test_<seed>.json）。
做法：題目套 chat template（跟訓練一樣），算模型說出 chosen 和 rejected 的平均 log 機率，
chosen 比較高就算對。亂猜是 50%。
另外分開算「題目在訓練集出現過」和「沒出現過」——同一題有好幾組回答，一筆一筆切的話會重複。
結果存到 Evaluators/results.json；每個 adapter 資料夾裡另存 eval_results.pt（torch.load 讀），
執行過程寫進 Evaluators/logs/evaluate_<時間>.log。
"""
import argparse
import json
import os
import re
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, MllamaForConditionalGeneration

BASE = "./models/Llama-3.2-11B-Vision-Instruct"


def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def logprob(model, tok, prompt, answer):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p + a], device=model.device)
    with torch.no_grad():
        logp = model(input_ids=ids).logits[0, :-1].float().log_softmax(-1)
    target = ids[0, len(p):]
    return logp[len(p) - 1:].gather(1, target[:, None]).mean().item()

# must change to adapt to vision
def evaluate(model, tok, split_dir, seed):
    test = load_json(f"{split_dir}/test_{seed}.json")
    seen = {r["question"].strip() for r in load_json(f"{split_dir}/train_{seed}.json")}
    rows = []
    for r in test:
        prompt = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                         tokenize=False, add_generation_prompt=True)
        ok = logprob(model, tok, prompt, r["answer_debiased"]) > logprob(model, tok, prompt, r["answer_biased"])
        rows.append({"id": r["id"], "ok": ok, "seen": r["question"].strip() in seen})

    def acc(xs):
        return round(100 * sum(x["ok"] for x in xs) / len(xs), 2) if xs else None
    return {"all": acc(rows), "seen": acc([x for x in rows if x["seen"]]),
            "unseen": acc([x for x in rows if not x["seen"]]),
            "n": len(rows), "n_seen": sum(x["seen"] for x in rows), "items": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="+", help="adapter 資料夾，或 base/seed_<seed>")
    ap.add_argument("--split", default="./split_dataset")
    ap.add_argument("--out", default="./Evaluators/results.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    model = MllamaForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    results = load_json(args.out) if os.path.exists(args.out) else {}
    log_dir = os.path.join(os.path.dirname(args.out), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log = open(os.path.join(log_dir, time.strftime("evaluate_%Y%m%d_%H%M.log")), "a", encoding="utf-8")
    log.write(f"split={args.split}\n")

    for path in args.adapters:
        seed, key = int(re.search(r"seed_(\d+)", path).group(1)), path.rstrip("/")
        if path.startswith("base"):
            res = evaluate(model, tok, args.split, seed)
        else:
            peft = PeftModel.from_pretrained(model, path)
            # 載入類別不對時 adapter 會整個沒掛上、也不報錯，所以檢查一次
            assert any("lora_B" in k and v.abs().sum() > 0 for k, v in peft.state_dict().items()), f"{path} 沒掛上"
            res = evaluate(peft, tok, args.split, seed)
            model = peft.unload()  # 拿掉 LoRA，換下一個
        results[key] = res
        line = (f"{key:40} 全部 {res['all']}%  看過的題 {res['seen']}% (n={res['n_seen']})  "
                f"沒看過的 {res['unseen']}% (n={res['n'] - res['n_seen']})")
        print(line, flush=True)
        log.write(line + "\n"); log.flush()
        # 關鍵數字存成 .pt：adapter 存在它自己的資料夾；未訓練的 base 沒有資料夾，存在 out 旁邊
        pt = {k: v for k, v in res.items() if k != "items"}
        pt.update({"adapter": key, "seed": seed, "split": args.split, "items": res["items"],
                   "time": time.strftime("%Y-%m-%d %H:%M")})
        pt_path = (os.path.join(os.path.dirname(args.out), f"base_seed{seed}_eval_results.pt")
                   if path.startswith("base") else os.path.join(path, "eval_results.pt"))
        torch.save(pt, pt_path)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:  # 每個都存，中途出錯不會全丟
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
