import os
import time
import torch
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
import transformers.utils.hub

from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login

# ==========================================
# 0. 環境設定
# ==========================================

base_model_id = "google/gemma-4-26B-A4B-it"
#base_model_id = "/work/yian0529/gemma-4-26B-A4B-it-sft-merged-bf16"
processor_model_id = "google/gemma-4-26B-A4B-it"
lora_model_path = "./gemma-4-26B-A4B-it-dpo-final-BBQ"
test_mode = True

# ==========================================
# 工具：取得模型輸入應放置的 device
def get_first_model_device(model):
    """
    找模型第一個非 meta parameter 的 device。
    device_map='auto' 時，inputs 通常放到第一層所在 GPU 即可。
    """
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_inputs_to_device(inputs, device):
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved



# ==========================================
# 1. 視覺選擇題生成器
# ==========================================
def generate_mmbench_answer(image, question_text, options_text, model, processor):
    system_prompt = (
        "You are an objective AI assistant. "
        "Observe the image, read the question and options, "
        "and output ONLY the single correct letter (e.g., A, B, C, or D). "
        "Do not provide any explanation."
    )

    full_question = f"{question_text}\n\n{options_text}\nAnswer:"

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_question},
            ],
        },
    ]

    # Gemma 4 multimodal chat template
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        # fallback：若當前 transformers / processor 版本不支援 tokenize=True
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        )

    input_device = get_first_model_device(model)
    inputs = move_inputs_to_device(inputs, input_device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    latency = end_time - start_time

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    num_tokens = len(generated_tokens)

    generated_text = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip().upper()

    # 正則表達式抓取 ABCD
    match = re.search(r"\b([A-D])\b", generated_text)
    if match:
        pred_letter = match.group(1)
    else:
        if "A" in generated_text:
            pred_letter = "A"
        elif "B" in generated_text:
            pred_letter = "B"
        elif "C" in generated_text:
            pred_letter = "C"
        elif "D" in generated_text:
            pred_letter = "D"
        else:
            pred_letter = "Z"
            print(f"\n[警告] 無法解析模型輸出: '{generated_text}'")

    return pred_letter, generated_text, latency, num_tokens


# ==========================================
# 2. 主程式
# ==========================================
def main():
    print(f"🚀 載入模型 {base_model_id} ...")

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(lora_model_path):
        print(f"正在掛載 LoRA 權重: {lora_model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            is_trainable=False,
        )
    else:
        print("⚠️ 找不到 LoRA 路徑，以 Baseline 原始模型進行評估。")
        model = base_model

    processor = AutoProcessor.from_pretrained(
        processor_model_id,
        trust_remote_code=True,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.eval()

    if hasattr(model, "hf_device_map"):
        print("\n🧭 hf_device_map:")
        print(model.hf_device_map)

    print("✅ 模型載入完成！")

    # ==========================================
    # 3. 載入 MMBench 資料集
    # ==========================================
    print("📥 載入 lmms-lab/MMBench 測試集...")

    # 使用 lmms-lab/MMBench 並指定子集 ("en" 或 "cn")
    dataset = load_dataset("lmms-lab/MMBench", "cn", split="dev")

    
    data_to_eval = dataset.select(range(500)) if test_mode else dataset

    total_questions = len(data_to_eval)
    correct_count = 0
    parse_error_count = 0

    response_times = []
    token_counts = []

    print(f"\n🎨 開始 MMBench 視覺推理評估 (共 {total_questions} 題)...")

    for row in tqdm(data_to_eval):
        image = row["image"].convert("RGB")
        question_text = row["question"]
        correct_ans = str(row.get("answer", "")).strip().upper()

        # 動態組合選項，因為有些題目沒有 C 或 D
        options_text = ""
        for letter in ["A", "B", "C", "D"]:
            if (
                letter in row
                and row[letter] is not None
                and str(row[letter]).strip() != ""
            ):
                options_text += f"({letter}) {row[letter]}\n"

        pred_letter, raw_text, latency, num_tokens = generate_mmbench_answer(
            image,
            question_text,
            options_text,
            model,
            processor,
        )

        response_times.append(latency)
        token_counts.append(num_tokens)

        if pred_letter == correct_ans:
            correct_count += 1
        if pred_letter == "Z":
            parse_error_count += 1

            '''
        if test_mode:
            print(f"\n[題目]: {question_text}")
            print(f"[模型回答]: '{raw_text}' (解析為 {pred_letter}) | [標準答案]: {correct_ans}")
            '''

    # ==========================================
    # 4. 效能與準確率報告
    # ==========================================
    accuracy = correct_count / total_questions if total_questions > 0 else 0

    total_time = sum(response_times)
    total_tokens = sum(token_counts)
    avg_time = np.mean(response_times) if response_times else 0
    std_time = np.std(response_times) if response_times else 0
    time_per_token = total_time / total_tokens if total_tokens > 0 else 0
    tps = total_tokens / total_time if total_time > 0 else 0

    report = f"""
{"=" * 60}
MMBench 視覺通用能力保留評估
{"=" * 60}

【模型資訊】
🔹 基礎模型: {base_model_id}
🔹 資料集: lmms-lab/MMBench (cn)
🔹 LoRA: {lora_model_path if os.path.exists(lora_model_path) else '無 (Baseline)'}

【MMBench 準確率 (1-Pass Local Eval)】
  總題數: {total_questions}
  答對題數: {correct_count}
  無法解析: {parse_error_count}
  🏆 綜合準確率 (Accuracy): {accuracy:.2%}

【推論效能統計】
  Total Time: {total_time:.2f} s
  Avg Latency: {avg_time:.3f} s ± {std_time:.3f} s
  Time per Token: {time_per_token:.4f} s/token
  Tokens per Second: {tps:.2f} tokens/s
{"=" * 60}
"""

    print(report)

    model_name_short = base_model_id.split("/")[-1]
    status = "LoRA" if os.path.exists(lora_model_path) else "Base"
    output_filename = f"MMBench_Report_{model_name_short}_{status}.txt"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 報告已儲存至：{output_filename}")


if __name__ == "__main__":
    main()