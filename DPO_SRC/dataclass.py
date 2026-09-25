import json 
from datasets import Dataset
def format_dpo(example,tokenizer):
        messages = [
            {
                "role": "user",
                "content": example["prompt"]
            }
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        chosen = example["chosen"]
        rejected = example["rejected"]

        if tokenizer.eos_token is not None:
            chosen += tokenizer.eos_token
            rejected += tokenizer.eos_token

        return {
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
        }
def load_dpo_dataset(train_path, tokenizer, TEST_MODE):
    with open(train_path, "r", encoding="utf-8") as f:
        dpo_data = json.load(f)

    if TEST_MODE:
        dpo_data = dpo_data[:4]

    dataset = Dataset.from_list(dpo_data)

    return dataset.map(
        format_dpo,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
    )
