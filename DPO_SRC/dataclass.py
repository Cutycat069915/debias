import json 
from datasets import Dataset
from PIL import Image
def format_dpo(example, processor):
        tokenizer =  processor.tokenizer
        messages = [
            {
                "role": "user",
                "content": example["question"]
            }
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        chosen = example["answer_debiased"]
        rejected = example["answer_biased"]

        if tokenizer.eos_token is not None:
            chosen += tokenizer.eos_token
            rejected += tokenizer.eos_token

        return {
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
        }
def format_dpo_vision(example,  processor):
    messages = [

        {
            "role" :"user",
            "content" : [
                {"type" : "image"},
                {
                 "type" : "text",
                 "text" : example["question"]
                }
        
            ]
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    chosen = example["answer_debiased"]
    rejected = example["answer_biased"]
    return {
        "prompt" : prompt_text,
        "chosen" : chosen,
        "rejected" :  rejected,
        "images" : [Image.open(example["image_path"]).convert("RGB")]
    }
def load_dpo_dataset(train_path, processor,formatter ,TEST_MODE):
    with open(train_path, "r", encoding="utf-8") as f:
        dpo_data = json.load(f)

    if TEST_MODE:
        dpo_data = dpo_data[:4]

    dataset = Dataset.from_list(dpo_data)

    return dataset.map(
        formatter,
        fn_kwargs={"processor": processor},
        remove_columns=dataset.column_names,
    )

