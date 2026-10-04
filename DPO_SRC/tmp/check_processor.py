from PIL import Image
from transformers import AutoProcessor

# your own imports
from dataclass import load_dpo_dataset, format_dpo_vision


def load_processor(model_id_or_path: str):
    return AutoProcessor.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
    )


data_path = "./split_dataset_vision/train_15.json"
model_path = "./models/Llama-3.2-11B-Vision-Instruct"


# 1. Load processor first
processor = load_processor(model_path)


# 2. Load + format dataset
dpo_dataset = load_dpo_dataset(
    data_path,
    processor,
    format_dpo_vision,
    True,   # TEST_MODE
)


# 3. Inspect one formatted example
example = dpo_dataset[0]

print("Dataset keys:")
print(example.keys())
print("=" * 80)


# 4. Get image
#
# If format_dpo_vision() returns:
#     "image": PIL_image
#
# use this:
image = example["image"]

# If instead you currently return:
#     "images": [PIL_image]
#
# use:
# image = example["images"][0]


print("Image type:", type(image))
print("Image size:", image.size)
print("Image mode:", image.mode)
print("=" * 80)


# 5. Inspect prompt
prompt = example["prompt"]

print("Prompt:")
print(prompt)
print("=" * 80)


# 6. THIS is the important experiment
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
)


# 7. Check exactly what MllamaProcessor generated
print("Processor output keys:")

for key, value in inputs.items():
    print(
        f"{key:25s}",
        type(value),
        getattr(value, "shape", None),
    )


print("=" * 80)

print(
    "pixel_values:",
    "pixel_values" in inputs,
)

print(
    "aspect_ratio_ids:",
    "aspect_ratio_ids" in inputs,
)

print(
    "aspect_ratio_mask:",
    "aspect_ratio_mask" in inputs,
)

print(
    "cross_attention_mask:",
    "cross_attention_mask" in inputs,
)

