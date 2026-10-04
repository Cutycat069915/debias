import torch
from PIL import Image
from trl import DPOTrainer, DPOConfig
from transformers import AutoProcessor, AutoModelForImageTextToText
from dataclass import *
from data_collator import *
def print_shapes(title, data):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if hasattr(data, "keys"):
        print("Keys:", list(data.keys()))

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:35s} {tuple(value.shape)}")
        else:
            print(f"{key:35s} {type(value)}")


def diagnose_vision_dpo(
    model,
    processor,
    dpo_dataset,
    dpo_config,
):
    # --------------------------------------------------
    # 1. Check formatted dataset example
    # --------------------------------------------------
    example = dpo_dataset[0]

    print("\n" + "=" * 100)
    print("1. FORMATTED DATASET EXAMPLE")
    print("=" * 100)

    print("Dataset columns:")
    print(dpo_dataset.column_names)

    print("\nExample keys:")
    print(example.keys())

    print("\nPrompt:")
    print(example["prompt"])

    print("\nChosen:")
    print(example["chosen"])

    print("\nRejected:")
    print(example["rejected"])

    if "image" in example:
        image = example["image"]

    elif "images" in example:
        images = example["images"]

        print("\nNumber of images:", len(images))

        image = images[0]

    else:
        raise KeyError(
            "Dataset contains neither 'image' nor 'images'"
        )

    print("\nImage type:", type(image))
    print("Image size:", image.size)
    print("Image mode:", image.mode)

    # --------------------------------------------------
    # 2. Test MllamaProcessor directly
    # --------------------------------------------------
    processor_output = processor(
        text=example["prompt"],
        images=image,
        return_tensors="pt",
    )

    print_shapes(
        "2. DIRECT PROCESSOR OUTPUT",
        processor_output,
    )

    required_vision_keys = [
        "pixel_values",
        "aspect_ratio_ids",
        "aspect_ratio_mask",
        "cross_attention_mask",
    ]

    print("\nVision key check:")

    for key in required_vision_keys:
        print(
            f"{key:30s}",
            key in processor_output,
        )

    # --------------------------------------------------
    # 3. Create trainer
    # --------------------------------------------------
    collator = MllamaVisionDPOCollator(processor)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=dpo_dataset,
        args=dpo_config,
        processing_class=processor,
        data_collator = collator,
    )

    print("\n" + "=" * 100)
    print("3. TRAINER INFORMATION")
    print("=" * 100)

    print("Trainer:", type(trainer))

    if hasattr(trainer, "data_collator"):
        print(
            "Data collator:",
            type(trainer.data_collator),
        )

    # Useful with newer TRL
    if hasattr(trainer, "_is_vision_dataset"):
        print(
            "_is_vision_dataset:",
            trainer._is_vision_dataset,
        )

    if hasattr(trainer, "_is_vlm"):
        print(
            "_is_vlm:",
            trainer._is_vlm,
        )

    # --------------------------------------------------
    # 4. Get ONE batch without training
    # --------------------------------------------------
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))

    print_shapes(
        "4. DPO TRAINER BATCH",
        batch,
    )

    # --------------------------------------------------
    # 5. Compare important dimensions
    # --------------------------------------------------
    print("\n" + "=" * 100)
    print("5. SEQUENCE LENGTH DIAGNOSTICS")
    print("=" * 100)

    # Different TRL versions may use slightly different names,
    # so inspect every relevant key.
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue

        if (
            "input_ids" in key
            or "attention_mask" in key
            or "cross_attention" in key
            or "aspect_ratio" in key
            or "pixel_values" in key
        ):
            print(
                f"{key:35s}",
                tuple(value.shape),
            )

    # --------------------------------------------------
    # 6. Explicitly check suspected Mllama mismatch
    # --------------------------------------------------
    if (
        "input_ids" in batch
        and "cross_attention_mask" in batch
    ):
        input_len = batch["input_ids"].shape[1]
        cross_len = batch["cross_attention_mask"].shape[1]

        print("\ninput_ids sequence length:")
        print(input_len)

        print("cross_attention_mask sequence length:")
        print(cross_len)

        if input_len != cross_len:
            print("\n❌ LENGTH MISMATCH FOUND")
            print(
                f"input_ids length = {input_len}, "
                f"cross_attention_mask length = {cross_len}"
            )
        else:
            print("\n✅ Sequence lengths match")

    else:
        print(
            "\nCould not directly compare "
            "'input_ids' and 'cross_attention_mask'."
        )

    return trainer, batch
def main():
    path = "./models/Llama-3.2-11B-Vision-Instruct"
    dataset_path =  "./split_dataset_vision/train_15.json"
    model = AutoModelForImageTextToText.from_pretrained(path)
    processor =  AutoProcessor.from_pretrained(path)
    dataset = load_dpo_dataset(dataset_path, processor, format_dpo_vision, True)
    dpo_config = DPOConfig(
        output_dir="./diagnosis_output",

        # Keep this simple for diagnosis
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,

        # Important for VLM debugging
        max_length=None,
        remove_unused_columns=False,

        # We are NOT actually training
        num_train_epochs=1,

        report_to="none",
    )
    diagnose_vision_dpo(model, processor, dataset,  dpo_config) 
    	
    ...
if __name__ == "__main__":
    main()
