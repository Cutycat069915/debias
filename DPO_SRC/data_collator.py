from trl.trainer.dpo_trainer import DataCollatorForVisionPreference
import torch


class MllamaVisionDPOCollator:
    def __init__(self, processor, max_length=None, pad_to_multiple_of=None):
        self.base = DataCollatorForVisionPreference(
            processor=processor,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, examples):
        batch = self.base(examples)

        if (
            "input_ids" in batch
            and "cross_attention_mask" in batch
        ):
            seq_len = batch["input_ids"].shape[1]
            mask_len = batch["cross_attention_mask"].shape[1]

            if mask_len < seq_len:
                missing = seq_len - mask_len

                extension = (
                    batch["cross_attention_mask"][:, -1:, :, :]
                    .repeat(1, missing, 1, 1)
                )

                batch["cross_attention_mask"] = torch.cat(
                    [
                        batch["cross_attention_mask"],
                        extension,
                    ],
                    dim=1,
                )

        return batch
