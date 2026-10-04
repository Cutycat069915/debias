import torch
from transformers import LogitsProcessor, LogitsProcessorList
class MCQLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, choices=("A", "B", "C", "D")):
        self.allowed_token_ids = set()
        for c in choices :
            for variant in (c,  f" {c}"):
                ids = tokenizer.encode(variant,  add_special_tokens=False)
                if len(ids) == 1:
                    self.allowed_token_ids.add(ids[0])
        self.allowed_token_ids = list(self.allowed_token_ids)
    def __call__(self, input_ids,  scores):
        mask = torch.full_like(scores,  float("-inf"))
        mask[:,  self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask

