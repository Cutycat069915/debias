# Tunable Hyperparameters in the Gemma 4 DPO + LoRA Training Script

## 1. LoRA Hyperparameters

| Hyperparameter | Current Value | What It Controls |
|---|---:|---|
| `r` | `16` | LoRA rank/capacity. Higher values increase trainable parameters. |
| `lora_alpha` | `32` | LoRA scaling strength. |
| `lora_dropout` | `0.05` | Dropout applied inside LoRA adapters. |
| `target_modules` | q/k/v/o + gate/up/down projections | Which model layers LoRA modifies. |
| `bias` | `"none"` | Whether bias parameters are trained. |
| `task_type` | `"CAUSAL_LM"` | Task type; normally fixed for this use case. |

Suggested values to test:

```python
r = [8, 16, 32, 64]
lora_alpha = [16, 32, 64]
lora_dropout = [0.0, 0.05, 0.1]
```

`target_modules` can also be treated as a hyperparameter.

Attention-only:
```text
q_proj
k_proj
v_proj
o_proj
```

Attention + MLP:
```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

---

## 2. DPO-Specific Hyperparameters

| Hyperparameter | Current Value | Meaning |
|---|---:|---|
| `beta` | `0.1` | Controls the strength of the policy/reference preference objective. |
| `max_length` | `768` | Maximum total sequence length. |
| `max_prompt_length` | `384` | Maximum prompt length. |

Suggested beta values:

```python
beta = [0.01, 0.05, 0.1, 0.2, 0.5]
```

For this bias-reduction experiment, `beta` is one of the most important hyperparameters.

---

## 3. Learning Rate and Optimizer Hyperparameters

Current settings:

```python
learning_rate=5e-6
optim="adamw_torch"
lr_scheduler_type="cosine"
warmup_steps=20
```

Tunable parameters:

- `learning_rate`
- `optim`
- `lr_scheduler_type`
- `warmup_steps`
- `warmup_ratio`
- `weight_decay`
- `adam_beta1`
- `adam_beta2`
- `adam_epsilon`

Suggested learning-rate search:

```python
learning_rate = [
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
]
```

Example additional optimizer settings:

```python
weight_decay=0.01
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-8
```

---

## 4. Learning-Rate Scheduler Hyperparameters

Current:

```python
lr_scheduler_type="cosine"
warmup_steps=20
```

Possible scheduler choices include:

```text
linear
cosine
cosine_with_restarts
constant
constant_with_warmup
```

You may also use:

```python
warmup_ratio = [0.0, 0.03, 0.05, 0.1]
```

instead of a fixed `warmup_steps`.

---

## 5. Batch-Size Hyperparameters

Current:

```python
per_device_train_batch_size=2
gradient_accumulation_steps=8
```

Both are tunable.

Effective batch size:

```text
effective batch size
= per_device_train_batch_size
× number_of_GPUs
× gradient_accumulation_steps
```

Possible values:

```python
per_device_train_batch_size = [1, 2, 4]
gradient_accumulation_steps = [4, 8, 16, 32]
```

It is usually better to think in terms of effective batch size.

---

## 6. Number of Epochs

Current:

```python
num_train_epochs=2
```

Possible values:

```python
num_train_epochs = [1, 2, 3, 5]
```

For DPO, too many epochs may overfit the preference dataset, so validation metrics should be monitored.

---

## 7. Sequence-Length Hyperparameters

Current:

```python
max_length=768
max_prompt_length=384
```

Possible values:

```python
max_prompt_length = [256, 384, 512]
max_length = [512, 768, 1024]
```

Longer sequence lengths significantly increase GPU memory usage.

---

## 8. Gradient Checkpointing

Current:

```python
USE_GRADIENT_CHECKPOINTING = False
```

Possible values:

```python
False
True
```

Effect:

- `False`: faster training, more VRAM usage.
- `True`: lower VRAM usage, more computation.

This mainly affects memory and runtime rather than model quality.

---

## 9. Numeric Precision

Current:

```python
DTYPE = torch.bfloat16
bf16=True
fp16=False
```

Possible precision settings include:

```text
FP32
BF16
FP16
```

For modern GPUs, BF16 is usually a sensible fixed choice rather than something to tune.

---

## 10. System Prompt

Current:

```python
{"role": "system", "content": "你是一個客觀且不具偏見的 AI 助手。"}
```

The system prompt is not a traditional optimizer hyperparameter, but it affects experimental results.

For a debiasing experiment, you may compare:

```text
With debiasing system prompt
vs.
No system prompt
```

If the goal is to measure whether DPO itself reduces bias, the system prompt should be kept constant across baselines, or evaluated separately.

---

## 11. EOS Handling

Current:

```python
chosen = chosen + tokenizer.eos_token
rejected = rejected + tokenizer.eos_token
```

EOS handling is a preprocessing choice.

You could compare:

```text
response
vs.
response + eos_token
```

Usually this should be fixed rather than tuned.

---

## 12. Dataset Hyperparameters

Dataset-related choices can also affect results:

- Number of training pairs
- Chosen/rejected construction
- Class balance
- Bias-category balance
- Language composition
- Sampling strategy
- Train/validation split

For bias datasets, category sampling can have a strong effect on the final model.

---

## 13. Reference Model

Current:

```python
ref_model=None
```

With PEFT, TRL can use the base model as the reference while disabling adapters.

Possible reference choices include:

```text
Base Gemma
SFT Gemma
Earlier checkpoint
Different aligned checkpoint
```

Reference-model choice changes the DPO optimization target.

---

## 14. DPO Loss Type

Depending on the installed TRL version, `DPOConfig` may support different loss functions.

Examples may include:

```text
sigmoid
hinge
ipo
robust
exo_pair
nca_pair
apo_zero
apo_down
```

The exact available options depend on the installed TRL version.

---

## 15. Label Smoothing

Possible parameter:

```python
label_smoothing
```

Example search:

```python
label_smoothing = [0.0, 0.05, 0.1]
```

This may help when preference labels contain noise or ambiguity.

---

## 16. Gradient Clipping

Transformers also exposes:

```python
max_grad_norm
```

Typical default:

```python
max_grad_norm=1.0
```

Possible values:

```python
max_grad_norm = [0.5, 1.0, 2.0]
```

This is normally kept fixed unless gradients become unstable.

---

# Complete Hyperparameter Categories

```text
LoRA
├── r
├── lora_alpha
├── lora_dropout
├── target_modules
├── bias
└── LoRA initialization/settings

DPO
├── beta
├── loss_type
├── label_smoothing
├── max_length
├── max_prompt_length
└── reference model

Optimization
├── learning_rate
├── optimizer
├── adam_beta1
├── adam_beta2
├── adam_epsilon
├── weight_decay
├── max_grad_norm
├── lr_scheduler_type
├── warmup_steps
└── warmup_ratio

Training
├── num_train_epochs
├── per_device_train_batch_size
├── gradient_accumulation_steps
├── gradient_checkpointing
└── precision / dtype

Data / Preprocessing
├── training-set size
├── train/validation split
├── sampling strategy
├── category balance
├── chosen/rejected construction
├── system prompt
├── chat template
├── EOS handling
└── truncation lengths
```

# Recommended Hyperparameters to Prioritize

| Priority | Hyperparameter | Current Value |
|---|---|---:|
| Very High | `beta` | `0.1` |
| Very High | `learning_rate` | `5e-6` |
| Very High | `num_train_epochs` | `2` |
| High | `r` | `16` |
| High | Effective batch size | `2 × GPUs × 8` |
| High | Training dataset size | Full dataset |
| Medium | `lora_alpha` | `32` |
| Medium | `lora_dropout` | `0.05` |
| Medium | `target_modules` | Attention + MLP |
| Medium | `max_length` | `768` |
| Medium | `max_prompt_length` | `384` |

# Suggested Initial Ablation

```python
beta = [0.05, 0.1, 0.2]
learning_rate = [2e-6, 5e-6, 1e-5]
r = [8, 16, 32]
num_train_epochs = [1, 2, 3]
```

Instead of immediately running the full 3^4 = 81 combination grid search, start by changing one variable at a time from the current baseline. This makes it much easier to determine which DPO setting is responsible for changes in BBQ/CrowS bias metrics.

