# LoRA Rank — Study Note

## Code

```python
MAX_SEQ_LEN = 2048
LORA_RANK   = 32

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

---

## Definition

LoRA rank (`r`) is the **inner bottleneck dimension** of the low-rank decomposition used to approximate the weight update during fine-tuning. Instead of learning a full update matrix `ΔW ∈ ℝ^(d_out × d_in)`, LoRA factorizes it as the product of two smaller matrices:

$$
\Delta W = B \cdot A
$$

where `A ∈ ℝ^(r × d_in)` and `B ∈ ℝ^(d_out × r)`.

The rank `r` determines how much expressive capacity the adapter has:

- A **higher rank** allows the update to capture more complex transformations.
- A **lower rank** forces the update to live on a more constrained subspace.

Trainable parameter count scales **linearly with `r`**, so rank directly controls the trade-off between adapter capacity, memory footprint, and overfitting risk.

---

## Choosing the Rank

| Rank      | Capacity            | Memory   | When to Use                                                                 |
|-----------|---------------------|----------|-----------------------------------------------------------------------------|
| 4 – 8     | Low                 | Tiny     | Style transfer, format adherence, light instruction tuning                  |
| 16 – 32   | Medium              | Small    | Standard fine-tuning, RLHF / DPO / GRPO ← **this script (`r=32`)**          |
| 64 – 128  | High                | Moderate | Domain adaptation, large behavioral shifts, niche-language code             |
| 256+      | Approaches full FT  | Large    | Rarely worth it — prefer full fine-tune or DoRA                             |

---

## Key Principle

Higher rank is **not automatically better**.

In reinforcement learning fine-tuning (GRPO, PPO, DPO), excessive rank causes:

- The policy to drift too far from the reference model
- The KL divergence penalty to inflate
- Reward hacking to amplify

**Rule of thumb:**

- **RL stages (GRPO / PPO / DPO):** keep rank modest — `16–32`.
- **SFT on large, diverse datasets:** higher ranks are justified when the model genuinely needs more capacity to absorb new patterns.

---

## Quick Math Reference

For a Qwen 7B `q_proj` with `d = 3584`:

- Full update params: `3584 × 3584 ≈ 12.8 M`
- LoRA at `r = 32`: `32 × (3584 + 3584) ≈ 229 K`
- **Reduction: ~56×**

Formula:

$$
\text{LoRA params per layer} = r \times (d_{in} + d_{out})
$$

# BitsAndBytes 4-bit Configuration

## Code

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

---

## What It Does

Compresses model weights to 4-bit integers during loading, reducing GPU memory by ~4× while keeping activations and gradients in higher precision. Only the frozen weights are quantized; training gradients flow through dequantized activations.

---

## Parameters Breakdown

### `load_in_4bit=True`
Enables 4-bit quantization. Replaces all `nn.Linear` with `bnb.nn.Linear4bit`. Weights are stored as packed 4-bit integers (2 weights per byte) + per-block scaling factors.

### `bnb_4bit_quant_type="nf4"`
**NormalFloat-4 (NF4).** The 16 representable values are placed at the quantiles of a standard normal distribution. Optimal for LLM weights (approximately N(0, σ²)). Minimizes quantization error.

**Alternative:** `"fp4"` (standard 4-bit float) — less accurate for LLMs, rarely used.

### `bnb_4bit_compute_dtype=torch.bfloat16`
**Storage ≠ Compute.** Weights stay in 4-bit storage. During matrix multiplication, bnb **dequantizes on-the-fly to bf16**, performs the matmul, then discards the dequantized copy. If hardware lacks bf16 (older GPUs), falls back to `torch.float16`. BF16 keeps FP32's range but sacrifices precision. FP16 keeps more precision but has a small range.

### `bnb_4bit_use_double_quant=True`
**Quantize the quantization constants.** NF4 uses per-block scale factors (fp32). Double quant quantizes these scales to 8-bit with a second-level scale. Saves ~0.37 bits/weight with no accuracy loss. Standard in QLoRA.

---

## Memory Impact

For a 7B model in fp16:
- **fp16:** ~14 GB
- **4-bit + double quant:** ~3.5 GB (~4× reduction)

With LoRA adapters on top, gradients/optimizer state still consume memory but weights dominate → massive savings.

---

## Why This Config

This is the **QLoRA recipe** (Dettmers et al., 2023):

- NF4 is theoretically optimal for weights
- Double quant is free memory savings
- bf16 compute matches training precision
- Combined: minimal quality loss, maximum memory efficiency

Standard for fine-tuning large models on consumer/mid-tier GPUs.

# BitsAndBytes: Compute Dtype & Double Quantization

## Code

```python
bnb_4bit_compute_dtype=torch.bfloat16 if is_bf16_supported() else torch.float16
bnb_4bit_use_double_quant=True
```

---

## `bnb_4bit_compute_dtype=torch.bfloat16`

### What It Does

Weights are stored as **4-bit**, but during multiplication they are temporarily converted to **bf16** for computation. After the matmul finishes, the bf16 copy is deleted.

### Why

- **4-bit storage:** Saves memory (3.5 GB instead of 14 GB for 7B model)
- **bf16 compute:** Math is accurate during matmul
- **Delete after:** Don't waste VRAM keeping the dequantized copy

### Simple Flow

```
Storage: 4-bit weights
    ↓
Matmul starts: Dequantize to bf16 (temporary)
    ↓
Compute: Matmul in bf16 (accurate)
    ↓
Output: bf16 result flows forward
    ↓
Cleanup: Dequantized weights deleted
```

### Fallback

If GPU is old (pre-Ampere, no bf16 support), use `torch.float16` instead. Less efficient but works.

---

## `bnb_4bit_use_double_quant=True`

### What It Does

The scale factors that convert 4-bit weights back to real numbers are also quantized (8-bit instead of 32-bit).

### Why

NF4 needs scale factors. Storing them in 32-bit wastes memory. **Double quant:** quantize the scales themselves to 8-bit.

### Simple Breakdown

```
Without double quant:
  - Weights: 4-bit
  - Scale factors: 32-bit ← overhead, nearly as big as weights

With double quant:
  - Weights: 4-bit
  - Scale factors: 8-bit ← much smaller
  - Meta-scale: 32-bit (one per layer, negligible)
```

### Memory Saved

Saves ~**0.37 bits per weight** with **zero accuracy loss**. Free memory reduction.

---

## Why Together

**Storage ≠ Compute + Double Quant = QLoRA Standard**

This combination (NF4 + double quant + bf16 compute) is the recommended setup from the QLoRA paper. It gives you:

- Maximum memory compression (weights in 4-bit)
- Minimal overhead (scales in 8-bit)
- Good accuracy (compute in bf16)

Used in all modern QLoRA fine-tuning.

## Tokenizer Setup

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
```

## Why each line

**`from_pretrained(MODEL_PATH)`** — load tokenizer from same path as model. Vocab mismatch = silent corruption.

**`pad_token = eos_token`** — Qwen/LLaMA/Mistral ship without a pad token. Reusing EOS is safe because `attention_mask` zeros out padded positions during training. Skip this → `DataCollator` crashes.

| Before padding              | After padding                  |
| --------------------------- | ------------------------------ |
| `[12, 45, 89]`              | `[12, 45, 89, PAD, PAD]`       |
| `[12, 78, 23, 56, 91]`      | `[12, 78, 23, 56, 91]`         |
| `[34, 22]`                  | `[34, 22, PAD, PAD, PAD]`      |


**`padding_side = "left"`** — decoder LLMs generate to the right. Right-padding makes the model continue from `[PAD]` → garbage. Left-padding keeps the last real token at the generation position. Required for GRPO (generates during training).

## Verify

```python
print(tokenizer.pad_token_id, tokenizer.eos_token_id)  # should match
print(tokenizer.padding_side)                          # 'left'
print(len(tokenizer))                                  # must equal model embedding size
```

## Gotchas

- Set pad_token **before** tokenizing, not after.
- Encoder models (BERT) → use `padding_side="right"`.
- Save with `tokenizer.save_pretrained(OUT)` or changes are lost.