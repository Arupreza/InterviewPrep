# LoRA Rank — Study Note

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