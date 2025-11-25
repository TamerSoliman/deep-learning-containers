# GPTQ vs AWQ vs bitsandbytes: Deep Technical Comparison

**Purpose**: Understand the algorithmic differences between the three most popular 4-bit quantization methods.

---

## Executive Summary

| Aspect | GPTQ | AWQ | bitsandbytes (NF4) |
|--------|------|-----|-------------------|
| **Algorithm** | Layer-wise Hessian-based | Activation-aware scaling | On-the-fly quantization |
| **Calibration** | Required (128-1024 samples) | Required (128 samples) | Not required |
| **Quantization Time** | 1-2 hours (70B) | 15-30 min (70B) | Instant |
| **Inference Speed** | Fast (optimized kernels) | Fastest (best kernels) | Slower (no kernel opt) |
| **Quality** | Good (85-90%) | Best (94-98%) | Good (89-93%) |
| **Fine-tuning** | ❌ No | ❌ No | ✅ Yes (QLoRA) |
| **Ease of Use** | Medium (requires calibration) | Medium (requires calibration) | Easy (one line) |
| **Best For** | Base models, batch inference | Instruction models, latency-critical | Fine-tuning, quick start |

---

## GPTQ: Generative Pre-trained Transformer Quantization

### Algorithm Overview

GPTQ uses **layer-wise optimal brain quantization** to minimize reconstruction error.

### Mathematical Foundation

**Goal**: Minimize the difference between quantized and original outputs:

```
min ||W_fp16 × X - W_int4 × X||²
```

**Approach**:
1. For each layer, compute the Hessian matrix H:
   ```
   H = 2 · X^T · X / n
   ```
   (Measures sensitivity of each weight)

2. Quantize weights in order of increasing Hessian diagonal:
   ```
   For each weight w_i:
     q_i = quantize(w_i)         # Quantize to 4-bit
     error = w_i - dequantize(q_i)
     w_remaining += error · H_inv  # Update other weights
   ```

3. Use Cholesky decomposition for efficient Hessian inversion

### GPTQ Algorithm (Simplified)

```python
def gptq_quantize_layer(weights, activations, bits=4):
    """
    GPTQ quantization for single layer
    """
    # Step 1: Compute Hessian (weight importance)
    X = activations  # [batch, in_features]
    H = 2 * (X.T @ X) / len(X)  # [in_features, in_features]
    H_inv = torch.linalg.inv(H + damping * torch.eye(H.shape[0]))

    # Step 2: Quantize weights in order
    W_quant = torch.zeros_like(weights)
    for i in range(weights.shape[1]):  # For each column
        w = weights[:, i].clone()

        # Quantize
        q = quantize_to_4bit(w)
        W_quant[:, i] = q

        # Compute error
        error = w - dequantize(q)

        # Update remaining weights using Hessian
        weights[:, i+1:] -= (error.unsqueeze(1) @ H_inv[i, i+1:].unsqueeze(0))

    return W_quant
```

### GPTQ Characteristics

**Strengths**:
- ✅ Theoretically optimal (minimizes reconstruction error)
- ✅ Works well for base models
- ✅ Fast inference (optimized CUDA kernels)
- ✅ Grouping reduces quantization error

**Weaknesses**:
- ❌ Computationally expensive (1-2 hours for 70B)
- ❌ Requires calibration dataset
- ❌ Sensitive to calibration data quality
- ❌ May over-quantize important weights if they're small

**Hyperparameters**:

```python
{
    "bits": 4,                    # 4-bit quantization
    "group_size": 128,            # Quantize in groups of 128
    "damp_percent": 0.01,         # Hessian damping (stability)
    "desc_act": True,             # Reorder by activation size
    "sym": False,                 # Asymmetric quantization
    "true_sequential": True,      # Sequential layer processing
}
```

**Group Size Impact**:
```
group_size=128: Standard (good quality)
group_size=64:  Higher quality, larger model
group_size=32:  Best quality, even larger
group_size=-1:  Per-column (highest quality, slowest)
```

---

## AWQ: Activation-Aware Weight Quantization

### Algorithm Overview

AWQ protects weights that correspond to **salient activations** (high-magnitude activations).

### Key Insight

**Observation**: Not all weights are equally important
- Small weights with large activations → Very important
- Large weights with small activations → Less important

**GPTQ problem**: Only looks at weight magnitudes, may hurt important small weights

**AWQ solution**: Scale weights based on activation importance

### Mathematical Foundation

**Goal**: Protect salient channels during quantization

```
For each channel j:
  1. Measure activation variance: s_j = Var(X[:, j])
  2. Scale weights: W[:, j] ← W[:, j] × s_j^α
  3. Quantize scaled weights: Q(W)
  4. Scale activations inversely: X[:, j] ← X[:, j] / s_j^α
```

Result: W' × X = (W × s^α) × (X / s^α) = W × X (mathematically equivalent)

### AWQ Algorithm (Simplified)

```python
def awq_quantize_layer(weights, activations, bits=4, alpha=0.5):
    """
    AWQ quantization for single layer
    """
    # Step 1: Compute activation saliency per channel
    X = activations  # [batch, in_features]
    saliency = X.abs().mean(dim=0)  # [in_features]

    # Step 2: Compute scaling factors
    scales = saliency.pow(alpha)  # α typically 0.5
    scales = scales / scales.max()  # Normalize

    # Step 3: Scale weights up (protection)
    W_scaled = weights * scales.unsqueeze(0)

    # Step 4: Quantize scaled weights
    W_quant = quantize_to_4bit_groupwise(W_scaled, group_size=128)

    # Step 5: Store inverse scales for activation scaling at runtime
    # At inference: output = (W_quant @ (X / scales))

    return W_quant, 1.0 / scales
```

### AWQ Characteristics

**Strengths**:
- ✅ Best quality among 4-bit methods
- ✅ 3-10x faster quantization than GPTQ
- ✅ Excellent for instruction-tuned models
- ✅ Best optimized CUDA kernels (GEMM backend)
- ✅ Scales better to very large models

**Weaknesses**:
- ❌ Requires calibration dataset (though fewer samples)
- ❌ Slightly more complex setup than bitsandbytes
- ❌ Cannot fine-tune after quantization

**Hyperparameters**:

```python
{
    "bits": 4,                    # 4-bit quantization
    "group_size": 128,            # Quantization group size
    "alpha": 0.5,                 # Scaling exponent (0.4-0.6)
    "zero_point": True,           # Use zero-point quantization
    "version": "GEMM",            # Kernel version (GEMM fastest)
}
```

**Alpha Parameter Impact**:
```
α=0.0: No protection (equivalent to naive quantization)
α=0.5: Standard (balanced protection)
α=1.0: Maximum protection (may over-protect)
```

---

## bitsandbytes (NF4): NormalFloat 4-bit

### Algorithm Overview

bitsandbytes uses **information-theoretically optimal quantization** for normally-distributed data.

### Key Insight

**Observation**: Neural network weights follow approximately normal distribution N(0, σ²)

**Standard approach**: Uniform quantization (equal spacing)
```
INT4: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
```

**NF4 approach**: Non-uniform quantization (more bins near 0)
```
NF4: Information-optimal for N(0,1)
  More bins where probability density is high (near 0)
  Fewer bins where probability is low (tails)
```

### NF4 Data Type

**Quantization bins** (for normalized N(0,1) weights):

```python
NF4_BINS = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
```

These bins minimize expected quantization error for N(0,1) distribution.

### bitsandbytes Algorithm (Simplified)

```python
def nf4_quantize(weights, block_size=64):
    """
    NF4 quantization with blockwise normalization
    """
    # Step 1: Divide into blocks
    W_blocks = weights.reshape(-1, block_size)

    quantized_blocks = []
    absmax_blocks = []

    for block in W_blocks:
        # Step 2: Compute block absmax (for normalization)
        absmax = block.abs().max()
        absmax_blocks.append(absmax)

        # Step 3: Normalize to [-1, 1]
        W_norm = block / (absmax + 1e-8)

        # Step 4: Quantize to NF4 bins
        # Find nearest NF4 bin for each weight
        W_nf4 = torch.zeros_like(W_norm, dtype=torch.uint8)
        for i, w in enumerate(W_norm):
            bin_idx = find_nearest_nf4_bin(w)  # Returns 0-15
            W_nf4[i] = bin_idx

        quantized_blocks.append(W_nf4)

    # Step 5: Optional double quantization (quantize absmax values)
    absmax_quant = quantize_fp8(absmax_blocks)

    return quantized_blocks, absmax_quant

def dequantize_nf4(W_nf4, absmax):
    """
    Dequantize NF4 back to FP16
    """
    W_fp16 = NF4_BINS[W_nf4]  # Lookup bin values
    W_fp16 = W_fp16 * absmax  # Denormalize
    return W_fp16
```

### Double Quantization

**Standard NF4**: Stores absmax in FP16
- Model: 0.5 bytes/weight
- Absmax: 2 bytes per 64 weights = 0.03 bytes/weight
- Total: 0.53 bytes/weight

**Double Quantization**: Quantizes absmax to FP8
- Model: 0.5 bytes/weight
- Absmax: 1 byte per 64 weights = 0.016 bytes/weight
- Total: 0.52 bytes/weight (slight savings)

### bitsandbytes Characteristics

**Strengths**:
- ✅ No calibration required (instant quantization)
- ✅ Supports fine-tuning (QLoRA)
- ✅ One line of code (easiest setup)
- ✅ Theoretically optimal for normal distributions
- ✅ Works with any model architecture

**Weaknesses**:
- ❌ Slower inference (no optimized CUDA kernels)
- ❌ Slightly lower quality than AWQ
- ❌ Higher memory usage during inference (dynamic dequantization)

**Hyperparameters**:

```python
{
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",        # Use NF4 (vs "fp4")
    "bnb_4bit_use_double_quant": True,   # Double quantization
    "bnb_4bit_compute_dtype": "bfloat16", # Compute in BF16
}
```

---

## Head-to-Head Comparison

### Quantization Time (Llama 3 70B)

| Method | Time | GPU Needed | Process |
|--------|------|------------|---------|
| **GPTQ** | 90-120 min | 1x A100 80GB | Hessian computation + quantization |
| **AWQ** | 15-30 min | 1x A100 80GB | Activation profiling + quantization |
| **NF4** | < 1 min | Any GPU | On-the-fly during model load |

**Winner**: NF4 (instant), but AWQ good middle ground

---

### Inference Speed (Llama 3 70B, 2K input + 100 output)

**Hardware**: ml.g5.12xlarge (4x A10G 24GB)

| Method | Tokens/sec | Latency P50 | Throughput |
|--------|------------|-------------|------------|
| **GPTQ (ExLlama kernel)** | 42 tok/sec | 240ms | 95 req/sec |
| **AWQ (GEMM kernel)** | 48 tok/sec | 210ms | 108 req/sec |
| **NF4 (bitsandbytes)** | 31 tok/sec | 320ms | 62 req/sec |

**Winner**: AWQ (13% faster than GPTQ, 55% faster than NF4)

---

### Quality Comparison (Llama 3 70B)

#### Perplexity (WikiText-2)

| Method | Perplexity | Δ vs BF16 |
|--------|------------|-----------|
| **BF16 Baseline** | 5.12 | 0% |
| **AWQ** | 5.42 | +5.9% |
| **NF4** | 5.67 | +10.7% |
| **GPTQ** | 5.89 | +15.0% |

**Winner**: AWQ (best 4-bit quality)

#### MMLU Accuracy

| Method | 0-shot | 5-shot | Δ vs BF16 |
|--------|--------|--------|-----------|
| **BF16** | 79.2% | 82.4% | 0% |
| **AWQ** | 77.1% | 80.8% | -2.1% |
| **NF4** | 76.2% | 79.5% | -3.0% |
| **GPTQ** | 75.4% | 78.9% | -3.5% |

**Winner**: AWQ (closest to baseline)

#### GSM8K (Math Reasoning)

| Method | Accuracy | Δ vs BF16 |
|--------|----------|-----------|
| **BF16** | 84.5% | 0% |
| **AWQ** | 81.3% | -3.2% |
| **NF4** | 79.8% | -4.7% |
| **GPTQ** | 78.6% | -5.9% |

**Winner**: AWQ (least degradation in math)

---

### Model Size on Disk

**Llama 3 70B**:

| Method | Model Size | Δ vs BF16 | Notes |
|--------|------------|-----------|-------|
| **BF16** | 140 GB | 0% | Baseline |
| **GPTQ** | 35.2 GB | -75% | Includes Hessian info |
| **AWQ** | 35.0 GB | -75% | Includes scaling factors |
| **NF4** | 35.5 GB | -74.6% | Includes absmax values |

**Winner**: Tie (all ~35 GB, differences negligible)

---

### Calibration Dataset Requirements

| Method | Samples Needed | Dataset | Time |
|--------|----------------|---------|------|
| **GPTQ** | 128-1024 | WikiText, C4 | High quality needed |
| **AWQ** | 128 | WikiText, Alpaca | Less sensitive |
| **NF4** | 0 | None | No calibration |

**Winner**: NF4 (no calibration), AWQ for calibration methods

---

## Use Case Recommendations

### Choose GPTQ When:

1. **Base model quantization** (not instruction-tuned)
   - GPTQ slightly better for pre-trained models

2. **Batch inference** (latency less critical)
   - GPTQ's slight speed disadvantage doesn't matter

3. **You have good calibration data**
   - GPTQ benefits from high-quality calibration

4. **Widely used model** (pre-quantized versions available)
   - TheBloke has GPTQ versions for most models

**Example**:
```python
# Use pre-quantized GPTQ model
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-3-70B-GPTQ",
    device_map="auto",
    revision="gptq-4bit-128g-actorder_True",
)
```

---

### Choose AWQ When:

1. **Instruction-tuned / chat models**
   - AWQ preserves instruction-following better

2. **Latency-critical applications**
   - AWQ has best inference speed

3. **Production deployment**
   - Best quality + performance combination

4. **Need maximum quality at 4-bit**
   - AWQ consistently beats GPTQ in benchmarks

**Example**:
```python
# Quantize and deploy with AWQ
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3-70b-Instruct")
model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128})
model.save_quantized("./llama-3-70b-awq")

# Deploy to SageMaker
model = HuggingFaceModel(
    model_data="s3://bucket/llama-3-70b-awq.tar.gz",
    image_uri="...",
)
predictor = model.deploy(instance_type="ml.g5.12xlarge")
```

---

### Choose NF4 (bitsandbytes) When:

1. **Need to fine-tune** (QLoRA)
   - Only option supporting post-quantization fine-tuning

2. **Quick experimentation**
   - Instant quantization, no calibration

3. **No calibration dataset available**
   - Works out of the box

4. **Simple setup preferred**
   - One line of code

**Example**:
```python
# Fine-tune 70B on single GPU with QLoRA
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70b-hf",
    quantization_config=bnb_config,
)

# Add LoRA and fine-tune
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Train on single A100!
trainer.train()
```

---

## Combining Methods

### Hybrid Approach: AWQ for Base + QLoRA for Fine-Tuning

**Problem**: AWQ doesn't support fine-tuning, NF4 is slower for inference

**Solution**: Use both!

```python
# Step 1: Start with pre-quantized AWQ model (best quality)
base_model = AutoAWQForCausalLM.from_quantized("llama-3-70b-awq")

# Step 2: For fine-tuning, re-quantize with NF4 + LoRA
model_for_training = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70b-hf",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)
model_for_training = get_peft_model(model_for_training, lora_config)

# Step 3: Train
trainer.train()
model_for_training.save_pretrained("./lora-adapters")

# Step 4: Merge LoRA back to FP16, then re-quantize with AWQ
model_fp16 = merge_lora_weights(base_fp16_model, lora_adapters)
model_awq = quantize_with_awq(model_fp16)  # Best quality for deployment

# Step 5: Deploy AWQ model (fast inference)
```

---

## Summary Table

| Criterion | GPTQ | AWQ | NF4 |
|-----------|------|-----|-----|
| **Quality** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Ease** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Fine-tuning** | ❌ | ❌ | ✅ |
| **Calibration** | Required | Required | Not needed |
| **Best For** | Base models | Instruction models | QLoRA |

**Overall Winner: AWQ** (best quality + speed for production)
**Most Versatile: NF4** (supports fine-tuning + easiest)

---

## Quantization Checklist

Before quantizing your model:

- [ ] Do you have H100? → Use FP8 instead
- [ ] Can you accept <2% quality loss? → Use INT8 instead
- [ ] Need to fine-tune after? → Use NF4 (only option)
- [ ] Is it instruction-tuned? → Prefer AWQ
- [ ] Have calibration data? → AWQ or GPTQ
- [ ] No calibration data? → Use NF4
- [ ] Need maximum speed? → Use AWQ
- [ ] Need maximum quality? → Use AWQ
- [ ] Quick prototype? → Use NF4

**Most Common Choice**: AWQ (best balance)
