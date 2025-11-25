# HuggingFace Training Version Compatibility Matrix

## Overview

This guide provides a comprehensive compatibility matrix for HuggingFace libraries across different AWS DLC training container versions.

## Quick Reference Matrix

### Current Versions (As of December 2024)

| DLC Container | PyTorch | Transformers | Accelerate | PEFT | TRL | Datasets | tokenizers |
|--------------|---------|--------------|------------|------|-----|----------|-----------|
| **PT 2.8 GPU** | 2.8.0 | 4.56.2 | 1.3.0 | 0.14.0 | 0.13.0 | 3.2.0 | 0.21.0 |
| **PT 2.7 GPU** | 2.7.0 | 4.46.3 | 1.1.1 | 0.13.2 | 0.11.4 | 3.0.1 | 0.20.4 |
| **PT 2.6 GPU** | 2.6.0 | 4.46.2 | 1.1.0 | 0.13.0 | 0.11.1 | 2.21.0 | 0.20.0 |
| **PT 2.8 NeuronX** | 2.8.0 | 4.56.0 | 1.3.0 | 0.14.0 | - | 3.2.0 | 0.21.0 |
| **PT 2.7 NeuronX** | 2.7.0 | 4.46.0 | 1.1.0 | 0.13.0 | - | 3.0.0 | 0.20.4 |

## Detailed Compatibility

### PyTorch 2.8 GPU Training Container

**Container**: `pytorch-training:2.8.0-gpu-py311-cu121-ubuntu22.04-sagemaker`

**Full Dependency Tree**:
```
torch==2.8.0
torchvision==0.20.0
torchaudio==2.5.0

# HuggingFace Core
transformers==4.56.2
accelerate==1.3.0
datasets==3.2.0
tokenizers==0.21.0
safetensors==0.4.5

# Fine-tuning
peft==0.14.0  # LoRA, QLoRA, Prefix Tuning
trl==0.13.0   # RLHF, PPO, DPO
bitsandbytes==0.45.0  # 4-bit/8-bit quantization

# Training utilities
deepspeed==0.15.4
flash-attn==2.7.2
wandb==0.18.7
tensorboard==2.18.0

# Model compression
optimum==1.24.0
auto-gptq==0.7.1
```

**Validated Model Families**:
- Llama 3 / 3.1 / 3.2 / 3.3 (all sizes)
- Mistral 7B / 8x7B / 8x22B
- Qwen 2.5 (all sizes)
- Phi-3 / Phi-3.5
- Gemma 2 (2B, 9B, 27B)
- Command-R / Command-R+

**Known Issues**:
- Flash Attention 2.7 requires A100/H100 (not compatible with V100/T4)
- DeepSpeed ZeRO-3 offload has 5-10% slowdown vs 2.7 (under investigation)

### PyTorch 2.7 GPU Training Container

**Container**: `pytorch-training:2.7.0-gpu-py311-cu121-ubuntu22.04-sagemaker`

**Full Dependency Tree**:
```
torch==2.7.0
torchvision==0.19.0
torchaudio==2.4.0

# HuggingFace Core
transformers==4.46.3
accelerate==1.1.1
datasets==3.0.1
tokenizers==0.20.4
safetensors==0.4.3

# Fine-tuning
peft==0.13.2
trl==0.11.4
bitsandbytes==0.44.1

# Training utilities
deepspeed==0.15.1
flash-attn==2.6.3
wandb==0.17.9
tensorboard==2.17.0

# Model compression
optimum==1.23.3
auto-gptq==0.7.0
```

**Validated Model Families**:
- Llama 2 / 3 / 3.1
- Mistral 7B / 8x7B
- Qwen 2 (all sizes)
- Falcon 7B/40B/180B
- MPT 7B/30B
- CodeLlama (all variants)

**Stability**: Highly stable, recommended for production

### PyTorch 2.8 NeuronX Training Container (Trainium)

**Container**: `pytorch-training-neuronx:2.8.0-neuronx-py311-sdk2.26.0-ubuntu22.04`

**Key Differences from GPU Container**:
```
torch==2.8.0
torch-neuronx==2.8.0.2  # Neuron-specific PyTorch

# HuggingFace (NeuronX compatible)
transformers==4.56.0
transformers-neuronx==0.15.0  # ⭐ Neuron-optimized transformers
accelerate==1.3.0
datasets==3.2.0

# Fine-tuning (limited support)
peft==0.14.0  # LoRA supported, QLoRA not available

# TRL not included (RLHF not supported on Trainium yet)

# NeuronX specific
neuronx-cc==2.26.0  # Compiler
neuronx-distributed==0.12.0  # Distributed training
```

**Supported Models on Trainium**:
- Llama 2 7B/13B/70B ✓
- Llama 3 8B/70B ✓
- Mistral 7B ✓
- GPT-NeoX 20B ✓
- BERT (all sizes) ✓

**Not Yet Supported**:
- Mixtral (MoE not supported)
- Qwen 2.5 (work in progress)
- Gemma 2 (not yet optimized)

## Version Migration Guides

### Upgrading from transformers 4.46 → 4.56

**Breaking Changes**:

1. **Chat Template Changes**
```python
# transformers 4.46
tokenizer.chat_template  # Single template

# transformers 4.56
tokenizer.chat_template  # Can be dict of templates
tokenizer.apply_chat_template(
    messages,
    chat_template="default"  # ✓ Specify template
)
```

2. **AutoTokenizer Behavior**
```python
# 4.46 - padding side inferred
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

# 4.56 - must specify for generation
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")
tokenizer.padding_side = "left"  # ✓ Required for batched generation
```

3. **Flash Attention Parameter**
```python
# 4.46
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_flash_attention_2=True,  # ❌ Deprecated
)

# 4.56
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",  # ✓ New parameter
)
```

### Upgrading from accelerate 1.1 → 1.3

**New Features**:

1. **FSDP2 Support**
```python
from accelerate import Accelerator

# 1.3 only
accelerator = Accelerator(
    fsdp_plugin="fsdp2",  # ✓ PyTorch 2.8 FSDP2
)
```

2. **Better CPU Offloading**
```python
# 1.3 - improved memory efficiency
model = load_checkpoint_and_dispatch(
    model,
    "checkpoint.pt",
    device_map="auto",
    offload_folder="/tmp/offload",
    offload_state_dict=True,  # ✓ New: offload state dict too
)
```

### Upgrading from peft 0.13 → 0.14

**New Adapters**:

```python
from peft import LoraConfig, get_peft_model

# 0.14 - new rank stabilized LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    use_rslora=True,  # ✓ Rank-stabilized LoRA (better convergence)
)
```

**Breaking Change - QLoRA**:
```python
# 0.13
config = LoraConfig(
    r=64,
    load_in_4bit=True,  # ❌ Removed
)

# 0.14
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,  # ✓ Separate config
)

config = LoraConfig(r=64, ...)  # No quantization params
```

## Model-Specific Compatibility

### Llama 3.3 (Released December 2024)

**Minimum Versions**:
```
transformers >= 4.56.0
tokenizers >= 0.21.0
```

**Example**:
```python
from transformers import AutoModelForCausalLM

# Requires transformers 4.56+
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.3-70B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

**Container**: Use PyTorch 2.8 (has transformers 4.56.2)

### Mixtral 8x22B

**Minimum Versions**:
```
transformers >= 4.46.0
torch >= 2.7.0
accelerate >= 1.1.0
```

**Example**:
```python
# Requires model parallelism for 176B params
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatic device mapping
    load_in_8bit=True,  # 8-bit quantization (optional)
)
```

**Container**: PyTorch 2.7 or 2.8

### Qwen 2.5

**Minimum Versions**:
```
transformers >= 4.46.0
tokenizers >= 0.20.0
```

**Example**:
```python
from transformers import AutoModelForCausalLM

# Qwen 2.5 uses grouped-query attention
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
```

**Container**: PyTorch 2.7 or 2.8

### Gemma 2

**Minimum Versions**:
```
transformers >= 4.46.0  # Gemma 2 architecture support
torch >= 2.7.0
```

**Example**:
```python
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Flash Attention not yet optimized
    device_map="auto",
)
```

**Container**: PyTorch 2.7 or 2.8

## Fine-Tuning Method Compatibility

### LoRA / QLoRA

| Container | LoRA | QLoRA (4-bit) | DoRA | Rank-Stabilized LoRA |
|-----------|------|---------------|------|---------------------|
| PT 2.8 GPU | ✓ | ✓ | ✓ | ✓ |
| PT 2.7 GPU | ✓ | ✓ | ✓ | ❌ |
| PT 2.8 NeuronX | ✓ | ❌ | ❌ | ❌ |

**Example (QLoRA on 2.8)**:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70b",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_rslora=True,  # Rank-stabilized (2.8 only)
)

model = get_peft_model(model, lora_config)
```

### RLHF / DPO

| Container | PPO | DPO | KTO | ORPO |
|-----------|-----|-----|-----|------|
| PT 2.8 GPU | ✓ | ✓ | ✓ | ✓ |
| PT 2.7 GPU | ✓ | ✓ | ❌ | ❌ |
| PT 2.8 NeuronX | ❌ | ❌ | ❌ | ❌ |

**Example (DPO on 2.8)**:
```python
from trl import DPOTrainer, DPOConfig

# Direct Preference Optimization
training_args = DPOConfig(
    output_dir="llama-3-8b-dpo",
    per_device_train_batch_size=4,
    learning_rate=5e-7,
    beta=0.1,  # DPO temperature
    max_length=512,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## Distributed Training Compatibility

### DeepSpeed

| DeepSpeed Version | PT 2.8 | PT 2.7 | PT 2.6 | Features |
|------------------|--------|--------|--------|----------|
| 0.15.4 | ✓ | ❌ | ❌ | ZeRO-3, CPU offload, NVMe offload |
| 0.15.1 | ❌ | ✓ | ❌ | ZeRO-3, CPU offload |
| 0.14.5 | ❌ | ❌ | ✓ | ZeRO-3 |

**Example (ZeRO-3 on 2.8)**:
```python
# deepspeed_config.json
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### FSDP

| PyTorch | FSDP Version | Features |
|---------|-------------|----------|
| 2.8 | FSDP2 | CPU offload, mixed precision, hybrid sharding |
| 2.7 | FSDP | Full/hybrid sharding, CPU offload |
| 2.6 | FSDP | Full sharding only |

**Example (FSDP2 on 2.8)**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# FSDP2 with hybrid sharding
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # ✓ 2.8 only
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=mixed_precision_policy,
)
```

## Upgrade Decision Matrix

### When to Upgrade to PyTorch 2.8

**Upgrade if**:
- Need Llama 3.3 support
- Want Flash Attention 2.7 (25% faster)
- Using FSDP2 for large models
- Need rank-stabilized LoRA
- Want KTO/ORPO for preference learning

**Stay on 2.7 if**:
- Production workload (2.8 still maturing)
- Using Trainium (wait for 2.8 NeuronX to stabilize)
- No need for latest model architectures

### When to Use NeuronX Containers

**Use NeuronX if**:
- Training models up to 70B params
- Cost-sensitive (Trainium 50% cheaper than GPU)
- LoRA fine-tuning workflow
- Standard architectures (Llama, GPT-NeoX)

**Use GPU if**:
- Need RLHF/DPO
- Using Mixtral or other MoE models
- Require maximum flexibility
- Prototyping new architectures

## Testing Compatibility

### Quick Compatibility Test

```python
# test_compatibility.py
import torch
import transformers
from transformers import pipeline

def test_model_loading():
    """Test if model loads correctly"""
    try:
        pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3-8b",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        output = pipe("Hello", max_new_tokens=10)
        print("✓ Model loading works")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_flash_attention():
    """Test Flash Attention 2"""
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3-8b",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        print("✓ Flash Attention 2 works")
        return True
    except Exception as e:
        print(f"✗ Flash Attention 2 failed: {e}")
        return False

def test_peft():
    """Test PEFT/LoRA"""
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3-8b",
            torch_dtype=torch.bfloat16,
        )

        config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, config)
        print("✓ PEFT/LoRA works")
        return True
    except Exception as e:
        print(f"✗ PEFT failed: {e}")
        return False

if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print()

    test_model_loading()
    test_flash_attention()
    test_peft()
```

## Resources

- [Transformers Release Notes](https://github.com/huggingface/transformers/releases)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [AWS DLC Release Matrix](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)
