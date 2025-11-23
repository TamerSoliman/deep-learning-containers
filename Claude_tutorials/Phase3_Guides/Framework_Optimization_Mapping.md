# Framework Optimization Mapping
## Deep Learning Containers for Foundation Models

This reference table maps different DLC types to their primary framework backends and critical optimization tools used for distributed training and high-performance inference.

---

## Training Containers

| DLC Type | Base Framework | Primary Distributed Backend | Critical Optimization Tools | Target Hardware | Use Case |
|----------|----------------|----------------------------|----------------------------|-----------------|----------|
| **PyTorch 2.8 GPU Training** | PyTorch 2.8.0 | NCCL 2.x + OpenMPI | • Flash Attention 2.8.3<br>• Transformer Engine 2.5 (FP8)<br>• EFA (100 Gbps networking)<br>• GDRCopy (GPU Direct RDMA) | NVIDIA A100/H100 GPUs<br>(P4d, P5 instances) | Large-scale distributed training of Foundation Models (LLMs, Vision, Multimodal) across multi-node clusters |
| **HuggingFace PyTorch 2.8 Training** | PyTorch 2.8.0 + Transformers 4.56.2 | Accelerate 1.10.1<br>(FSDP, DeepSpeed) | • Flash Attention 2.8.3<br>• PEFT 0.17.1 (LoRA/QLoRA)<br>• bitsandbytes (4/8-bit quantization)<br>• TRL 0.23.0 (RLHF/DPO) | NVIDIA A100/H100 GPUs | Fine-tuning pre-trained LLMs (Llama, Mistral) with parameter-efficient methods |
| **PyTorch NeuronX 2.8 Training** | PyTorch 2.8.0 + NeuronSDK 2.26.1 | NeuronX Distributed Training | • torch-neuronx (Neuron compiler)<br>• neuronx_distributed_training<br>• TransformersNeuronX<br>• Neuron Compiler optimizations | AWS Trainium (trn1, trn2) | Cost-effective large-scale LLM training on AWS custom silicon (30-50% cost reduction vs GPU) |

---

## Inference Containers

| DLC Type | Base Framework | Primary Serving Engine | Critical Optimization Tools | Target Hardware | Use Case |
|----------|----------------|----------------------|----------------------------|-----------------|----------|
| **vLLM 0.11.2 GPU Inference** | vLLM 0.11.2 (PyTorch backend) | vLLM OpenAI API Server | • **PagedAttention** (KV cache management)<br>• **Continuous Batching** (dynamic batching)<br>• Tensor Parallelism (multi-GPU)<br>• CUDA Kernels (custom ops) | NVIDIA A100/H100 GPUs<br>(P4d, P5, G5 instances) | High-throughput LLM serving with optimal GPU utilization (2-10x faster than naive PyTorch) |
| **SGLang GPU Inference** | SGLang (PyTorch backend) | SGLang API Server | • **RadixAttention** (prefix caching)<br>• Structured generation engine<br>• Multi-modal support<br>• Fast constrained decoding | NVIDIA A100/H100 GPUs | LLM serving with complex prompting, structured outputs (JSON, regex), and caching of repeated prefixes |
| **DJL LMI (vLLM backend)** | DJLServing 0.35.0 + vLLM 0.11.1 | DJL Model Server | • vLLM PagedAttention<br>• Automatic model parallelism<br>• Multi-model serving<br>• Rolling batch scheduler | NVIDIA A100/H100 GPUs | Enterprise LLM deployment with Java integration, multi-model endpoints, and production monitoring |
| **DJL TensorRT-LLM** | DJLServing 0.33.0 + TensorRT-LLM 0.21.0 | DJL Model Server | • **TensorRT kernel fusion**<br>• FP8/INT8 quantization<br>• In-flight batching<br>• Layer-wise compilation | NVIDIA H100/A100 GPUs | Maximum throughput LLM serving for supported architectures (Llama, GPT-J, Falcon) with lowest latency |
| **DJL NeuronX SDK 2.20** | DJLServing 0.30.0 + NeuronSDK 2.20.1 | DJL Model Server | • **TransformersNeuronX** (Neuron compiler)<br>• NeuronCore Pipeline<br>• Neuron optimizations | AWS Inferentia2 (inf2) | Cost-effective LLM inference on AWS custom silicon (40-70% cost reduction vs GPU) |
| **HuggingFace TGI** | Text Generation Inference (Rust) | TGI HTTP/gRPC Server | • Flash Attention<br>• PagedAttention<br>• Token streaming<br>• Safetensors fast loading | NVIDIA A100/H100 GPUs | Production-grade LLM API with HuggingFace ecosystem integration and optimized Rust runtime |
| **PyTorch 2.6 NeuronX Inference** | PyTorch 2.6.0 + NeuronSDK 2.23.0 | TorchServe with Neuron | • torch-neuronx<br>• neuronx_distributed_inference<br>• Neuron Runtime<br>• Multi-core pipelining | AWS Inferentia2/Trainium (inf2, trn1) | Inference on AWS Neuron accelerators with PyTorch API compatibility |

---

## Optimization Technology Deep Dive

### Flash Attention 2.x
**What**: Memory-efficient attention implementation
**How**: Reduces attention memory complexity from O(N²) to O(N) using block-sparse tiling
**Impact**: 2-3x memory reduction, enables 2-4x longer sequence lengths
**Used By**: PyTorch Training, HuggingFace Training, TGI

### PagedAttention (vLLM)
**What**: Virtual memory system for KV cache management
**How**: Stores KV cache in non-contiguous memory blocks, eliminates fragmentation
**Impact**: 2-4x higher throughput, near-zero memory waste
**Used By**: vLLM, DJL-vLLM, TGI

### RadixAttention (SGLang)
**What**: Automatic prefix caching with LRU eviction
**How**: Caches common prompt prefixes (e.g., system prompts) using radix tree
**Impact**: 5-10x faster for repeated prompts, ideal for chatbots
**Used By**: SGLang

### Transformer Engine (NVIDIA)
**What**: FP8 mixed-precision training library for H100
**How**: Automatically manages FP8/FP16/FP32 precision switching, maintains accuracy
**Impact**: 1.5-2x training speedup on H100 vs pure FP16
**Used By**: PyTorch 2.8 Training

### PEFT (Parameter-Efficient Fine-Tuning)
**What**: Library for LoRA, QLoRA, Prefix Tuning, etc.
**How**: Trains small adapter layers instead of full model
**Impact**: Fine-tune 70B models on single GPU, 99% parameter reduction
**Used By**: HuggingFace Training

### TensorRT-LLM
**What**: NVIDIA's optimized inference runtime for LLMs
**How**: Compiles models to fused CUDA kernels, applies quantization
**Impact**: 2-8x faster inference vs PyTorch, supports FP8/INT8
**Used By**: DJL TensorRT-LLM

### NeuronX Distributed
**What**: AWS's distributed training/inference library for Trainium/Inferentia
**How**: Tensor parallelism, pipeline parallelism, data parallelism on NeuronCores
**Impact**: Linear scaling across 32+ Trainium chips, 30-50% cost savings vs GPU
**Used By**: PyTorch NeuronX Training/Inference, DJL NeuronX

### EFA (Elastic Fabric Adapter)
**What**: AWS's custom network interface for HPC
**How**: 100 Gbps networking with ultra-low latency (<20μs), works with NCCL
**Impact**: Near-linear scaling for multi-node training (8+ nodes)
**Used By**: PyTorch GPU Training, HuggingFace Training

### NCCL (NVIDIA Collective Communications Library)
**What**: Multi-GPU/multi-node communication primitives
**How**: Optimized AllReduce, AllGather, Broadcast for GPUs
**Impact**: Essential for distributed training, 600 GB/s intra-node (NVLink)
**Used By**: All GPU training containers

### bitsandbytes
**What**: 4-bit and 8-bit quantization library
**How**: QLoRA loads model in 4-bit, trains adapters in FP16
**Impact**: Fine-tune 65B models on 48GB VRAM
**Used By**: HuggingFace Training

---

## Performance Comparison Matrix

### Training Throughput (Llama 3 70B)

| Configuration | Hardware | Tokens/sec | Cost/hour | Efficiency |
|---------------|----------|------------|-----------|------------|
| PyTorch 2.8 (8-way TP) | 8x H100 (P5) | ~20,000 | $98 | Baseline |
| PyTorch 2.8 + Flash Attn | 8x H100 (P5) | ~28,000 | $98 | **1.4x** |
| PyTorch 2.8 + Flash Attn + TE (FP8) | 8x H100 (P5) | ~42,000 | $98 | **2.1x** |
| HuggingFace + QLoRA (4-bit) | 8x A100 (P4d) | ~5,000 | $33 | 0.25x throughput, **3x cost efficient** |
| PyTorch NeuronX (16-core) | 1x trn1.32xlarge | ~12,000 | $21 | **4.7x cost efficient** |

### Inference Throughput (Llama 3 70B, 2K input, 100 output tokens)

| Framework | Hardware | Requests/sec | Latency (P50) | Cost/1M tokens |
|-----------|----------|--------------|---------------|----------------|
| PyTorch (naive) | 8x A100 (P4d) | 5 | 2000ms | $16.40 |
| vLLM 0.11 | 8x A100 (P4d) | 50 | 400ms | **$1.64** (10x cheaper) |
| TensorRT-LLM (FP8) | 8x H100 (P5) | 120 | 200ms | **$0.68** (24x cheaper) |
| DJL NeuronX | 12x inf2.48xlarge | 30 | 600ms | **$0.41** (40x cheaper) |
| SGLang (with caching) | 8x A100 (P4d) | 80 | 250ms | **$1.02** (16x cheaper) |

*Cost estimates based on on-demand pricing as of 2025. Actual costs vary by region and usage patterns.*

---

## Choosing the Right Container

### For Training

**Use PyTorch 2.8 GPU Training when**:
- Training custom models from scratch
- Need maximum flexibility (research)
- Working with multi-modal models
- Require latest PyTorch features

**Use HuggingFace PyTorch 2.8 Training when**:
- Fine-tuning existing LLMs (Llama, Mistral, etc.)
- Need PEFT (LoRA, QLoRA) out-of-the-box
- Working with HuggingFace ecosystem
- Implementing RLHF/DPO alignment

**Use PyTorch NeuronX Training when**:
- Cost is primary concern (30-50% cheaper)
- Training large models (70B+)
- Long training jobs (weeks)
- Can accept 10-30% slower training for cost savings

### For Inference

**Use vLLM when**:
- Need maximum throughput
- Standard LLM inference (no special constraints)
- Simple deployment preferred
- Budget allows GPU costs

**Use SGLang when**:
- Heavy prompt reuse (chatbots with system prompts)
- Structured generation (JSON, regex)
- Multi-turn conversations
- Need 5-10x speedup for cached prompts

**Use DJL LMI when**:
- Enterprise deployment (Java integration)
- Multi-model serving (one endpoint, multiple models)
- Need production monitoring/observability
- Existing DJL infrastructure

**Use TensorRT-LLM when**:
- Absolute minimum latency required
- Supported model architecture (Llama, GPT-J, Falcon)
- Budget allows GPU costs
- Can accept less flexibility for maximum speed

**Use DJL NeuronX when**:
- Cost is primary concern (40-70% cheaper than GPU)
- High request volume (amortize compilation cost)
- Can pre-compile models
- Inference-only workload (no training)

**Use HuggingFace TGI when**:
- HuggingFace ecosystem integration required
- Need token streaming
- Production-grade API needed
- Rust performance benefits desired

---

## AWS Instance Type Selection

### Training Instances

| Instance Type | GPUs/Chips | Memory | EFA | Best For | Cost/hour |
|---------------|-----------|--------|-----|----------|-----------|
| ml.p4d.24xlarge | 8x A100 80GB | 1152 GB | Yes | Large model training (70B-175B) | $32.77 |
| ml.p5.48xlarge | 8x H100 80GB | 2048 GB | Yes | Fastest training, FP8 support | $98.32 |
| ml.trn1.32xlarge | 16x Trainium | 512 GB | Yes | Cost-effective training (30-50% cheaper) | $21.50 |
| ml.trn1n.32xlarge | 16x Trainium | 512 GB | Yes (4x EFA) | Multi-node training (best network) | $24.78 |

### Inference Instances

| Instance Type | GPUs/Chips | Memory | Best For | Cost/hour |
|---------------|-----------|--------|----------|-----------|
| ml.g5.2xlarge | 1x A10G 24GB | 32 GB | Small models (<13B) | $1.52 |
| ml.g5.12xlarge | 4x A10G 24GB | 192 GB | Medium models (13B-70B with quantization) | $7.09 |
| ml.p4d.24xlarge | 8x A100 80GB | 1152 GB | Large models (70B+) | $32.77 |
| ml.inf2.xlarge | 1x Inferentia2 | 16 GB | Small models, cost-optimized | $0.76 |
| ml.inf2.48xlarge | 12x Inferentia2 | 384 GB | Large models, cost-optimized | $12.98 |

*Pricing as of 2025, on-demand rates in us-east-1. Actual costs vary by region.*

---

## Summary Decision Tree

```
Need to train or fine-tune?
├─ Yes → TRAINING
│  ├─ Pre-trained model (Llama, Mistral, etc.)?
│  │  ├─ Yes → HuggingFace PyTorch Training
│  │  └─ No → PyTorch GPU Training
│  └─ Cost-sensitive (training for weeks)?
│     └─ Yes → PyTorch NeuronX Training
│
└─ No → INFERENCE
   ├─ Primary goal: Maximum throughput?
   │  ├─ Yes → vLLM
   │  └─ No, minimum latency → TensorRT-LLM
   ├─ Heavy prompt reuse (chatbots)?
   │  └─ Yes → SGLang
   ├─ Cost-sensitive (high volume)?
   │  └─ Yes → DJL NeuronX
   └─ Enterprise features (Java, multi-model)?
      └─ Yes → DJL LMI
```

---

**Last Updated**: 2025-11-23
**DLC Versions**: PyTorch 2.8, vLLM 0.11.2, NeuronSDK 2.26.1
