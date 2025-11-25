# Phase 1: DLC Discovery & Curation Plan for Foundation Models

## Repository Analysis Summary

This document outlines the comprehensive discovery and analysis plan for AWS Deep Learning Containers (DLCs) optimized for Foundation Models including Large Language Models (LLMs), Vision-Language Models, and Diffusion Models.

---

## 1. Identified Foundation Model DLC Categories

### Training Containers (Complex, Multi-GPU/Multi-Node Optimized)

#### 1.1 PyTorch Native Training DLCs
- **PyTorch 2.8.0 GPU Training (CUDA 12.9)**
  - Path: `pytorch/training/docker/2.8/py3/cu129/Dockerfile.gpu`
  - Framework: PyTorch 2.8.0
  - Task: Distributed Training (LLMs, Vision Models)
  - Optimization: NCCL, EFA, Flash Attention 2.8.3, Transformer Engine 2.5
  - Target: Multi-GPU distributed training on P4d/P5 instances

- **PyTorch 2.7.1 GPU Training (CUDA 12.8)**
  - Path: `pytorch/training/docker/2.7/py3/cu128/Dockerfile.gpu`
  - Framework: PyTorch 2.7.1
  - Task: Distributed Training (LLMs, Vision Models)
  - Optimization: NCCL, EFA, Flash Attention
  - Target: Multi-GPU distributed training

- **PyTorch 2.7.0 ARM64 GPU Training**
  - Path: `pytorch/training/docker/2.7/arm64/py3/cu128/Dockerfile.gpu`
  - Framework: PyTorch 2.7.0
  - Task: Distributed Training on ARM architecture
  - Optimization: ARM64-optimized NCCL, EFA
  - Target: AWS Graviton-based GPU instances

#### 1.2 HuggingFace Training DLCs (LLM & Multimodal Focused)
- **HuggingFace PyTorch 2.8.0 GPU Training**
  - Path: `huggingface/pytorch/training/docker/2.8/py3/cu129/Dockerfile.gpu`
  - Framework: PyTorch 2.8.0 + Transformers 4.56.2
  - Task: LLM Fine-tuning, Multimodal Training
  - Model Types: LLMs (Llama, Mistral), Vision-Language (CLIP, LLaVA), Diffusion (Stable Diffusion)
  - Key Libraries: Transformers, Accelerate 1.10.1, PEFT 0.17.1, TRL 0.23.0, Diffusers 0.35.1
  - Optimization: Flash Attention 2.8.3, Distributed training via Accelerate
  - Target: SageMaker training jobs for Foundation Models

- **HuggingFace PyTorch 2.5.1 GPU Training**
  - Path: `huggingface/pytorch/training/docker/2.5/py3/cu124/Dockerfile.gpu`
  - Framework: PyTorch 2.5.1 + Transformers 4.49.0
  - Task: LLM Fine-tuning
  - Model Types: LLMs, Vision-Language Models
  - Key Libraries: Transformers, Accelerate, PEFT, TRL
  - Optimization: Flash Attention, LoRA/QLoRA support

#### 1.3 AWS Neuron Training DLCs (Trainium/Inferentia2)
- **HuggingFace PyTorch 2.8.0 NeuronX Training**
  - Path: `huggingface/pytorch/training/docker/2.8/py3/sdk2.26.0/Dockerfile.neuronx`
  - Framework: PyTorch 2.8.0 + NeuronSDK 2.26.0 + Transformers 4.55.4
  - Task: LLM Training on AWS Trainium
  - Model Types: LLMs optimized for Neuron
  - Optimization: NeuronX Distributed, TransformersNeuronX, NeuronX Distributed Training
  - Target: trn1/trn2 instances for cost-effective LLM training

- **HuggingFace PyTorch 2.7.0 NeuronX Training**
  - Path: `huggingface/pytorch/training/docker/2.7/py3/sdk2.24.1/Dockerfile.neuronx`
  - Framework: PyTorch 2.7.0 + NeuronSDK 2.24.1 + Transformers 4.51.0
  - Task: LLM Training on AWS Trainium
  - Optimization: NeuronX Distributed Training, optimized for trn1 instances

- **PyTorch 2.8.0 NeuronX Training (Native)**
  - Path: Available via GitHub (external link in available_images.md)
  - Framework: PyTorch 2.8.0 + NeuronSDK 2.26.1
  - Task: Custom distributed training on Trainium
  - Optimization: torch-neuronx, neuronx_distributed_training
  - Target: trn1/trn2 instances

- **PyTorch 2.7.0 NeuronX Training (Native)**
  - Path: Available via GitHub (external link)
  - Framework: PyTorch 2.7.0 + NeuronSDK 2.25.0
  - Task: Custom distributed training on Trainium
  - Optimization: torch-neuronx, neuronx_distributed_training

#### 1.4 Specialized Training DLCs
- **StabilityAI Training Container** (if available in training path)
  - Focus: Diffusion model training
  - Model Types: Stable Diffusion variants

---

### Inference Containers (High-Performance Serving)

#### 2.1 High-Performance LLM Serving (vLLM-based)
- **vLLM 0.11.2 GPU Inference**
  - Path: `vllm/x86_64/gpu/Dockerfile`
  - Framework: vLLM 0.11.2 (OpenAI-compatible API)
  - Task: LLM Inference with PagedAttention
  - Model Types: LLMs (Llama, Mistral, GPT-style models)
  - Optimization: PagedAttention, Continuous Batching, KV Cache optimization
  - Target: Real-time LLM serving on g5/p4 instances
  - Entry Point: `vllm/build_artifacts/sagemaker_entrypoint.sh`

- **vLLM ARM64 GPU Inference**
  - Path: `vllm/arm64/gpu/Dockerfile.arm64`
  - Framework: vLLM on ARM64 architecture
  - Task: LLM Inference on ARM-based GPUs
  - Optimization: ARM64-optimized PagedAttention
  - Target: ARM-based GPU instances

#### 2.2 Advanced LLM Serving (SGLang)
- **SGLang GPU Inference**
  - Path: `sglang/x86_64/gpu/Dockerfile`
  - Framework: SGLang with RadixAttention
  - Task: Fast LLM Inference with structured generation
  - Model Types: LLMs with complex prompting
  - Optimization: RadixAttention (prefix caching), fast structured generation
  - Target: Interactive LLM applications
  - Entry Point: `sglang/build_artifacts/sagemaker_entrypoint.sh`

#### 2.3 Large Model Inference (LMI/DJL-based)
- **DJL LMI 17.0.0 with vLLM 0.11.1**
  - Image: `djl-inference:0.35.0-lmi17.0.0-cu128`
  - Framework: DJLServing + vLLM backend
  - Task: Enterprise LLM Inference
  - Model Types: Large LLMs (70B+)
  - Optimization: vLLM backend, automatic model parallelism
  - Target: Production LLM deployments

- **DJL LMI 16.0.0 with vLLM 0.10.2**
  - Image: `djl-inference:0.34.0-lmi16.0.0-cu128`
  - Framework: DJLServing + vLLM backend
  - Task: Enterprise LLM Inference

- **DJL TensorRT-LLM 0.21.0**
  - Image: `djl-inference:0.33.0-tensorrtllm0.21.0-cu128`
  - Framework: DJLServing + TensorRT-LLM backend
  - Task: Optimized LLM Inference
  - Model Types: LLMs compiled with TensorRT-LLM
  - Optimization: TensorRT kernel fusion, INT8/FP8 quantization
  - Target: Maximum throughput for supported models

- **DJL NeuronX SDK 2.20.1**
  - Image: `djl-inference:0.30.0-neuronx-sdk2.20.1`
  - Framework: DJLServing + NeuronX backend
  - Task: LLM Inference on AWS Inferentia2
  - Optimization: TransformersNeuronX, Neuron compiler
  - Target: inf2 instances for cost-effective inference

#### 2.4 HuggingFace Inference DLCs
- **HuggingFace PyTorch 2.6.0 GPU Inference**
  - Path: `huggingface/pytorch/inference/docker/2.6/py3/cu124/Dockerfile.gpu`
  - Framework: PyTorch 2.6.0 + Transformers 4.49.0
  - Task: Standard Transformers Inference
  - Model Types: LLMs, Vision-Language, Diffusion
  - Target: Flexible inference for various model types

- **HuggingFace Text Generation Inference (TGI)**
  - Image: Available via releases
  - Framework: TGI (Rust-based optimized serving)
  - Task: Production LLM serving
  - Model Types: Decoder-only LLMs
  - Optimization: Flash Attention, Paged Attention, Token streaming
  - Target: Production-grade LLM APIs

- **HuggingFace NeuronX Inference 2.8.0**
  - Path: `huggingface/pytorch/inference/docker/2.8/py3/sdk2.26.0/Dockerfile.neuronx`
  - Framework: PyTorch 2.8.0 + NeuronSDK 2.26.0 + Transformers 4.55.4
  - Task: LLM Inference on AWS Inferentia2/Trainium
  - Model Types: LLMs compiled for Neuron
  - Optimization: NeuronX Distributed Inference, TransformersNeuronX
  - Target: inf2/trn1 instances

- **HuggingFace NeuronX TGI**
  - Image: Available via releases
  - Framework: TGI optimized for NeuronX
  - Task: Production LLM serving on Inferentia2
  - Target: inf2 instances

#### 2.5 PyTorch Native Inference DLCs
- **PyTorch 2.6.0 GPU Inference**
  - Path: `pytorch/inference/docker/2.6/py3/cu124/Dockerfile.gpu`
  - Framework: PyTorch 2.6.0 with TorchServe
  - Task: General-purpose model serving
  - Entry Point: `pytorch/inference/docker/build_artifacts/torchserve-entrypoint.py`

- **PyTorch 2.6.0 ARM64 GPU Inference**
  - Path: `pytorch/inference/docker/2.6/arm64/py3/cu124/Dockerfile.gpu`
  - Framework: PyTorch 2.6.0 on ARM64
  - Task: ARM-based inference

- **PyTorch 2.8.0 NeuronX Inference**
  - Framework: PyTorch 2.8.0 + NeuronSDK 2.26.1
  - Task: Neuron-optimized inference
  - Optimization: torch-neuronx, neuronx_distributed_inference
  - Target: inf2/trn1/trn2 instances

#### 2.6 Specialized Inference DLCs
- **StabilityAI PyTorch 2.0.1 SGM Inference**
  - Path: `stabilityai/pytorch/inference/...`
  - Framework: PyTorch 2.0.1 + SGM 0.1.0
  - Task: Diffusion Model Inference
  - Model Types: Stable Diffusion XL, SD 2.1
  - Target: Image generation endpoints

---

## 2. Selected DLCs for Deep Dive Analysis

### 2.1 Training DLCs (8 Complex Containers)

| # | Container | Framework | Path | Key Optimizations |
|---|-----------|-----------|------|-------------------|
| 1 | **PyTorch 2.8 GPU Training** | PyTorch 2.8.0 | `pytorch/training/docker/2.8/py3/cu129/Dockerfile.gpu` | NCCL, EFA, Flash Attention 2.8.3, TE 2.5 |
| 2 | **HuggingFace PyTorch 2.8 GPU** | PT 2.8 + Transformers | `huggingface/pytorch/training/docker/2.8/py3/cu129/Dockerfile.gpu` | Flash Attn, PEFT, TRL, Accelerate, Diffusers |
| 3 | **HuggingFace PyTorch 2.5 GPU** | PT 2.5 + Transformers | `huggingface/pytorch/training/docker/2.5/py3/cu124/Dockerfile.gpu` | Flash Attn, PEFT, LoRA/QLoRA |
| 4 | **HuggingFace NeuronX 2.8** | PT 2.8 + Neuron 2.26 | `huggingface/pytorch/training/docker/2.8/py3/sdk2.26.0/Dockerfile.neuronx` | NeuronX Distributed, TransformersNeuronX |
| 5 | **HuggingFace NeuronX 2.7** | PT 2.7 + Neuron 2.24 | `huggingface/pytorch/training/docker/2.7/py3/sdk2.24.1/Dockerfile.neuronx` | NeuronX Distributed Training |
| 6 | **PyTorch 2.7 ARM64 GPU** | PyTorch 2.7.0 ARM | `pytorch/training/docker/2.7/arm64/py3/cu128/Dockerfile.gpu` | ARM-optimized NCCL, EFA |
| 7 | **PyTorch 2.8 NeuronX Training** | PT 2.8 + Neuron 2.26 | External (GitHub) | torch-neuronx, neuronx_distributed_training |
| 8 | **PyTorch 2.7 NeuronX Training** | PT 2.7 + Neuron 2.25 | External (GitHub) | torch-neuronx, neuronx_distributed_training |

### 2.2 Inference DLCs (8 Complex Containers)

| # | Container | Framework | Path/Image | Key Optimizations |
|---|-----------|-----------|------------|-------------------|
| 1 | **vLLM 0.11.2 GPU** | vLLM 0.11.2 | `vllm/x86_64/gpu/Dockerfile` | PagedAttention, Continuous Batching |
| 2 | **SGLang GPU** | SGLang | `sglang/x86_64/gpu/Dockerfile` | RadixAttention, Prefix Caching |
| 3 | **DJL LMI vLLM 17.0** | DJL + vLLM 0.11.1 | `djl-inference:0.35.0-lmi17.0.0-cu128` | Auto model parallelism, vLLM backend |
| 4 | **DJL TensorRT-LLM 0.21** | DJL + TRT-LLM | `djl-inference:0.33.0-tensorrtllm0.21.0-cu128` | TensorRT kernel fusion, INT8/FP8 |
| 5 | **DJL NeuronX SDK 2.20** | DJL + NeuronX | `djl-inference:0.30.0-neuronx-sdk2.20.1` | TransformersNeuronX, Neuron compiler |
| 6 | **HuggingFace TGI** | TGI (Rust) | Release-based | Flash Attn, Paged Attn, Streaming |
| 7 | **HuggingFace NeuronX 2.8** | PT + Neuron + Transformers | `huggingface/pytorch/inference/docker/2.8/py3/sdk2.26.0/Dockerfile.neuronx` | NeuronX Distributed Inference |
| 8 | **vLLM ARM64 GPU** | vLLM ARM64 | `vllm/arm64/gpu/Dockerfile.arm64` | ARM-optimized PagedAttention |

---

## 3. Container Variety Analysis

### 3.1 Framework Distribution
- **PyTorch**: 12 containers (Training + Inference)
- **HuggingFace** (PyTorch-based): 8 containers
- **vLLM**: 2 containers (x86, ARM64)
- **SGLang**: 1 container
- **DJL/LMI**: 3 containers (vLLM, TensorRT-LLM, NeuronX backends)
- **Neuron/NeuronX**: 6 containers (Training + Inference)

### 3.2 Task Distribution
- **Training**: 8+ containers
- **Inference**: 8+ containers
- **Total Unique DLCs**: 17+ identified

### 3.3 Model Type Coverage
- **LLMs** (Large Language Models): All containers support (Llama, Mistral, GPT-style)
- **Vision-Language Models**: HuggingFace containers (CLIP, LLaVA, BLIP)
- **Diffusion Models**: HuggingFace with Diffusers, StabilityAI (Stable Diffusion, SDXL)
- **Vision Models**: PyTorch + HuggingFace containers (ViT, DINO, SAM)

### 3.4 Hardware/Accelerator Coverage
- **GPU (NVIDIA CUDA)**: 14 containers
- **ARM64 GPU**: 3 containers
- **AWS Neuron (Trainium/Inferentia2)**: 6 containers

---

## 4. Key Artifacts for Phase 2 Annotation

### 4.1 Training Container Artifacts
Each training container will have annotated:
1. **Dockerfile** - Multi-stage build, optimization layers, distributed training setup
2. Supporting scripts (if applicable):
   - `start_with_right_hostname.sh` - EFA network configuration
   - `dockerd_entrypoint.sh` - Container lifecycle management

### 4.2 Inference Container Artifacts
Each inference container will have annotated:
1. **Dockerfile** - Model serving setup, optimization libraries
2. **Entry Point Scripts**:
   - `sagemaker_entrypoint.sh` (vLLM, SGLang) - Environment variable parsing, model loading
   - `torchserve-entrypoint.py` (PyTorch) - TorchServe initialization
   - `neuron-entrypoint.py` (Neuron) - NeuronX model loading

---

## 5. Directory Structure for Deliverables

```
Claude_tutorials/
├── 00_Discovery_Plan.md (this file)
├── Phase2_Annotations/
│   ├── Training_DLCs/
│   │   ├── 01_PyTorch_2.8_GPU_Training_Dockerfile_ANNOTATED.md
│   │   ├── 02_HuggingFace_PyTorch_2.8_GPU_Training_Dockerfile_ANNOTATED.md
│   │   ├── 03_HuggingFace_PyTorch_2.5_GPU_Training_Dockerfile_ANNOTATED.md
│   │   ├── 04_HuggingFace_NeuronX_2.8_Training_Dockerfile_ANNOTATED.md
│   │   ├── 05_HuggingFace_NeuronX_2.7_Training_Dockerfile_ANNOTATED.md
│   │   ├── 06_PyTorch_2.7_ARM64_GPU_Training_Dockerfile_ANNOTATED.md
│   │   ├── 07_PyTorch_2.8_NeuronX_Training_Dockerfile_ANNOTATED.md
│   │   └── 08_PyTorch_2.7_NeuronX_Training_Dockerfile_ANNOTATED.md
│   └── Inference_DLCs/
│       ├── 01_vLLM_0.11_GPU_Entrypoint_ANNOTATED.md
│       ├── 02_SGLang_GPU_Entrypoint_ANNOTATED.md
│       ├── 03_DJL_LMI_vLLM_Config_ANNOTATED.md
│       ├── 04_DJL_TensorRT_LLM_Config_ANNOTATED.md
│       ├── 05_DJL_NeuronX_Config_ANNOTATED.md
│       ├── 06_HuggingFace_TGI_Config_ANNOTATED.md
│       ├── 07_HuggingFace_NeuronX_2.8_Entrypoint_ANNOTATED.md
│       └── 08_vLLM_ARM64_GPU_Entrypoint_ANNOTATED.md
├── Phase3_Guides/
│   ├── Container_Internal_Configuration_Guide.md
│   ├── Framework_Optimization_Mapping.md
│   ├── deploy_sagemaker_sdk_template.py
│   └── deploy_aws_cdk_template.py
└── README.md
```

---

## 6. Next Steps

1. ✅ **Phase 1 Complete**: Discovery and curation plan documented
2. **Phase 2 (In Progress)**:
   - Annotate 8 Training DLC Dockerfiles with optimization explanations
   - Annotate 8 Inference DLC entry point scripts with model loading logic
3. **Phase 3 (Pending)**:
   - Create Container Internal Configuration Guide
   - Develop SageMaker Python SDK deployment template
   - Develop AWS CDK deployment template
   - Create Framework Optimization Mapping table

---

## 7. Key Insights

### Common Optimization Patterns Identified:
1. **Training Containers**:
   - EFA (Elastic Fabric Adapter) for low-latency inter-node communication
   - NCCL 2.x for optimized GPU-to-GPU communication
   - Flash Attention 2.x for memory-efficient attention computation
   - Transformer Engine for mixed-precision training
   - NeuronX Distributed for AWS Trainium/Inferentia2

2. **Inference Containers**:
   - **PagedAttention** (vLLM) - Efficient KV cache management
   - **RadixAttention** (SGLang) - Prefix caching for structured generation
   - **Continuous Batching** - Dynamic batching for variable-length inputs
   - **TensorRT-LLM** - Kernel fusion and quantization
   - **TransformersNeuronX** - Model compilation for Neuron accelerators

3. **Environment Variables** (discovered in entry scripts):
   - `SM_VLLM_*` - vLLM configuration parameters
   - `SM_SGLANG_*` - SGLang configuration parameters
   - Model paths typically default to `/opt/ml/model` in SageMaker

---

**Document Status**: Phase 1 Complete ✅
**Last Updated**: 2025-11-23
**Total DLCs Identified**: 17+ unique Foundation Model containers
