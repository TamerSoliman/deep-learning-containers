# DJL Large Model Inference (LMI) Containers - External Reference

**Source**: AWS maintains DJL inference containers as pre-built images (no Dockerfiles in this repository)

**Repository**: https://github.com/deepjavalibrary/djl-serving

**Note**: These are enterprise-grade inference containers with Java integration, designed for production LLM deployments with multiple backend options.

---

## Overview: DJL Serving Architecture

### What is DJL?

**Deep Java Library (DJL)** is an enterprise ML serving framework providing:
- Java-native model serving (integrates with Spring Boot, microservices)
- Multiple backend engines (vLLM, TensorRT-LLM, LMI-Dist, NeuronX)
- Automatic model parallelism configuration
- Multi-model serving (one endpoint, multiple models)
- Production monitoring and observability

### LMI (Large Model Inference) Containers

Starting with DJLServing 0.28.0, AWS provides **LMI** containers optimized for Foundation Models with three backend options:

1. **LMI with vLLM Backend** - High throughput with PagedAttention
2. **LMI with TensorRT-LLM Backend** - Maximum performance with NVIDIA optimization
3. **LMI with NeuronX Backend** - Cost-effective inference on AWS Inferentia2

---

## Container #3: DJL LMI 17.0.0 with vLLM 0.11.1 Backend

### Available Images

| DJL Version | vLLM Version | Image URI | Python |
|-------------|--------------|-----------|--------|
| 0.35.0 | vLLM 0.11.1 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-lmi17.0.0-cu128` | 3.12 |
| 0.34.0 | vLLM 0.10.2 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16.0.0-cu128` | 3.12 |
| 0.33.0 | vLLM 0.8.4 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128` | 3.12 |

### What's Inside

**Core Components**:
- **DJLServing 0.35.0**: Java model server with REST/gRPC APIs
- **vLLM 0.11.1**: Python inference engine (same as standalone vLLM DLC)
- **Transformers 4.57.1**: HuggingFace model loading
- **Accelerate 1.0.1**: Model parallelism utilities

**Key Optimizations**:
- PagedAttention for KV cache management
- Continuous batching for throughput
- Automatic tensor parallelism
- Rolling batch scheduler

### Configuration via serving.properties

Instead of environment variables, DJL uses a `serving.properties` file:

```properties
# Model Configuration
option.model_id=meta-llama/Llama-3-70b-hf
option.task=text-generation

# vLLM Backend Settings
engine=Python
option.entryPoint=djl_python.huggingface
option.rolling_batch=vllm

# GPU Configuration
option.tensor_parallel_degree=8
option.max_model_len=4096
option.gpu_memory_utilization=0.95

# Performance Tuning
option.dtype=bfloat16
option.trust_remote_code=true
option.max_rolling_batch_size=64
option.enable_prefix_caching=true
```

### Deployment Example (SageMaker SDK)

```python
from sagemaker import Model

# Option 1: Environment variable configuration (simple)
model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-lmi17.0.0-cu128",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3-70b-hf",
        "TENSOR_PARALLEL_DEGREE": "8",
        "MAX_MODEL_LEN": "4096",
        "OPTION_ROLLING_BATCH": "vllm",
        "HF_TOKEN": "hf_...",  # If gated model
    }
)

# Option 2: serving.properties configuration (advanced)
# 1. Create serving.properties locally
# 2. Package as model.tar.gz
# 3. Upload to S3
# 4. Set model_data when creating Model

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-lmi17.0.0-cu128",
    model_data="s3://my-bucket/llama-3-70b-config.tar.gz",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)

predictor = model.deploy(
    instance_type="ml.p4d.24xlarge",
    initial_instance_count=1,
)
```

### Inference API (OpenAI-compatible)

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

response = client.invoke_endpoint(
    EndpointName='llama-3-70b-djl',
    ContentType='application/json',
    Body=json.dumps({
        "inputs": "Explain quantum computing",
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    })
)

result = json.loads(response['Body'].read())
print(result['generated_text'])
```

### When to Use DJL LMI vLLM

**✅ Use DJL LMI vLLM when**:
- Enterprise Java ecosystem integration required
- Multi-model serving (one endpoint, multiple models)
- Advanced observability/monitoring needed
- Team familiar with DJL/Java stack
- Need production-grade model management

**❌ Use standalone vLLM DLC instead when**:
- Python-native deployment preferred
- Simpler configuration desired
- No Java integration requirements
- Single model per endpoint

### Unique Features vs Standalone vLLM

1. **Multi-Model Serving**:
   ```properties
   # Serve multiple models on same endpoint
   models=llama-70b,mistral-7b

   llama-70b.option.model_id=meta-llama/Llama-3-70b-hf
   llama-70b.option.tensor_parallel_degree=8

   mistral-7b.option.model_id=mistralai/Mistral-7B-v0.1
   mistral-7b.option.tensor_parallel_degree=1
   ```

2. **Java Integration**:
   ```java
   // Call from Spring Boot microservice
   @Autowired
   private SageMakerRuntime sagemaker;

   public String generateText(String prompt) {
       InvokeEndpointRequest request = new InvokeEndpointRequest()
           .withEndpointName("llama-3-70b-djl")
           .withContentType("application/json")
           .withBody(ByteBuffer.wrap(toJson(prompt)));

       InvokeEndpointResult result = sagemaker.invokeEndpoint(request);
       return parseResult(result.getBody());
   }
   ```

3. **Advanced Monitoring**:
   - Built-in metrics (latency, throughput, queue depth)
   - Request/response logging
   - Model warmup tracking
   - Custom health checks

---

## Container #4: DJL LMI with TensorRT-LLM 0.21.0 Backend

### Available Images

| DJL Version | TRT-LLM Version | Image URI | Python |
|-------------|-----------------|-----------|--------|
| 0.33.0 | TensorRT-LLM 0.21.0 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-tensorrtllm0.21.0-cu128` | 3.12 |
| 0.32.0 | TensorRT-LLM 0.12.0 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-tensorrtllm0.12.0-cu125` | 3.10 |
| 0.30.0 | TensorRT-LLM 0.12.0 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-tensorrtllm0.12.0-cu125` | 3.10 |

### What Makes TensorRT-LLM Different

**TensorRT-LLM** is NVIDIA's optimized inference runtime providing:
- **Kernel Fusion**: Combines multiple operations into single GPU kernel
- **Quantization**: FP8, INT8, INT4 for 2-8x speedup
- **In-Flight Batching**: Dynamic batching during generation
- **Graph Optimization**: Layer-wise compilation for minimum latency

**Performance**: 2-8x faster than PyTorch, 1.5-3x faster than vLLM

**Trade-off**: Less flexible, requires model compilation, supports fewer architectures

### Supported Model Architectures

✅ **Fully Supported**:
- Llama 2/3 (all sizes)
- GPT-J, GPT-NeoX
- Falcon (7B, 40B, 180B)
- MPT (7B, 30B)
- Mistral/Mixtral
- Qwen 1.5/2.0

❌ **Not Supported**:
- Custom architectures
- Models requiring remote code
- Vision-language models (limited support)

### Configuration via serving.properties

```properties
# Model Configuration
option.model_id=meta-llama/Llama-3-70b-hf
option.task=text-generation

# TensorRT-LLM Backend
engine=Python
option.entryPoint=djl_python.tensorrt_llm
option.rolling_batch=trtllm

# Compilation Settings
option.tensor_parallel_degree=8
option.max_input_len=2048
option.max_output_len=2048
option.max_batch_size=64

# Quantization (optional, for speedup)
option.dtype=float16
option.quantization=fp8  # or int8, int4

# TensorRT Optimization
option.enable_chunked_context=true
option.enable_kv_cache_reuse=true
```

### Model Compilation Process

**IMPORTANT**: TensorRT-LLM requires **offline compilation** before deployment:

```bash
# 1. Download HuggingFace model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70b-hf")
model.save_pretrained("./llama-3-70b")

# 2. Convert to TensorRT-LLM format
python convert_checkpoint.py \
  --model_dir ./llama-3-70b \
  --output_dir ./trt_ckpt \
  --dtype float16 \
  --tp_size 8

# 3. Build TensorRT engine
trtllm-build \
  --checkpoint_dir ./trt_ckpt \
  --output_dir ./trt_engine \
  --gemm_plugin float16 \
  --max_batch_size 64 \
  --max_input_len 2048 \
  --max_output_len 2048 \
  --max_beam_width 1

# 4. Package and upload to S3
tar -czf llama-3-70b-trt.tar.gz -C trt_engine .
aws s3 cp llama-3-70b-trt.tar.gz s3://my-bucket/
```

### Deployment with Pre-Compiled Model

```python
from sagemaker import Model

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-tensorrtllm0.21.0-cu128",
    model_data="s3://my-bucket/llama-3-70b-trt.tar.gz",  # Pre-compiled TRT engine
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    env={
        "TENSOR_PARALLEL_DEGREE": "8",
        "OPTION_ROLLING_BATCH": "trtllm",
    }
)

predictor = model.deploy(
    instance_type="ml.p4d.24xlarge",
    initial_instance_count=1,
)
```

### Performance Comparison: TensorRT-LLM vs vLLM

**Llama 3 70B Inference (8x A100, 2K input + 100 output tokens)**:

| Backend | Throughput (req/sec) | Latency P50 (ms) | Latency P99 (ms) |
|---------|---------------------|------------------|------------------|
| **TensorRT-LLM (FP16)** | 120 | 200 | 350 |
| **TensorRT-LLM (FP8)** | 180 | 150 | 280 |
| **vLLM (BF16)** | 50 | 400 | 650 |
| **PyTorch (naive)** | 5 | 2000 | 3500 |

**Speedup**: TensorRT-LLM FP8 is **3.6x faster** than vLLM

### When to Use DJL TensorRT-LLM

**✅ Use TensorRT-LLM when**:
- Absolute minimum latency required (<200ms P50)
- Supported model architecture (Llama, GPT-J, Falcon, Mistral)
- Can pre-compile models offline
- Budget allows GPU costs
- High request volume (amortize compilation time)

**❌ Use vLLM instead when**:
- Model architecture not supported by TensorRT-LLM
- Rapid experimentation (no compilation step)
- Custom model code required
- Flexibility > maximum performance

---

## Container #5: DJL LMI with NeuronX SDK 2.20.1 Backend

### Available Images

| DJL Version | NeuronSDK Version | Image URI | Python |
|-------------|-------------------|-----------|--------|
| 0.30.0 | SDK 2.20.1 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1` | 3.10 |
| 0.29.0 | SDK 2.19.1 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-neuronx-sdk2.19.1` | 3.10 |
| 0.28.0 | SDK 2.18.2 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-neuronx-sdk2.18.2` | 3.10 |

### What is AWS Neuron (Inferentia2)?

**AWS Neuron** is custom silicon for cost-effective ML inference:
- **Inferentia2**: AWS-designed ML accelerator (inf2 instances)
- **40-70% cheaper** than equivalent GPU instances
- **NeuronSDK**: Compiler + runtime for PyTorch/TensorFlow
- **TransformersNeuronX**: HuggingFace integration

**Architecture**:
```
inf2.48xlarge:
  ├─ 12x NeuronCores v2 (each NeuronCore = mini-GPU)
  ├─ 384 GB system memory
  └─ 32 GB on-chip memory (HBM)

Cost: ~$12.98/hour (vs $32.77/hour for ml.p4d.24xlarge)
Savings: 60% cost reduction
```

### Configuration via serving.properties

```properties
# Model Configuration
option.model_id=meta-llama/Llama-3-70b-hf
option.task=text-generation

# NeuronX Backend
engine=Python
option.entryPoint=djl_python.transformers-neuronx
option.rolling_batch=transformers-neuronx

# NeuronCore Configuration
option.tensor_parallel_degree=12  # Use all 12 NeuronCores
option.n_positions=4096
option.dtype=fp16
option.model_loading_timeout=1800  # 30 minutes

# Neuron Compilation
option.compiled_graph_path=/opt/ml/model/compiled
option.trust_remote_code=true
```

### Model Compilation for Neuron

**CRITICAL**: Neuron requires **ahead-of-time compilation** (like TensorRT-LLM):

```python
# Option 1: Compile during first deployment (slow startup)
# SageMaker compiles on first request, caches for subsequent requests
model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3-70b-hf",
        "TENSOR_PARALLEL_DEGREE": "12",
        "N_POSITIONS": "4096",
        "OPTION_ROLLING_BATCH": "transformers-neuronx",
    }
)
# First startup: 30-60 minutes (compilation)
# Subsequent restarts: 5-10 minutes (load cached compilation)

# Option 2: Pre-compile offline (recommended for production)
# Use transformers-neuronx library to compile, upload to S3
```

### Offline Compilation Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx import constants

# 1. Load HuggingFace model
model_id = "meta-llama/Llama-3-70b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Convert to NeuronX format
neuron_model = LlamaForSampling.from_pretrained(
    model_id,
    batch_size=4,
    tp_degree=12,  # Tensor parallelism across 12 NeuronCores
    n_positions=4096,
    amp='fp16',
    neuron_config=constants.GenerationConfig(
        max_length=4096,
        top_k=50,
    )
)

# 3. Compile (this takes 30-60 minutes)
neuron_model.to_neuron()

# 4. Save compiled artifacts
neuron_model.save_pretrained("./llama-3-70b-neuron")

# 5. Package and upload
import tarfile
with tarfile.open("llama-3-70b-neuron.tar.gz", "w:gz") as tar:
    tar.add("./llama-3-70b-neuron", arcname=".")

# Upload to S3
import boto3
s3 = boto3.client('s3')
s3.upload_file("llama-3-70b-neuron.tar.gz", "my-bucket", "models/llama-3-70b-neuron.tar.gz")
```

### Deployment with Pre-Compiled Neuron Model

```python
model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1",
    model_data="s3://my-bucket/models/llama-3-70b-neuron.tar.gz",  # Pre-compiled
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    env={
        "TENSOR_PARALLEL_DEGREE": "12",
        "OPTION_ROLLING_BATCH": "transformers-neuronx",
    }
)

predictor = model.deploy(
    instance_type="ml.inf2.48xlarge",  # Inferentia2 instance
    initial_instance_count=1,
)
# Startup: ~5 minutes (no compilation, just loading)
```

### Performance & Cost Analysis

**Llama 3 70B Inference Comparison**:

| Instance Type | Hardware | Requests/sec | Latency P50 | Cost/hour | Cost/1M tokens |
|---------------|----------|--------------|-------------|-----------|----------------|
| **ml.p4d.24xlarge** | 8x A100 (vLLM) | 50 | 400ms | $32.77 | $1.64 |
| **ml.p4d.24xlarge** | 8x A100 (TRT-LLM FP8) | 120 | 200ms | $32.77 | $0.68 |
| **ml.inf2.48xlarge** | 12x Inferentia2 (Neuron) | 30 | 600ms | **$12.98** | **$0.41** |

**When Neuron Wins**:
- **Cost**: 40-60% cheaper than GPU for same model
- **Efficiency**: Optimized specifically for inference (not training)
- **Scale**: High request volume amortizes compilation cost

**When GPU Wins**:
- **Latency**: GPU ~2x faster for latency-critical applications
- **Flexibility**: No compilation step, faster iteration

### When to Use DJL NeuronX

**✅ Use DJL NeuronX when**:
- Cost is primary concern (40-70% savings vs GPU)
- High request volume (>100K requests/day)
- Can pre-compile models offline
- Inference-only workload (no training)
- Standard architectures (Llama, GPT, BERT)

**❌ Use GPU instead when**:
- Minimum latency critical (<200ms)
- Rapid experimentation (compilation too slow)
- Custom model architectures (Neuron compiler limited)
- Low request volume (compilation overhead not justified)

---

## Architecture Comparison: All Three DJL Backends

### Internal Stack Comparison

```
┌─────────────────────────────────────────────────────┐
│           DJL Model Server (Java)                   │
│  REST API | gRPC API | Metrics | Health Checks      │
└─────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│  vLLM Backend  │ │TensorRT-LLM │ │ NeuronX Backend │
│                │ │   Backend   │ │                 │
│ PagedAttention │ │ Kernel      │ │ TransformersNX  │
│ Continuous     │ │ Fusion      │ │ Neuron Compiler │
│ Batching       │ │ FP8/INT8    │ │ NeuronCores     │
└────────────────┘ └─────────────┘ └─────────────────┘
        │                 │                 │
┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│  NVIDIA GPUs   │ │ NVIDIA GPUs │ │  Inferentia2    │
│  (A100, H100)  │ │ (A100, H100)│ │  (NeuronCores)  │
└────────────────┘ └─────────────┘ └─────────────────┘
```

### Decision Matrix

| Requirement | Best Backend | Reason |
|-------------|--------------|--------|
| **Maximum throughput** | vLLM | PagedAttention + continuous batching |
| **Minimum latency** | TensorRT-LLM | Kernel fusion + FP8 quantization |
| **Lowest cost** | NeuronX | 40-70% cheaper hardware |
| **Maximum flexibility** | vLLM | Supports most architectures |
| **Fastest deployment** | vLLM | No compilation required |
| **Production stability** | Any (all enterprise-grade) | DJL provides consistent interface |

### Unified DJL Advantages

**All three backends benefit from DJL's**:
1. **Multi-model serving**: One endpoint, multiple models
2. **Java integration**: Spring Boot, microservices
3. **Advanced monitoring**: Built-in metrics, logging
4. **Auto-scaling**: Request queue management
5. **Graceful degradation**: Fallback strategies

---

## Complete Deployment Example: Multi-Backend Strategy

### Scenario: Production LLM API with Cost Optimization

```python
from sagemaker import Model

# 1. Deploy vLLM for low-latency tier (expensive, fast)
vllm_model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-lmi17.0.0-cu128",
    role=role,
    env={"HF_MODEL_ID": "meta-llama/Llama-3-70b-hf", "TENSOR_PARALLEL_DEGREE": "8"},
)
vllm_endpoint = vllm_model.deploy(
    endpoint_name="llama-70b-vllm-fast",
    instance_type="ml.p4d.24xlarge",
    initial_instance_count=1,
)

# 2. Deploy NeuronX for high-volume tier (cheap, adequate latency)
neuron_model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1",
    model_data="s3://bucket/llama-70b-neuron.tar.gz",  # Pre-compiled
    role=role,
    env={"TENSOR_PARALLEL_DEGREE": "12"},
)
neuron_endpoint = neuron_model.deploy(
    endpoint_name="llama-70b-neuron-cheap",
    instance_type="ml.inf2.48xlarge",
    initial_instance_count=2,  # 2 instances for scale
)

# 3. Application-level routing (in your API layer)
def route_request(request_priority):
    if request_priority == "high":
        return invoke_endpoint("llama-70b-vllm-fast")  # Fast, expensive
    else:
        return invoke_endpoint("llama-70b-neuron-cheap")  # Cheap, adequate
```

**Cost Analysis**:
- High-priority: 10% of traffic → vLLM ($32.77/hr) → $3.28/hr
- Standard: 90% of traffic → 2x Neuron ($25.96/hr) → $23.36/hr
- **Total**: $26.64/hr (vs $32.77/hr all-vLLM = **19% savings**)

---

## Summary: DJL LMI Container Family

### Three Containers, One Interface

**DJL LMI vLLM** (Container #3):
- Backend: vLLM 0.11.1 with PagedAttention
- Best for: High throughput, flexibility
- Instance: ml.p4d.24xlarge (8x A100)

**DJL TensorRT-LLM** (Container #4):
- Backend: TensorRT-LLM 0.21.0 with kernel fusion
- Best for: Minimum latency, maximum performance
- Instance: ml.p4d.24xlarge (8x A100)
- Requires: Pre-compilation

**DJL NeuronX** (Container #5):
- Backend: NeuronSDK 2.20.1 with TransformersNeuronX
- Best for: Cost optimization, high volume
- Instance: ml.inf2.48xlarge (12x Inferentia2)
- Requires: Pre-compilation

### Common DJL Features (All Containers)

- ✅ Multi-model serving
- ✅ Java integration (Spring Boot)
- ✅ Advanced monitoring/observability
- ✅ Production-grade reliability
- ✅ Auto-scaling support

### External Resources

- **DJL Serving GitHub**: https://github.com/deepjavalibrary/djl-serving
- **LMI Documentation**: https://docs.djl.ai/docs/serving/serving/docs/lmi/index.html
- **Model Compatibility**: https://docs.djl.ai/docs/serving/serving/docs/lmi/user_guides/model_compatibility.html
- **Configuration Guide**: https://docs.djl.ai/docs/serving/serving/docs/lmi/user_guides/lmi_input_output_schema.html

---

**Note**: These containers are maintained separately from the main deep-learning-containers repository. For source code, Dockerfiles, and detailed configuration examples, refer to the official DJL Serving repository.
