# HuggingFace Text Generation Inference (TGI) - External Reference

**Source**: HuggingFace maintains TGI as a standalone project (not part of AWS DLC repository)

**Repository**: https://github.com/huggingface/text-generation-inference

**Note**: TGI is a production-grade Rust-based inference server optimized for decoder-only LLMs with built-in optimizations and token streaming.

---

## Overview: What is TGI?

**Text Generation Inference (TGI)** is HuggingFace's production LLM serving solution:
- **Language**: Rust (high performance, memory safe)
- **Python Backend**: For model loading (transformers library)
- **Optimizations**: Flash Attention, Paged Attention, quantization
- **APIs**: REST + gRPC with OpenAI compatibility
- **Target**: Production deployments requiring reliability and performance

### Key Differentiators

```
vLLM (Python):
  ├─ Maximum throughput via PagedAttention
  ├─ Python ecosystem integration
  └─ Easiest deployment

TGI (Rust):
  ├─ Production-grade reliability (Rust memory safety)
  ├─ Token streaming (real-time output)
  ├─ HuggingFace ecosystem integration
  ├─ Built-in safeguards (max concurrent requests, timeouts)
  └─ Advanced monitoring

SGLang (Python):
  ├─ RadixAttention for prefix caching
  └─ Structured generation
```

**When to choose TGI**: Production deployments needing HuggingFace integration, token streaming, and enterprise reliability.

---

## TGI Architecture

### Internal Stack

```
┌─────────────────────────────────────────────────────┐
│             TGI Router (Rust)                       │
│  HTTP Server | gRPC Server | Load Balancer         │
│  Request Queue | Rate Limiting | Metrics            │
└─────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼────────────────┐   ┌──────────────▼──────┐
│  Batch Manager (Rust)  │   │  Health Checks      │
│  Dynamic Batching      │   │  /health, /metrics  │
│  Token Streaming       │   └─────────────────────┘
└────────────────────────┘
        │
┌───────▼────────────────────────────────────────────┐
│       Python Backend (transformers)                │
│  Model Loading | Tokenization | Generation         │
│  Flash Attention | Paged Attention | Quantization  │
└────────────────────────────────────────────────────┘
        │
┌───────▼────────┐
│  NVIDIA GPUs   │
│  (A100, H100)  │
└────────────────┘
```

### Optimizations Built-In

1. **Flash Attention 2**: Memory-efficient attention (2-3x memory reduction)
2. **Paged Attention**: KV cache management (reduces fragmentation)
3. **Continuous Batching**: Dynamic request batching during generation
4. **Quantization**: bitsandbytes (INT8), GPTQ, AWQ support
5. **Safetensors**: Fast model loading (2-10x faster than pickle)
6. **Token Streaming**: Real-time output via Server-Sent Events (SSE)

---

## Available Images

TGI is distributed through HuggingFace's container registry and AWS Marketplace:

### Official Images

| Version | Image URI | Key Features |
|---------|-----------|--------------|
| **Latest** | `ghcr.io/huggingface/text-generation-inference:latest` | Latest optimizations |
| **2.4.0** | `ghcr.io/huggingface/text-generation-inference:2.4.0` | Stable release |
| **2.3.1** | `ghcr.io/huggingface/text-generation-inference:2.3.1` | Production-tested |

### AWS SageMaker Integration

```python
# TGI on SageMaker via HuggingFace DLC
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.0-tgi2.4.0-gpu-py311-cu122-ubuntu22.04",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3-70b-hf",
        "HF_TOKEN": "hf_...",  # For gated models
        "MAX_INPUT_LENGTH": "2048",
        "MAX_TOTAL_TOKENS": "4096",
        "MAX_BATCH_PREFILL_TOKENS": "4096",
        "QUANTIZE": "bitsandbytes-nf4",  # Optional quantization
    }
)
```

**Note**: Exact image URIs may vary by release. Check HuggingFace documentation for latest versions.

---

## Configuration via Environment Variables

TGI is configured entirely through environment variables (no config files):

### Core Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MODEL_ID` or `HF_MODEL_ID` | HuggingFace model ID | Required | `meta-llama/Llama-3-70b-hf` |
| `REVISION` | Model revision/branch | `main` | `main`, `fp16`, specific commit |
| `HF_TOKEN` | HuggingFace API token | None | `hf_...` (for gated models) |
| `NUM_SHARD` | Tensor parallelism degree | 1 | 8 (for 8 GPUs) |
| `DTYPE` | Data type | `float16` | `bfloat16`, `float16` |

### Memory & Performance

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MAX_INPUT_LENGTH` | Max input tokens | 1024 | 2048, 4096 |
| `MAX_TOTAL_TOKENS` | Max input + output | 2048 | 4096, 8192 |
| `MAX_BATCH_PREFILL_TOKENS` | Prefill batch size | 4096 | 8192, 16384 |
| `MAX_BATCH_TOTAL_TOKENS` | Total batch tokens | 16384 | 32768 |
| `WAITING_SERVED_RATIO` | Batch scheduling ratio | 1.2 | 1.5 |

### Advanced Optimizations

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `QUANTIZE` | Quantization method | None | `bitsandbytes`, `gptq`, `awq` |
| `CUDA_MEMORY_FRACTION` | GPU memory fraction | 0.95 | 0.90 |
| `ROPE_SCALING` | RoPE position scaling | None | `linear`, `dynamic` |
| `ROPE_FACTOR` | RoPE scaling factor | None | 2.0, 4.0 |

### Production Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MAX_CONCURRENT_REQUESTS` | Max simultaneous requests | 128 | 256 |
| `MAX_WAITING_TOKENS` | Max queued tokens | 20 | 50 |
| `PORT` | HTTP server port | 80 | 8080 (for SageMaker) |
| `TRUST_REMOTE_CODE` | Allow custom model code | false | true |

---

## Deployment Examples

### 1. SageMaker Deployment (Simple)

```python
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:latest",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    env={
        # Model Configuration
        "HF_MODEL_ID": "meta-llama/Llama-3-70b-hf",
        "HF_TOKEN": "hf_...",

        # GPU Configuration
        "NUM_SHARD": "8",  # 8x A100 on ml.p4d.24xlarge

        # Memory Settings
        "MAX_INPUT_LENGTH": "2048",
        "MAX_TOTAL_TOKENS": "4096",
        "MAX_BATCH_PREFILL_TOKENS": "8192",

        # Performance
        "DTYPE": "bfloat16",
        "CUDA_MEMORY_FRACTION": "0.95",

        # Production
        "MAX_CONCURRENT_REQUESTS": "256",
    }
)

predictor = model.deploy(
    instance_type="ml.p4d.24xlarge",
    initial_instance_count=1,
    endpoint_name="llama-3-70b-tgi",
)
```

### 2. With Quantization (Cost Optimization)

```python
# 4-bit quantization - fits 70B on 4x A10G (ml.g5.12xlarge)
model = HuggingFaceModel(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:latest",
    role=role,
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3-70b-hf",
        "HF_TOKEN": "hf_...",
        "NUM_SHARD": "4",  # 4x A10G
        "QUANTIZE": "bitsandbytes-nf4",  # 4-bit NormalFloat quantization
        "MAX_INPUT_LENGTH": "2048",
        "MAX_TOTAL_TOKENS": "4096",
    }
)

predictor = model.deploy(
    instance_type="ml.g5.12xlarge",  # Much cheaper than ml.p4d.24xlarge
    initial_instance_count=1,
)

# Cost savings: $7.09/hr (g5.12xlarge) vs $32.77/hr (p4d.24xlarge) = 78% savings
```

### 3. With RoPE Scaling (Extended Context)

```python
# Extend Llama 3's 8K context to 32K via RoPE scaling
model = HuggingFaceModel(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:latest",
    role=role,
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3-70b-hf",
        "NUM_SHARD": "8",

        # RoPE Scaling for extended context
        "ROPE_SCALING": "dynamic",
        "ROPE_FACTOR": "4.0",  # 8K × 4 = 32K context

        # Adjust token limits
        "MAX_INPUT_LENGTH": "16384",  # 16K input
        "MAX_TOTAL_TOKENS": "32768",  # 32K total

        # May need more memory
        "MAX_BATCH_PREFILL_TOKENS": "16384",
        "MAX_BATCH_TOTAL_TOKENS": "65536",
    }
)

predictor = model.deploy(instance_type="ml.p4d.24xlarge", initial_instance_count=1)
```

---

## Inference API

TGI provides multiple API formats:

### 1. OpenAI-Compatible Chat Completions

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

response = client.invoke_endpoint(
    EndpointName='llama-3-70b-tgi',
    ContentType='application/json',
    Body=json.dumps({
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain quantum computing."}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False,  # Set True for streaming
    })
)

result = json.loads(response['Body'].read())
print(result['choices'][0]['message']['content'])
```

### 2. Token Streaming (Server-Sent Events)

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

# Streaming request
response = client.invoke_endpoint_with_response_stream(
    EndpointName='llama-3-70b-tgi',
    ContentType='application/json',
    Body=json.dumps({
        "inputs": "Write a story about AI:",
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.9,
            "stream": True,
        }
    })
)

# Stream tokens as they're generated
for event in response['Body']:
    chunk = json.loads(event['PayloadPart']['Bytes'].decode())
    if 'token' in chunk:
        print(chunk['token']['text'], end='', flush=True)
```

### 3. HuggingFace Native Format

```python
response = client.invoke_endpoint(
    EndpointName='llama-3-70b-tgi',
    ContentType='application/json',
    Body=json.dumps({
        "inputs": "Explain AI in simple terms:",
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "return_full_text": False,
        }
    })
)

result = json.loads(response['Body'].read())
print(result[0]['generated_text'])
```

---

## Advanced Features

### 1. Safetensors Fast Loading

TGI uses **safetensors** format for 2-10x faster model loading:

```
Traditional PyTorch (pickle):
  Llama 3 70B loading time: 10-15 minutes

TGI with safetensors:
  Llama 3 70B loading time: 2-3 minutes

Speedup: 5x faster startup
```

**How it works**:
- Safetensors stores tensors in memory-mapped format
- Direct loading without deserialization
- No arbitrary code execution (security benefit)

### 2. Continuous Batching

TGI implements **iteration-level batching** (like vLLM):

```
Traditional Static Batching:
  Request 1: [████████████████████] 2000ms (100 tokens)
  Request 2: [████████████████████] 2000ms (100 tokens)
  Batch: Must wait for slowest (2000ms)

TGI Continuous Batching:
  Request 1: [████████████████████] 2000ms (100 tokens)
  Request 2: [██████]              600ms  (30 tokens, finished early)
  New Request 3 joins batch immediately after Request 2 completes

Throughput: 2-3x improvement
```

### 3. Automatic Tensor Parallelism

TGI automatically distributes model across GPUs based on `NUM_SHARD`:

```python
# NUM_SHARD=8 on ml.p4d.24xlarge
# TGI automatically:
#   - Splits model layers across 8 GPUs
#   - Initializes NCCL communication
#   - Manages all-reduce operations
#   - No manual configuration needed

env = {"NUM_SHARD": "8"}  # That's it!
```

### 4. Grammar-Based Generation (Experimental)

TGI supports structured outputs via grammar constraints:

```python
# JSON schema enforcement
response = client.invoke_endpoint(
    Body=json.dumps({
        "inputs": "Generate a user profile",
        "parameters": {
            "grammar": {
                "type": "json",
                "value": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
    })
)
# Output guaranteed valid JSON: {"name": "Alice", "age": 30}
```

---

## Performance Comparison

### Llama 3 70B Inference (8x A100, 2K input + 100 output)

| Framework | Throughput (req/sec) | Latency P50 (ms) | Streaming Support | Ecosystem |
|-----------|---------------------|------------------|-------------------|-----------|
| **TGI** | 60 | 350 | ✅ SSE | HuggingFace |
| **vLLM** | 50 | 400 | ⚠️ Limited | OpenAI-like |
| **SGLang** | 45 | 380 | ✅ SSE | SGLang |
| **PyTorch** | 5 | 2000 | ❌ | PyTorch |

**TGI Advantages**:
- Slightly higher throughput than vLLM
- Native token streaming (best UX)
- HuggingFace integration (datasets, tokenizers)
- Rust reliability

### Cost Analysis: Quantized 70B on Smaller Instance

```
TGI with bitsandbytes-nf4 quantization:
  Instance: ml.g5.12xlarge (4x A10G 24GB)
  Model: Llama 3 70B quantized to 4-bit (~35 GB)
  Throughput: ~25 req/sec
  Cost: $7.09/hour

vLLM without quantization:
  Instance: ml.p4d.24xlarge (8x A100 80GB)
  Model: Llama 3 70B in BF16 (~140 GB)
  Throughput: ~50 req/sec (2x faster)
  Cost: $32.77/hour (4.6x more expensive)

Cost per 1M tokens:
  TGI (quantized): ~$7.09/hr ÷ 90K tokens/hr = $78.78
  vLLM (full precision): ~$32.77/hr ÷ 180K tokens/hr = $182.06

Savings: 57% cheaper despite lower throughput (for cost-sensitive workloads)
```

---

## When to Use HuggingFace TGI

### ✅ Use TGI when:

1. **HuggingFace Ecosystem Integration**:
   - Using HF models, datasets, tokenizers
   - Need seamless HF Hub integration
   - Existing HF workflows

2. **Token Streaming Required**:
   - Chatbot interfaces (real-time output)
   - User-facing applications
   - Reduced perceived latency

3. **Production Reliability**:
   - Rust memory safety (no segfaults)
   - Built-in rate limiting
   - Graceful error handling
   - Battle-tested in production

4. **Advanced Monitoring**:
   - Prometheus metrics out-of-the-box
   - Request/response logging
   - Health checks (`/health`, `/metrics`)

5. **Quantization for Cost Savings**:
   - bitsandbytes, GPTQ, AWQ support
   - Run larger models on smaller instances
   - Trade-off: slight accuracy loss for 4x cost reduction

### ❌ Use vLLM instead when:

- Maximum throughput is priority (vLLM slightly faster)
- Python-native deployment preferred
- Don't need HuggingFace integration
- Simplicity over features

### ❌ Use SGLang instead when:

- Prefix caching critical (chatbots with system prompts)
- Structured generation primary use case
- 5-10x speedup for repeated prompts

---

## Supported Models

TGI supports **decoder-only** LLMs (not encoder-only or encoder-decoder):

### ✅ Fully Supported Architectures:

- **Llama** (all versions: 1, 2, 3)
- **Mistral/Mixtral** (including MoE models)
- **GPT-2, GPT-J, GPT-NeoX**
- **Falcon** (7B, 40B, 180B)
- **Qwen** (1.5, 2.0, 2.5)
- **MPT** (7B, 30B)
- **BLOOM** (all sizes)
- **CodeLlama, StarCoder**
- **Phi** (Microsoft)

### ❌ Not Supported:

- **Encoder-only**: BERT, RoBERTa (use `transformers` directly)
- **Encoder-decoder**: T5, BART, FLAN-T5 (use `transformers`)
- **Vision models**: CLIP (use custom serving)
- **Multimodal**: LLaVA (limited support, check docs)

### Check Compatibility:

```python
# HuggingFace Hub model card shows TGI support
# Look for "Inference Endpoints" or "TGI" badge
# Or test locally:

docker run --gpus all \
  -p 8080:80 \
  -v $PWD/cache:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3-70b-hf \
  --num-shard 1

# If it starts successfully, model is supported
```

---

## Monitoring & Observability

### Metrics Endpoint

TGI exposes Prometheus metrics at `/metrics`:

```bash
curl http://endpoint:8080/metrics

# Key metrics:
tgi_request_duration_seconds_bucket    # Latency histogram
tgi_request_success_total              # Successful requests
tgi_request_failure_total              # Failed requests
tgi_batch_inference_duration_seconds   # Batch processing time
tgi_queue_size                         # Requests in queue
tgi_request_inference_duration_seconds # Model inference time
```

### CloudWatch Integration (SageMaker)

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Get endpoint metrics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='ModelLatency',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': 'llama-3-70b-tgi'},
        {'Name': 'VariantName', 'Value': 'AllTraffic'},
    ],
    StartTime=datetime.now() - timedelta(hours=1),
    EndTime=datetime.now(),
    Period=300,
    Statistics=['Average', 'Maximum', 'Minimum'],
)

# Create alarm for high latency
cloudwatch.put_metric_alarm(
    AlarmName='tgi-high-latency',
    MetricName='ModelLatency',
    Namespace='AWS/SageMaker',
    Statistic='Average',
    Period=300,
    EvaluationPeriods=2,
    Threshold=5000,  # 5 seconds
    ComparisonOperator='GreaterThanThreshold',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': 'llama-3-70b-tgi'},
    ],
)
```

---

## Troubleshooting

### Issue: OOM during model loading

```
CUDA out of memory. Tried to allocate 20.00 GiB
```

**Solutions**:

1. Increase `NUM_SHARD`:
   ```python
   "NUM_SHARD": "8",  # Use all GPUs
   ```

2. Enable quantization:
   ```python
   "QUANTIZE": "bitsandbytes-nf4",  # 4-bit
   ```

3. Use larger instance:
   ```python
   instance_type="ml.p4d.24xlarge",  # 8x A100 80GB
   ```

### Issue: Slow first request

```
First request takes 30 seconds, subsequent requests are fast
```

**Explanation**: Model compilation (CUDA kernels) on first forward pass

**Solution**: Warmup requests during health check:
```python
# After deployment, send warmup requests
for _ in range(10):
    predictor.predict({"inputs": "warmup", "parameters": {"max_new_tokens": 1}})
# Now production requests will be fast
```

### Issue: Token streaming not working on SageMaker

```
Streaming returns full response at once
```

**Solution**: Use `invoke_endpoint_with_response_stream`:
```python
# Wrong (no streaming):
response = client.invoke_endpoint(...)

# Correct (streaming):
response = client.invoke_endpoint_with_response_stream(
    EndpointName='llama-3-70b-tgi',
    Body=json.dumps({"inputs": "...", "parameters": {"stream": True}})
)
```

---

## External Resources

- **Official Repository**: https://github.com/huggingface/text-generation-inference
- **Documentation**: https://huggingface.co/docs/text-generation-inference
- **Model Compatibility**: https://huggingface.co/docs/text-generation-inference/supported_models
- **AWS Blog**: https://aws.amazon.com/blogs/machine-learning/deploy-bloom-176b-and-opt-30b-on-amazon-sagemaker-with-large-model-inference-deep-learning-containers-and-deepspeed/
- **HuggingFace Hub**: https://huggingface.co/docs/inference-endpoints (uses TGI under the hood)

---

## Summary

**HuggingFace TGI = Production-Grade Rust LLM Serving**

- **Language**: Rust (memory safe, high performance)
- **Optimizations**: Flash Attention, Paged Attention, Continuous Batching
- **Unique**: Token streaming, safetensors loading, HF ecosystem
- **Best For**: Production deployments, chatbots, HF workflows
- **Trade-off**: Slightly more complex than vLLM, decoder-only models

**Key Advantage**: Best-in-class token streaming + HuggingFace integration + Rust reliability.
