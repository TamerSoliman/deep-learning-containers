# Annotated Entry Point Script: SGLang GPU Inference Container

**Source**: `sglang/build_artifacts/sagemaker_entrypoint.sh`

**Purpose**: SageMaker entry point for SGLang inference - transforms environment variables to CLI arguments and launches SGLang server with RadixAttention optimization.

---

## Key Innovation: RadixAttention for Prefix Caching

### What Makes SGLang Different from vLLM:

```
vLLM (PagedAttention):
  ├─ Optimizes KV cache memory fragmentation
  ├─ 2-4x throughput improvement
  └─ No automatic prompt reuse

SGLang (RadixAttention):
  ├─ Everything vLLM does + Automatic Prefix Caching
  ├─ Detects repeated prompt prefixes (e.g., system prompts)
  ├─ Caches using radix tree with LRU eviction
  └─ 5-10x faster for chatbots with repeated system prompts
```

**Use Case**: Multi-turn conversations, chatbots, structured generation

---

## Complete Entry Point Script

```bash
#!/bin/bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true
```

**WHAT**: AWS telemetry collection for DLC usage metrics
**WHY**: Helps AWS understand container usage patterns, failures (opt-out available)
**HOW**: Runs asynchronously, errors suppressed to prevent deployment failures

---

```bash
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    bash /usr/local/bin/start_cuda_compat.sh
fi
```

**WHAT**: CUDA Forward Compatibility Layer startup
**WHY**: Allows newer CUDA runtime (in container) to work with older drivers (on host)
**HOW**: Checks if NVIDIA tools exist, then starts compat layer if needed

**Example**:
- Host has NVIDIA driver 525.x (CUDA 12.0 compatible)
- Container has CUDA 12.6
- Compat layer bridges the gap

---

```bash
echo "Starting server"

PREFIX="SM_SGLANG_"
ARG_PREFIX="--"

ARGS=()
```

**WHAT**: Initialize argument transformation system
**WHY**: SageMaker uses environment variables, SGLang CLI uses arguments
**HOW**:
- `PREFIX="SM_SGLANG_"` - All SGLang configs start with this
- `ARG_PREFIX="--"` - CLI arguments use double-dash format
- `ARGS=()` - Empty array to accumulate arguments

---

## Environment Variable Transformation Logic

```bash
while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")
```

**WHAT**: Transform SM_SGLANG_* environment variables to --sglang-* CLI arguments
**WHY**: Enables declarative configuration via SageMaker environment variables
**HOW**:

### Transformation Steps:

1. **`env | grep "^${PREFIX}"`**: Find all SM_SGLANG_* variables
2. **`IFS='=' read -r key value`**: Split at `=` into key/value
3. **`${key#"${PREFIX}"}`**: Remove SM_SGLANG_ prefix
   - `SM_SGLANG_TENSOR_PARALLEL_SIZE` → `TENSOR_PARALLEL_SIZE`
4. **`tr '[:upper:]' '[:lower:]'`**: Convert to lowercase
   - `TENSOR_PARALLEL_SIZE` → `tensor_parallel_size`
5. **`tr '_' '-'`**: Replace underscores with dashes
   - `tensor_parallel_size` → `tensor-parallel-size`
6. **`ARGS+=("${ARG_PREFIX}${arg_name}")`**: Add --tensor-parallel-size
7. **`if [ -n "$value" ]; then ARGS+=("$value")`**: Add value if non-empty

### Example Transformations:

| Environment Variable | Resulting CLI Argument |
|---------------------|------------------------|
| `SM_SGLANG_TENSOR_PARALLEL_SIZE=8` | `--tensor-parallel-size 8` |
| `SM_SGLANG_MAX_TOTAL_TOKENS=8192` | `--max-total-tokens 8192` |
| `SM_SGLANG_ENABLE_FLASHINFER=true` | `--enable-flashinfer true` |
| `SM_SGLANG_DISABLE_RADIX_CACHE=false` | `--disable-radix-cache false` |
| `SM_SGLANG_DTYPE=bfloat16` | `--dtype bfloat16` |

---

## Default Value Injection

```bash
# Add default port only if not already set
if ! [[ " ${ARGS[@]} " =~ " --port " ]]; then
    ARGS+=(--port "${SM_SGLANG_PORT:-8080}")
fi
```

**WHAT**: Set default port to 8080 if not specified
**WHY**: SageMaker expects containers to listen on port 8080
**HOW**:
- Check if `--port` already in ARGS array
- If not present, use `SM_SGLANG_PORT` env var, or fallback to 8080
- Syntax `${VAR:-default}` means "use VAR if set, else use default"

---

```bash
# Add default host only if not already set
if ! [[ " ${ARGS[@]} " =~ " --host " ]]; then
    ARGS+=(--host "${SM_SGLANG_HOST:-0.0.0.0}")
fi
```

**WHAT**: Bind to all network interfaces (0.0.0.0) by default
**WHY**: SageMaker's load balancer needs to reach the container from any IP
**HOW**:
- 127.0.0.1 = localhost only (would fail in SageMaker)
- 0.0.0.0 = all interfaces (allows external connections)

---

```bash
# Add default model-path only if not already set
if ! [[ " ${ARGS[@]} " =~ " --model-path " ]]; then
    ARGS+=(--model-path "${SM_SGLANG_MODEL_PATH:-/opt/ml/model}")
fi
```

**WHAT**: Default model location to `/opt/ml/model` (SageMaker standard)
**WHY**: SageMaker downloads model artifacts to `/opt/ml/model` automatically
**HOW**:
- If model_data specified in SageMaker Model → downloaded to /opt/ml/model
- If no model_data → container must download (use HuggingFace model ID instead)

**Example Model Loading Paths**:
```python
# Option 1: Pre-downloaded model from S3
# SageMaker Model: model_data="s3://bucket/llama-3-70b.tar.gz"
# Container loads from: /opt/ml/model

# Option 2: Download from HuggingFace Hub at runtime
# SageMaker Env: SM_SGLANG_MODEL_PATH="meta-llama/Llama-3-70b-hf"
# SGLang downloads from Hub during startup
```

---

## Server Launch

```bash
echo "Running command: exec python3 -m sglang.launch_server ${ARGS[@]}"
exec python3 -m sglang.launch_server "${ARGS[@]}"
```

**WHAT**: Launch SGLang server with assembled arguments
**WHY**: `exec` replaces bash process with Python, ensuring PID 1 is Python (proper signal handling)
**HOW**:
- `python3 -m sglang.launch_server` - Runs SGLang's main entry point
- `${ARGS[@]}` - Expands to all accumulated CLI arguments
- Echo command first for debugging (visible in CloudWatch Logs)

### What Happens Next (Inside sglang.launch_server):

1. **Model Loading** (2-10 minutes for large models):
   ```python
   # SGLang internal process
   model = AutoModelForCausalLM.from_pretrained(
       args.model_path,
       torch_dtype=torch.bfloat16,
       device_map="auto",  # Distributes across GPUs
   )
   ```

2. **RadixAttention Initialization**:
   ```python
   # Creates radix tree for prefix caching
   radix_cache = RadixCache(
       max_cached_tokens=args.max_total_tokens // 2,
       eviction_policy="lru",
   )
   ```

3. **Multi-GPU Setup** (if tensor_parallel_size > 1):
   ```python
   # Initializes NCCL process group
   dist.init_process_group(backend="nccl")
   # Splits model layers across GPUs
   ```

4. **HTTP Server Startup**:
   ```python
   # FastAPI server on port 8080
   uvicorn.run(app, host="0.0.0.0", port=8080)
   ```

5. **Health Check Endpoint**:
   ```
   GET /health → Returns 200 OK when ready
   ```

6. **SageMaker Health Check**:
   - SageMaker pings `/ping` or `/health` every 30 seconds
   - Once healthy, endpoint status changes to `InService`
   - Takes 10-30 minutes total for large models

---

## Complete Lifecycle Example

### 1. SageMaker Model Creation

```python
from sagemaker import Model

model = Model(
    name="llama-3-70b-sglang",
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:latest",
    model_data="s3://my-bucket/llama-3-70b.tar.gz",  # Optional
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    env={
        "SM_SGLANG_TENSOR_PARALLEL_SIZE": "8",
        "SM_SGLANG_MAX_TOTAL_TOKENS": "8192",
        "SM_SGLANG_ENABLE_FLASHINFER": "true",
        "SM_SGLANG_DTYPE": "bfloat16",
        "SM_SGLANG_TRUST_REMOTE_CODE": "true",
        # Optional: Use HuggingFace Hub instead of S3
        # "SM_SGLANG_MODEL_PATH": "meta-llama/Llama-3-70b-hf",
        # "HF_TOKEN": "hf_...",
    },
)
```

### 2. Container Startup (This Entrypoint Script)

```bash
# Environment variables set by SageMaker:
SM_SGLANG_TENSOR_PARALLEL_SIZE=8
SM_SGLANG_MAX_TOTAL_TOKENS=8192
SM_SGLANG_ENABLE_FLASHINFER=true
SM_SGLANG_DTYPE=bfloat16

# Transformed to CLI arguments:
python3 -m sglang.launch_server \
  --tensor-parallel-size 8 \
  --max-total-tokens 8192 \
  --enable-flashinfer true \
  --dtype bfloat16 \
  --port 8080 \
  --host 0.0.0.0 \
  --model-path /opt/ml/model
```

### 3. Model Loading (15-25 minutes)

```
[CloudWatch Logs]
Loading model from /opt/ml/model
Model size: 140 GB (70B parameters × 2 bytes/param)
Distributing across 8 GPUs (tensor parallelism)
GPU 0: Layers 0-9
GPU 1: Layers 10-19
...
GPU 7: Layers 70-79
Initializing RadixAttention cache (4096 tokens)
Server listening on 0.0.0.0:8080
```

### 4. SageMaker Health Checks

```
SageMaker → GET http://container:8080/health
Container → 200 OK (model loaded, ready to serve)
SageMaker → Endpoint status: InService
```

### 5. Inference Request

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

# Request with system prompt (gets cached by RadixAttention)
response = client.invoke_endpoint(
    EndpointName='llama-3-70b-sglang',
    ContentType='application/json',
    Body=json.dumps({
        "text": "You are a helpful AI assistant.\n\nUser: Explain quantum computing.",
        "sampling_params": {
            "max_new_tokens": 256,
            "temperature": 0.7,
        }
    })
)

# First request: 2000ms (no cache hit)
# Second request with same system prompt: 400ms (5x faster!)
```

---

## SGLang-Specific Optimizations

### 1. RadixAttention (Prefix Caching)

**Traditional Approach (vLLM)**:
```
Request 1: "You are helpful.\n\nUser: What is AI?"
  → Process entire prompt: 1500ms

Request 2: "You are helpful.\n\nUser: Explain ML?"
  → Process entire prompt again: 1500ms (no reuse)
```

**SGLang RadixAttention**:
```
Request 1: "You are helpful.\n\nUser: What is AI?"
  → Process prompt: 1500ms
  → Cache prefix "You are helpful.\n\n" (radix tree)

Request 2: "You are helpful.\n\nUser: Explain ML?"
  → Cache hit on prefix: 300ms (reuse KV cache)
  → Only process "User: Explain ML?": 300ms
  → Total: 300ms (5x faster!)
```

**Radix Tree Structure**:
```
Root
 └─ "You are helpful.\n\n" [cached KV states]
     ├─ "User: What is AI?" [child KV states]
     └─ "User: Explain ML?" [child KV states]
```

### 2. Structured Generation

SGLang natively supports:
- **Regex-constrained generation**: Force output to match pattern
- **JSON schema validation**: Guarantee valid JSON
- **Grammar-based generation**: Context-free grammar constraints

```python
# Example: Force JSON output
response = client.invoke_endpoint(
    Body=json.dumps({
        "text": "Generate a user profile",
        "sampling_params": {
            "regex": r'\{"name": "[^"]+", "age": \d+\}',
        }
    })
)
# Output guaranteed to match: {"name": "Alice", "age": 30}
```

### 3. FlashInfer Integration

**WHAT**: Even faster attention kernel than FlashAttention
**WHY**: Optimized specifically for inference (not training)
**HOW**: Set `SM_SGLANG_ENABLE_FLASHINFER=true`

**Performance**:
- FlashAttention 2: 2-3x faster than naive attention
- FlashInfer: 1.2-1.5x faster than FlashAttention 2
- Best for decoding phase (generating tokens)

---

## Configuration Reference

### Common Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SM_SGLANG_MODEL_PATH` | Model path or HF ID | `/opt/ml/model` | `meta-llama/Llama-3-70b-hf` |
| `SM_SGLANG_TENSOR_PARALLEL_SIZE` | GPUs to use | 1 | 8 (for ml.p4d.24xlarge) |
| `SM_SGLANG_MAX_TOTAL_TOKENS` | Max sequence length | 4096 | 8192, 16384 |
| `SM_SGLANG_ENABLE_FLASHINFER` | Use FlashInfer kernel | false | true |
| `SM_SGLANG_DISABLE_RADIX_CACHE` | Disable prefix caching | false | true (if not needed) |
| `SM_SGLANG_MEM_FRACTION_STATIC` | GPU memory for model weights | 0.85 | 0.90 |
| `SM_SGLANG_DTYPE` | Data type | auto | bfloat16, float16 |
| `SM_SGLANG_TRUST_REMOTE_CODE` | Allow custom model code | false | true (Qwen, MPT) |
| `SM_SGLANG_PORT` | HTTP server port | 8080 | 8080 (fixed for SageMaker) |
| `SM_SGLANG_HOST` | Bind address | 0.0.0.0 | 0.0.0.0 (fixed for SageMaker) |
| `HF_TOKEN` | HuggingFace API token | None | hf_... (for gated models) |

### Memory Configuration

```python
# For Llama 3 70B on 8x A100 80GB:
env = {
    "SM_SGLANG_TENSOR_PARALLEL_SIZE": "8",
    "SM_SGLANG_MAX_TOTAL_TOKENS": "8192",
    "SM_SGLANG_MEM_FRACTION_STATIC": "0.88",
}

# Calculation:
# Model size: 140 GB (70B × 2 bytes/param in BF16)
# Per GPU: 140 GB / 8 = 17.5 GB
# KV cache: (80 GB - 17.5 GB) × 0.88 = ~55 GB/GPU
# Total KV cache: 55 GB × 8 = 440 GB
# Supports ~8192 tokens with batch size ~50
```

---

## Comparison: SGLang vs vLLM vs TGI

| Feature | SGLang | vLLM | TGI |
|---------|--------|------|-----|
| **Prefix Caching** | ✅ RadixAttention (automatic) | ❌ (manual only) | ✅ (limited) |
| **Structured Generation** | ✅ Native (regex, JSON, grammar) | ⚠️ Via guidance | ⚠️ Via constraints |
| **Attention Kernel** | FlashInfer (fastest) | FlashAttention 2 | FlashAttention 2 |
| **Throughput** | High | Highest | High |
| **Latency (cached prompts)** | Lowest (5-10x) | Medium | Medium |
| **Multi-modal** | ✅ (vision, audio) | ✅ (vision) | ✅ (vision) |
| **API Compatibility** | OpenAI + SGLang | OpenAI | HuggingFace + OpenAI |
| **Language** | Python | Python + Cython | Rust |

**When to Use SGLang**:
- ✅ Chatbots with repeated system prompts
- ✅ Multi-turn conversations
- ✅ Structured output requirements (JSON, regex)
- ✅ RAG with repeated document context
- ✅ Agent workflows with repeated instructions

**When to Use vLLM Instead**:
- ✅ Maximum throughput for single-turn requests
- ✅ No prompt reuse patterns
- ✅ Simplicity preferred

---

## Real-World Performance Example

### Chatbot with System Prompt (1000 requests)

**Scenario**: Customer support bot with 500-token system prompt

```
System Prompt (500 tokens):
"You are a customer support agent for Acme Corp.
Always be polite, reference our return policy..."

User Messages (varying):
- "How do I return an item?"
- "What's your shipping policy?"
- "Track my order #12345"
```

**vLLM Performance**:
```
Request 1: Process 500 tokens (system) + 10 tokens (user) = 1200ms
Request 2: Process 500 tokens (system) + 12 tokens (user) = 1200ms
...
Request 1000: Process 500 tokens (system) + 8 tokens (user) = 1200ms

Total: 1000 × 1200ms = 1,200,000ms = 20 minutes
```

**SGLang Performance (with RadixAttention)**:
```
Request 1: Process 500 tokens (system) + 10 tokens (user) = 1200ms
            Cache system prompt in radix tree

Request 2: Retrieve cached 500 tokens + process 12 tokens = 250ms
Request 3: Retrieve cached 500 tokens + process 8 tokens = 250ms
...
Request 1000: Retrieve cached 500 tokens + process 9 tokens = 250ms

Total: 1200ms + (999 × 250ms) = 250,950ms = 4.2 minutes

Speedup: 20 min / 4.2 min = 4.8x faster
```

**Cost Savings**:
- ml.p4d.24xlarge: $32.77/hour
- vLLM: 20 minutes = $10.92
- SGLang: 4.2 minutes = $2.28
- **Savings**: $8.64 per 1000 requests (79% reduction)

---

## Troubleshooting

### Issue: Model loading timeout

```
CloudWatch Logs:
Downloading model from HuggingFace Hub...
[After 30 minutes] Health check timeout
```

**Solution**: Pre-download model to S3, set `model_data` in SageMaker Model:
```python
model = Model(
    model_data="s3://bucket/llama-3-70b.tar.gz",  # Pre-downloaded
    env={
        "SM_SGLANG_MODEL_PATH": "/opt/ml/model",  # Use local path
    }
)
```

### Issue: Out of memory (OOM)

```
CUDA out of memory. Tried to allocate 20.00 GiB
```

**Solutions**:
1. Reduce `SM_SGLANG_MAX_TOTAL_TOKENS`:
   ```python
   "SM_SGLANG_MAX_TOTAL_TOKENS": "4096",  # From 8192
   ```

2. Increase `SM_SGLANG_TENSOR_PARALLEL_SIZE`:
   ```python
   "SM_SGLANG_TENSOR_PARALLEL_SIZE": "8",  # Use all GPUs
   ```

3. Use larger instance (more GPU memory):
   ```python
   instance_type="ml.p4d.24xlarge",  # 8x A100 80GB
   ```

### Issue: Radix cache not working

```
Logs show no cache hits despite repeated prompts
```

**Check**:
```python
env = {
    "SM_SGLANG_DISABLE_RADIX_CACHE": "false",  # Ensure enabled
}
```

**Verify cache hits in logs**:
```
[RadixCache] Hit rate: 85% (850/1000 requests)
```

---

## Summary

**SGLang Entry Point = Smart Configuration Layer**

- Transforms SageMaker environment variables to SGLang CLI arguments
- Sets sensible defaults (port 8080, host 0.0.0.0, model /opt/ml/model)
- Enables RadixAttention for 5-10x faster chatbot/RAG inference
- Supports structured generation (JSON, regex)
- Optimized for repeated prompt patterns

**Key Advantage**: Automatic prefix caching without code changes - just reuse system prompts and watch latency drop!
