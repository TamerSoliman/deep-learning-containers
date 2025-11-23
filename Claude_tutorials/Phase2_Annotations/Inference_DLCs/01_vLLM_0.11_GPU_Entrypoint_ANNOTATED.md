# Annotated Entry Point Script: vLLM 0.11.2 SageMaker Inference

**Source**: `vllm/build_artifacts/sagemaker_entrypoint.sh`

**Purpose**: This script is the entry point for vLLM inference containers deployed to SageMaker endpoints. It transforms SageMaker environment variables into vLLM CLI arguments and launches the vLLM OpenAI-compatible API server.

**Execution Context**: Runs when SageMaker creates an endpoint with this container

---

## Complete Script

```bash
#!/bin/bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

PREFIX="SM_VLLM_"
ARG_PREFIX="--"

ARGS=(--port 8080)

while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
```

---

## LINE-BY-LINE ANNOTATION

### Section 1: Shebang and Telemetry

```bash
#!/bin/bash
```

**WHAT**: Standard bash shebang
**WHY**: Makes the script executable directly (`./sagemaker_entrypoint.sh`)

```bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true
```

**WHAT**: Executes telemetry collection script if it exists

**WHY - Telemetry for AWS**:
- **Purpose**: Collects anonymous usage metrics
  - Container version
  - vLLM version
  - Instance type
  - Region
- **Why `>/dev/null 2>&1`**: Suppresses all output (stdout and stderr)
- **Why `|| true`**: Ensures script continues even if telemetry fails
  - If telemetry script doesn't exist: continue
  - If telemetry script errors: continue
  - Critical: Never let telemetry break inference!

**BEST PRACTICE**: Defensive programming - auxiliary functions should never break main functionality

---

### Section 2: Configuration Constants

```bash
PREFIX="SM_VLLM_"
ARG_PREFIX="--"

ARGS=(--port 8080)
```

**WHAT**: Sets up configuration for environment variable parsing

**WHY - Environment Variable Convention**:
- **PREFIX="SM_VLLM_"**: SageMaker vLLM environment variable namespace
  - `SM` = SageMaker (AWS convention)
  - `VLLM` = vLLM specific (vs other inference frameworks)
  - All vLLM configuration uses this prefix

**EXAMPLE Environment Variables**:
```bash
SM_VLLM_MODEL=/opt/ml/model          # Where model is stored
SM_VLLM_TENSOR_PARALLEL_SIZE=4        # Use 4 GPUs
SM_VLLM_MAX_MODEL_LEN=4096           # Max sequence length
SM_VLLM_GPU_MEMORY_UTILIZATION=0.9   # Use 90% of GPU memory
SM_VLLM_TRUST_REMOTE_CODE=true       # Allow custom model code
```

**ARGS=(--port 8080)**:
- **Port 8080**: SageMaker standard inference port
  - SageMaker routes HTTP traffic from endpoint to container port 8080
  - This is hardcoded - all SageMaker inference containers must listen on 8080

---

### Section 3: Environment Variable Transformation

```bash
while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")
```

**WHAT**: Converts environment variables to vLLM CLI arguments

**WHY - Bridging SageMaker and vLLM APIs**:
- **SageMaker uses environment variables**: Easy to set via SDK/console
- **vLLM uses CLI arguments**: `--model`, `--tensor-parallel-size`, etc.
- This script bridges the gap

**HOW IT WORKS - Step by Step**:

#### Step 1: Get all SM_VLLM_* environment variables
```bash
env | grep "^${PREFIX}"
```
**Output example**:
```
SM_VLLM_MODEL=/opt/ml/model
SM_VLLM_TENSOR_PARALLEL_SIZE=4
SM_VLLM_MAX_MODEL_LEN=4096
```

#### Step 2: Parse each variable
```bash
while IFS='=' read -r key value; do
```
- **IFS='='**: Split on equals sign
- **read -r key value**: Read key and value
- Example: `key="SM_VLLM_MODEL"`, `value="/opt/ml/model"`

#### Step 3: Transform key to CLI argument name
```bash
arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
```

**Breaking down the transformation**:
1. `"${key#"${PREFIX}"}"`: Remove `SM_VLLM_` prefix
   - `SM_VLLM_TENSOR_PARALLEL_SIZE` → `TENSOR_PARALLEL_SIZE`

2. `tr '[:upper:]' '[:lower:]'`: Convert to lowercase
   - `TENSOR_PARALLEL_SIZE` → `tensor_parallel_size`

3. `tr '_' '-'`: Replace underscores with hyphens
   - `tensor_parallel_size` → `tensor-parallel-size`

**Result**: `SM_VLLM_TENSOR_PARALLEL_SIZE` → `tensor-parallel-size`

#### Step 4: Build argument array
```bash
ARGS+=("${ARG_PREFIX}${arg_name}")
if [ -n "$value" ]; then
    ARGS+=("$value")
fi
```

- **`ARGS+=("--${arg_name}")`**: Add flag (e.g., `--tensor-parallel-size`)
- **`if [ -n "$value" ]`**: Check if value is not empty
- **`ARGS+=("$value")`**: Add value (e.g., `4`)

**Example transformation**:
```
SM_VLLM_TENSOR_PARALLEL_SIZE=4
  ↓
--tensor-parallel-size 4
```

**Handling boolean flags**:
```bash
SM_VLLM_DISABLE_LOG_STATS=  # Empty value
  ↓
--disable-log-stats  # Flag only, no value
```

---

### Section 4: Launch vLLM Server

```bash
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
```

**WHAT**: Replaces current shell process with vLLM server

**WHY - Each Component**:

#### `exec`:
- **What it does**: Replaces the current shell process (PID 1) with vLLM
- **Why it matters**:
  - No parent shell process consuming memory
  - vLLM receives all signals (SIGTERM, SIGINT) directly
  - Proper graceful shutdown when SageMaker stops endpoint
- **Without `exec`**: Shell would be PID 1, vLLM would be child process
  - Extra memory overhead
  - Signal handling issues

#### `python3 -m vllm.entrypoints.openai.api_server`:
- **-m flag**: Run module as script
- **vllm.entrypoints.openai.api_server**: OpenAI-compatible API server
  - Implements OpenAI Chat Completions API
  - Drop-in replacement for OpenAI API

#### `"${ARGS[@]}"`:
- **Bash array expansion**: Expands all arguments properly
- **Example**:
  ```bash
  ARGS=(--port 8080 --model /opt/ml/model --tensor-parallel-size 4)
  "${ARGS[@]}" expands to:
  --port 8080 --model /opt/ml/model --tensor-parallel-size 4
  ```

---

## Complete Workflow Example

### SageMaker Endpoint Creation:

```python
from sagemaker.model import Model

# Define environment variables
env = {
    "SM_VLLM_MODEL": "meta-llama/Llama-3-70b-hf",
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "8",  # Use all 8 GPUs
    "SM_VLLM_MAX_MODEL_LEN": "4096",
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.95",
    "SM_VLLM_TRUST_REMOTE_CODE": "true",
}

# Create model
model = Model(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.2-sagemaker",
    model_data="s3://bucket/llama3-70b/",
    role=role,
    env=env,
)

# Deploy endpoint
predictor = model.deploy(
    instance_type="ml.p4d.24xlarge",  # 8x A100 GPUs
    initial_instance_count=1,
)
```

### What Happens Inside the Container:

1. **SageMaker starts container**:
   - Mounts model from S3 to `/opt/ml/model/`
   - Sets environment variables

2. **Container executes entry point**:
   ```bash
   /usr/local/bin/sagemaker_entrypoint.sh
   ```

3. **Entry point script runs**:
   ```bash
   # Telemetry (silent)
   bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

   # Parse environment variables
   # Builds ARGS array:
   ARGS=(
     --port 8080
     --model meta-llama/Llama-3-70b-hf
     --tensor-parallel-size 8
     --max-model-len 4096
     --gpu-memory-utilization 0.95
     --trust-remote-code true
   )

   # Launch vLLM
   exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
   ```

4. **vLLM initializes**:
   - Downloads model from HuggingFace Hub (or loads from /opt/ml/model/)
   - Initializes 8-way tensor parallelism across GPUs
   - Allocates KV cache (95% of GPU memory)
   - Starts HTTP server on port 8080

5. **SageMaker health check**:
   - Sends GET request to `http://localhost:8080/ping`
   - vLLM responds with 200 OK
   - SageMaker marks endpoint as "InService"

6. **Endpoint ready for inference**:
   ```python
   response = predictor.predict({
       "model": "meta-llama/Llama-3-70b-hf",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 100,
   })
   ```

---

## Common Environment Variables and Their Effects

### Model Configuration

| Environment Variable | Transforms To | Purpose |
|---------------------|---------------|---------|
| `SM_VLLM_MODEL` | `--model` | HuggingFace model ID or local path |
| `SM_VLLM_TOKENIZER` | `--tokenizer` | Override tokenizer (rare) |
| `SM_VLLM_TRUST_REMOTE_CODE` | `--trust-remote-code` | Allow custom model code (needed for some models) |

### Performance Tuning

| Environment Variable | Transforms To | Purpose |
|---------------------|---------------|---------|
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size` | Number of GPUs to split model across |
| `SM_VLLM_PIPELINE_PARALLEL_SIZE` | `--pipeline-parallel-size` | Number of pipeline stages (rare) |
| `SM_VLLM_MAX_MODEL_LEN` | `--max-model-len` | Maximum sequence length |
| `SM_VLLM_GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | Fraction of GPU memory for KV cache (0.0-1.0) |
| `SM_VLLM_MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | Batch size limit |

### Advanced Features

| Environment Variable | Transforms To | Purpose |
|---------------------|---------------|---------|
| `SM_VLLM_QUANTIZATION` | `--quantization` | Quantization method (e.g., `awq`, `gptq`) |
| `SM_VLLM_DTYPE` | `--dtype` | Data type (`float16`, `bfloat16`, `float32`) |
| `SM_VLLM_ENABLE_PREFIX_CACHING` | `--enable-prefix-caching` | Cache prompt prefixes for faster inference |
| `SM_VLLM_DISABLE_LOG_STATS` | `--disable-log-stats` | Disable statistics logging |

---

## Why This Design is Elegant

### 1. Declarative Configuration
- Users set environment variables (declarative)
- No need to modify scripts or build custom images
- Easy to version control and replicate

### 2. SageMaker Native
- Follows SageMaker conventions (`SM_` prefix, port 8080)
- Integrates with SageMaker SDK seamlessly
- Works with SageMaker features (auto-scaling, monitoring)

### 3. Flexibility
- Any vLLM CLI argument can be set via environment variable
- Example: New vLLM feature `--enable-chunked-prefill`
  - Set `SM_VLLM_ENABLE_CHUNKED_PREFILL=true`
  - Script automatically transforms to `--enable-chunked-prefill true`
  - No code changes needed!

### 4. Fail-Safe
- Telemetry failures don't break inference
- Default port 8080 always set
- Environment variable parsing is robust

---

## Comparison: EC2 vs SageMaker Entry Points

**EC2 Entry Point** (`dockerd_entrypoint.sh`):
- More general purpose
- No automatic environment variable transformation
- User runs `docker run ... vllm serve --model ...` manually

**SageMaker Entry Point** (this script):
- Tailored for managed endpoints
- Automatic configuration from environment variables
- User just calls `model.deploy(env={...})`

---

## Real-World Inference Example

### Deploying Llama 3 70B with 8-way Tensor Parallelism:

```python
# 1. Define configuration
env = {
    "SM_VLLM_MODEL": "meta-llama/Llama-3-70b-hf",
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "8",
    "SM_VLLM_MAX_MODEL_LEN": "8192",  # Support long contexts
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.95",
    "SM_VLLM_ENABLE_PREFIX_CACHING": "true",  # Cache system prompts
}

# 2. Deploy
model = Model(image_uri=vllm_image, env=env, ...)
predictor = model.deploy(instance_type="ml.p4d.24xlarge")

# 3. Inference
response = predictor.predict({
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    "max_tokens": 500,
    "temperature": 0.7,
})

print(response["choices"][0]["message"]["content"])
```

**What happens under the hood**:
1. Entry point transforms env vars to:
   ```
   python3 -m vllm.entrypoints.openai.api_server \
     --port 8080 \
     --model meta-llama/Llama-3-70b-hf \
     --tensor-parallel-size 8 \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.95 \
     --enable-prefix-caching true
   ```

2. vLLM splits 70B model across 8 GPUs (~9GB per GPU)

3. Allocates 72GB KV cache per GPU (8 × 72 = 576GB total)

4. Processes request:
   - System prompt cached (prefix caching)
   - User message processed
   - Generates 500 tokens
   - Returns via OpenAI-compatible API

**Performance**:
- First request: ~2 seconds (includes system prompt processing)
- Subsequent requests with same system prompt: ~0.5 seconds (prefix cached)
- Throughput: ~500 tokens/second (continuous batching)

---

## Summary: Entry Point Lifecycle

```
SageMaker starts container
  ↓
sagemaker_entrypoint.sh (this script)
  ↓
Telemetry (silent, non-blocking)
  ↓
Parse SM_VLLM_* environment variables
  ↓
Transform to vLLM CLI arguments
  ↓
exec vllm.entrypoints.openai.api_server
  ↓
vLLM initializes model (multi-GPU)
  ↓
Listen on port 8080
  ↓
SageMaker health check (GET /ping)
  ↓
Endpoint InService
  ↓
Serve inference requests (POST /v1/chat/completions)
```

**Key Takeaway**: This 20-line script enables declarative, production-ready LLM deployment with zero boilerplate. It's the glue between SageMaker's managed infrastructure and vLLM's high-performance serving.
