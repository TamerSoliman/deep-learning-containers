# Container Internal Configuration Guide
## Deep Learning Containers for Foundation Models

**Purpose**: This guide documents the internal configuration, environment variables, file paths, and lifecycle hooks that control model loading and execution inside AWS Deep Learning Containers (DLCs).

**Audience**: ML Engineers and DevOps practitioners deploying Foundation Models to production

---

## Table of Contents
1. [Container Lifecycle Overview](#container-lifecycle-overview)
2. [Standard File Paths](#standard-file-paths)
3. [Environment Variables Reference](#environment-variables-reference)
4. [Model Loading Mechanisms](#model-loading-mechanisms)
5. [Entry Point Script Lifecycle](#entry-point-script-lifecycle)
6. [Configuration Patterns by Framework](#configuration-patterns-by-framework)

---

## Container Lifecycle Overview

### Training Container Lifecycle (SageMaker)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CONTAINER START                                              │
│    - SageMaker launches container on EC2 instance              │
│    - Mounts:                                                    │
│      • Training data:    /opt/ml/input/data/                   │
│      • Configuration:    /opt/ml/input/config/                 │
│      • Model output:     /opt/ml/model/                        │
│      • Checkpoints:      /opt/ml/checkpoints/                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. ENTRY POINT EXECUTION                                        │
│    - Runs: /usr/local/bin/start_with_right_hostname.sh         │
│    - Actions:                                                   │
│      • Fix hostname for multi-node communication               │
│      • Validate EFA connectivity (if multi-node)               │
│      • Run telemetry collection                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. SAGEMAKER TRAINING MODULE                                    │
│    - Executes: sagemaker_pytorch_container.training:main       │
│    - Actions:                                                   │
│      • Read hyperparameters from:                              │
│        /opt/ml/input/config/hyperparameters.json               │
│      • Read resource config from:                              │
│        /opt/ml/input/config/resourceconfig.json                │
│      • Set distributed training env vars (NCCL, MPI)           │
│      • Discover training instances via SageMaker API           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. USER TRAINING SCRIPT                                         │
│    - Runs user's script (e.g., train.py)                       │
│    - Script has access to:                                     │
│      • Data channels: /opt/ml/input/data/<channel>/            │
│      • Hyperparameters via SM_HP_* env vars                    │
│      • Distributed setup (already configured)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. MODEL PERSISTENCE                                            │
│    - Script saves model to: /opt/ml/model/                     │
│    - SageMaker uploads /opt/ml/model/ to S3                    │
│    - Checkpoints in /opt/ml/checkpoints/ also uploaded         │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Container Lifecycle (SageMaker Endpoint)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CONTAINER START                                              │
│    - SageMaker launches container                              │
│    - Mounts:                                                    │
│      • Model artifacts:  /opt/ml/model/                        │
│    - Sets environment variables (user-defined)                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. ENTRY POINT EXECUTION                                        │
│    - vLLM:    /usr/local/bin/sagemaker_entrypoint.sh           │
│    - SGLang:  /usr/local/bin/sagemaker_entrypoint.sh           │
│    - PyTorch: /usr/local/bin/torchserve-entrypoint.py          │
│    - Neuron:  /usr/local/bin/neuron-entrypoint.py              │
│                                                                 │
│    Actions:                                                     │
│      • Parse SM_<FRAMEWORK>_* environment variables            │
│      • Transform to CLI arguments                              │
│      • Validate model path exists                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. MODEL SERVER INITIALIZATION                                  │
│    - vLLM:      python -m vllm.entrypoints.openai.api_server   │
│    - SGLang:    python -m sglang.launch_server                 │
│    - TorchServe: torchserve --start                            │
│                                                                 │
│    Actions:                                                     │
│      • Load model from /opt/ml/model/ (or download from Hub)   │
│      • Initialize inference backend (KV cache, etc.)           │
│      • Start HTTP server on port 8080                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. HEALTH CHECK                                                 │
│    - SageMaker sends: GET http://localhost:8080/ping           │
│    - Server responds: 200 OK                                   │
│    - Endpoint status: InService                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. INFERENCE SERVING                                            │
│    - Client sends: POST /invocations (SageMaker standard)      │
│               or: POST /v1/chat/completions (OpenAI API)       │
│    - Server processes request and returns response             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Standard File Paths

### SageMaker Training Job Paths

| Path | Purpose | Read/Write | Lifecycle |
|------|---------|------------|-----------|
| `/opt/ml/input/data/<channel>/` | Training data mounted from S3 | Read-only | Persistent during job |
| `/opt/ml/input/config/hyperparameters.json` | Hyperparameters as JSON | Read-only | Persistent during job |
| `/opt/ml/input/config/resourceconfig.json` | Cluster configuration (hosts, GPUs) | Read-only | Persistent during job |
| `/opt/ml/model/` | Final model output | Write | Uploaded to S3 after job |
| `/opt/ml/checkpoints/` | Training checkpoints | Read/Write | Uploaded to S3 periodically |
| `/opt/ml/output/` | Logs and metrics | Write | Uploaded to S3 after job |
| `/opt/ml/code/` | User's training script | Read-only | Downloaded from S3 |

**Example - Accessing Data in Training Script**:
```python
import os

# Training data (e.g., from S3 bucket via "training" channel)
train_path = "/opt/ml/input/data/training/"
files = os.listdir(train_path)  # ['train.json', 'train.jsonl', ...]

# Save trained model
model.save_pretrained("/opt/ml/model/")
# SageMaker automatically uploads /opt/ml/model/ to S3 after training

# Save checkpoints during training
checkpoint_dir = "/opt/ml/checkpoints/step-1000/"
model.save_pretrained(checkpoint_dir)
# SageMaker uploads checkpoints to S3 every few minutes
```

### SageMaker Inference Endpoint Paths

| Path | Purpose | Read/Write | Notes |
|------|---------|------------|-------|
| `/opt/ml/model/` | Model artifacts from S3 | Read-only | Pre-populated before container start |
| `/tmp/` | Temporary storage | Read/Write | Instance storage, not persisted |
| `/dev/shm/` | Shared memory | Read/Write | Used for inter-process communication |

**Example - Model Loading in Inference**:
```python
# vLLM automatically looks for model in:
# 1. /opt/ml/model/ (if local files present)
# 2. Otherwise downloads from HuggingFace Hub

# If you set SM_VLLM_MODEL=/opt/ml/model:
# vLLM loads: /opt/ml/model/config.json, /opt/ml/model/model.safetensors, etc.

# If you set SM_VLLM_MODEL=meta-llama/Llama-3-70b-hf:
# vLLM downloads from HuggingFace Hub to /tmp/vllm_cache/
```

### CUDA and Driver Paths

| Path | Purpose | Notes |
|------|---------|-------|
| `/usr/local/cuda/` | CUDA Toolkit installation | CUDA_HOME points here |
| `/usr/local/cuda/lib64/` | CUDA libraries (cuDNN, cuBLAS, NCCL) | In LD_LIBRARY_PATH |
| `/opt/amazon/efa/` | EFA (Elastic Fabric Adapter) drivers | For multi-node communication |
| `/opt/amazon/openmpi/` | OpenMPI installation | For distributed training |
| `/opt/amazon/ofi-nccl/` | OFI-NCCL plugin | Allows NCCL to use EFA |

---

## Environment Variables Reference

### Training Container Environment Variables

#### SageMaker-Set Variables (Automatic)

| Variable | Example Value | Purpose |
|----------|---------------|---------|
| `SM_CHANNEL_TRAINING` | `/opt/ml/input/data/training` | Path to "training" data channel |
| `SM_CHANNEL_VALIDATION` | `/opt/ml/input/data/validation` | Path to "validation" data channel |
| `SM_MODEL_DIR` | `/opt/ml/model` | Where to save final model |
| `SM_OUTPUT_DATA_DIR` | `/opt/ml/output/data` | Where to write output files |
| `SM_NUM_GPUS` | `8` | Number of GPUs on instance |
| `SM_NUM_CPUS` | `96` | Number of CPUs on instance |
| `SM_HOSTS` | `["algo-1", "algo-2"]` | List of hosts in cluster (multi-node) |
| `SM_CURRENT_HOST` | `algo-1` | Current host name |
| `SM_NUM_NODES` | `2` | Total number of nodes |

**Example - Using in Training Script**:
```python
import os
import json

# Get data paths
train_dir = os.environ["SM_CHANNEL_TRAINING"]
val_dir = os.environ.get("SM_CHANNEL_VALIDATION", None)

# Get cluster info
num_gpus = int(os.environ["SM_NUM_GPUS"])
hosts = json.loads(os.environ["SM_HOSTS"])
current_host = os.environ["SM_CURRENT_HOST"]
is_master = current_host == hosts[0]

print(f"Training on {len(hosts)} nodes, {num_gpus} GPUs per node")

# Save model
model_dir = os.environ["SM_MODEL_DIR"]
model.save_pretrained(model_dir)
```

#### User-Defined Hyperparameters

Hyperparameters are accessible via `SM_HP_*` prefix:

```python
# In SageMaker SDK:
estimator = PyTorch(
    hyperparameters={
        "learning_rate": 1e-5,
        "batch_size": 32,
        "epochs": 3,
    }
)

# In training script:
import os

lr = float(os.environ["SM_HP_LEARNING_RATE"])  # "1e-5" → 0.00001
batch_size = int(os.environ["SM_HP_BATCH_SIZE"])  # "32" → 32
epochs = int(os.environ["SM_HP_EPOCHS"])  # "3" → 3
```

#### Distributed Training Variables

| Variable | Example Value | Purpose |
|----------|---------------|---------|
| `MASTER_ADDR` | `algo-1` | Master node address (auto-set by SM) |
| `MASTER_PORT` | `7777` | Master node port (auto-set by SM) |
| `RANK` | `0` | Global rank (0 to world_size-1) |
| `LOCAL_RANK` | `0` | Rank within node (0 to num_gpus-1) |
| `WORLD_SIZE` | `16` | Total number of processes (nodes × GPUs) |
| `NCCL_SOCKET_IFNAME` | `eth0` | Network interface for NCCL |
| `FI_PROVIDER` | `efa` | Use EFA for inter-node communication |

**Example - Distributed Training Setup**:
```python
import torch.distributed as dist

# SageMaker has already set these environment variables
dist.init_process_group(backend="nccl")

rank = dist.get_rank()  # Read from RANK env var
world_size = dist.get_world_size()  # Read from WORLD_SIZE env var

print(f"Process {rank}/{world_size}")
```

### Inference Container Environment Variables

#### vLLM Configuration (SM_VLLM_* namespace)

| Variable | Transforms To | Example Value | Purpose |
|----------|---------------|---------------|---------|
| `SM_VLLM_MODEL` | `--model` | `/opt/ml/model` or `meta-llama/Llama-3-70b-hf` | Model to load |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size` | `8` | Number of GPUs for tensor parallelism |
| `SM_VLLM_PIPELINE_PARALLEL_SIZE` | `--pipeline-parallel-size` | `2` | Number of pipeline stages |
| `SM_VLLM_MAX_MODEL_LEN` | `--max-model-len` | `4096` | Maximum sequence length |
| `SM_VLLM_GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | `0.95` | Fraction of GPU memory to use |
| `SM_VLLM_TRUST_REMOTE_CODE` | `--trust-remote-code` | `true` | Allow custom model code |
| `SM_VLLM_QUANTIZATION` | `--quantization` | `awq` | Quantization method (awq, gptq, etc.) |
| `SM_VLLM_DTYPE` | `--dtype` | `bfloat16` | Model data type |
| `SM_VLLM_ENABLE_PREFIX_CACHING` | `--enable-prefix-caching` | `true` | Cache prompt prefixes |
| `SM_VLLM_MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | `8192` | Max tokens in batch |
| `SM_VLLM_MAX_NUM_SEQS` | `--max-num-seqs` | `256` | Max concurrent sequences |

**Example - vLLM Deployment**:
```python
from sagemaker.model import Model

env = {
    "SM_VLLM_MODEL": "meta-llama/Llama-3-70b-hf",
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "8",
    "SM_VLLM_MAX_MODEL_LEN": "8192",
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.95",
    "SM_VLLM_ENABLE_PREFIX_CACHING": "true",
    "SM_VLLM_DTYPE": "bfloat16",
}

model = Model(image_uri=vllm_image, model_data=None, env=env, role=role)
predictor = model.deploy(instance_type="ml.p4d.24xlarge")
```

#### SGLang Configuration (SM_SGLANG_* namespace)

| Variable | Transforms To | Default Value | Purpose |
|----------|---------------|---------------|---------|
| `SM_SGLANG_MODEL_PATH` | `--model-path` | `/opt/ml/model` | Path to model |
| `SM_SGLANG_HOST` | `--host` | `0.0.0.0` | Bind address |
| `SM_SGLANG_PORT` | `--port` | `8080` | Server port |
| `SM_SGLANG_TP_SIZE` | `--tp-size` | `1` | Tensor parallelism size |
| `SM_SGLANG_MEM_FRACTION_STATIC` | `--mem-fraction-static` | `0.9` | GPU memory allocation |

**Example - SGLang Deployment**:
```python
env = {
    "SM_SGLANG_MODEL_PATH": "/opt/ml/model",
    "SM_SGLANG_TP_SIZE": "8",
    "SM_SGLANG_MEM_FRACTION_STATIC": "0.95",
}

model = Model(
    image_uri=sglang_image,
    model_data="s3://bucket/llama3-70b.tar.gz",
    env=env,
    role=role,
)
predictor = model.deploy(instance_type="ml.p4d.24xlarge")
```

#### HuggingFace Hub Configuration

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `HF_HUB_ENABLE_HF_TRANSFER` | Enable fast Rust-based downloads | `1` |
| `HF_TOKEN` | HuggingFace API token (for gated models) | `hf_...` |
| `HF_HOME` | Cache directory | `/tmp/.cache/huggingface` |
| `TRANSFORMERS_CACHE` | Transformers cache directory | `/tmp/.cache/huggingface/transformers` |

**Example - Loading Gated Model**:
```python
# For models like Llama 3 that require authentication
env = {
    "SM_VLLM_MODEL": "meta-llama/Llama-3-70b-hf",
    "HF_TOKEN": "hf_your_token_here",  # Get from huggingface.co/settings/tokens
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "8",
}

# vLLM will use HF_TOKEN to authenticate and download the model
```

---

## Model Loading Mechanisms

### Mechanism 1: Pre-Downloaded Model (Fastest)

**Setup**:
```bash
# Download model locally
huggingface-cli download meta-llama/Llama-3-70b-hf --local-dir ./llama3-70b/

# Create model.tar.gz
tar -czf model.tar.gz -C llama3-70b .

# Upload to S3
aws s3 cp model.tar.gz s3://my-bucket/models/llama3-70b.tar.gz
```

**Deployment**:
```python
env = {
    "SM_VLLM_MODEL": "/opt/ml/model",  # vLLM loads from local path
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "8",
}

model = Model(
    image_uri=vllm_image,
    model_data="s3://my-bucket/models/llama3-70b.tar.gz",  # SageMaker extracts to /opt/ml/model/
    env=env,
    role=role,
)
```

**Container Behavior**:
1. SageMaker downloads `model.tar.gz` from S3
2. Extracts to `/opt/ml/model/`
3. Container starts, vLLM sees `/opt/ml/model/config.json`
4. vLLM loads model from local path (no download needed)
5. **Startup time**: ~2-5 minutes

### Mechanism 2: Hub Download at Runtime (Simpler)

**Deployment**:
```python
env = {
    "SM_VLLM_MODEL": "meta-llama/Llama-3-70b-hf",  # HuggingFace model ID
    "HF_TOKEN": "hf_...",  # If gated model
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "8",
}

model = Model(
    image_uri=vllm_image,
    model_data=None,  # No pre-downloaded model
    env=env,
    role=role,
)
```

**Container Behavior**:
1. Container starts, vLLM sees model ID (not a local path)
2. vLLM calls HuggingFace Hub API
3. Downloads model to `/tmp/.cache/huggingface/hub/`
4. Loads model from cache
5. **Startup time**: ~10-30 minutes (depends on model size and network)

**Trade-offs**:
- **Pre-downloaded**: Faster startup, requires S3 storage
- **Hub download**: Simpler deployment, slower first startup, model cached for subsequent restarts

### Mechanism 3: Lazy Loading (Large Models)

For very large models (100B+), vLLM supports lazy loading:

```python
env = {
    "SM_VLLM_MODEL": "meta-llama/Llama-3.1-405b-hf",
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "16",
    "SM_VLLM_LOAD_FORMAT": "dummy",  # Don't actually load weights yet
}

# vLLM initializes structure but doesn't load all weights
# Weights are loaded on-demand as needed
```

---

## Entry Point Script Lifecycle

### vLLM Entry Point (`sagemaker_entrypoint.sh`)

**Script Flow**:
```bash
#!/bin/bash

# 1. Telemetry (non-blocking)
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# 2. Parse environment variables
PREFIX="SM_VLLM_"
ARGS=(--port 8080)

while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    ARGS+=("--${arg_name}" "$value")
done < <(env | grep "^${PREFIX}")

# 3. Launch vLLM server
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
```

**Example Transformation**:
```bash
# Environment variables:
SM_VLLM_MODEL=meta-llama/Llama-3-70b-hf
SM_VLLM_TENSOR_PARALLEL_SIZE=8
SM_VLLM_MAX_MODEL_LEN=4096

# Becomes:
python3 -m vllm.entrypoints.openai.api_server \
  --port 8080 \
  --model meta-llama/Llama-3-70b-hf \
  --tensor-parallel-size 8 \
  --max-model-len 4096
```

### PyTorch TorchServe Entry Point (`torchserve-entrypoint.py`)

**Script Responsibilities**:
1. Validate `/opt/ml/model/` contains model files
2. Generate TorchServe config file
3. Register model with TorchServe
4. Start TorchServe on port 8080
5. Expose SageMaker-compatible `/ping` and `/invocations` endpoints

---

## Configuration Patterns by Framework

### Pattern 1: vLLM (Declarative, Environment-Driven)

**Characteristics**:
- Configuration via environment variables only
- No config files needed
- Entry point script transforms env vars to CLI args
- Model auto-discovery from /opt/ml/model/ or Hub

**Best For**: Simple deployments, quick experimentation

**Example**:
```python
env = {"SM_VLLM_MODEL": "...", "SM_VLLM_TENSOR_PARALLEL_SIZE": "8"}
model.deploy(env=env)
```

### Pattern 2: DJL LMI (Declarative, Properties File)

**Characteristics**:
- Configuration via `serving.properties` file included in model.tar.gz
- Supports complex configurations (quantization, LoRA adapters)
- Properties file mounted to container

**Best For**: Complex deployments, multiple model variants, advanced quantization

**Example `serving.properties`**:
```properties
engine=Python
option.model_id=meta-llama/Llama-3-70b-hf
option.tensor_parallel_degree=8
option.max_rolling_batch_size=64
option.dtype=fp16
option.quantize=awq
option.rolling_batch=vllm
```

**Deployment**:
```bash
# Create model package
mkdir llama3-70b-deploy
echo "engine=Python" > llama3-70b-deploy/serving.properties
echo "option.model_id=meta-llama/Llama-3-70b-hf" >> llama3-70b-deploy/serving.properties
tar -czf model.tar.gz -C llama3-70b-deploy .

# Upload to S3 and deploy
```

### Pattern 3: HuggingFace TGI (Container-Native)

**Characteristics**:
- Configuration via environment variables (TGI-specific)
- Built-in optimizations for specific model architectures
- Rust-based for maximum performance

**Environment Variables**:
```python
env = {
    "HF_MODEL_ID": "meta-llama/Llama-3-70b-hf",
    "HF_TOKEN": "hf_...",
    "MAX_INPUT_LENGTH": "4096",
    "MAX_TOTAL_TOKENS": "8192",
    "MAX_BATCH_TOTAL_TOKENS": "16384",
    "QUANTIZE": "bitsandbytes-nf4",  # TGI-specific quantization
}
```

### Pattern 4: NeuronX (Compiler-Based)

**Characteristics**:
- Model must be pre-compiled for AWS Neuron
- Compilation happens offline, not at inference time
- Compiled model (`.neuron` files) included in model.tar.gz

**Workflow**:
```python
# Offline compilation (on Inf2/Trn1 instance):
import torch_neuronx
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b-hf")
neuron_model = torch_neuronx.trace(
    model,
    example_inputs,
    compiler_workdir="./llama3-8b-neuron",
)
neuron_model.save("./llama3-8b-neuron/model.neuron")

# Package and deploy
tar -czf model.tar.gz -C llama3-8b-neuron .
aws s3 cp model.tar.gz s3://bucket/neuron-models/llama3-8b.tar.gz

# Inference deployment
model = Model(
    image_uri=neuron_image,
    model_data="s3://bucket/neuron-models/llama3-8b.tar.gz",
    instance_type="ml.inf2.xlarge",
)
```

---

## Debugging Tips

### Viewing Container Logs

**During Training**:
```bash
# From CloudWatch Logs:
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

**During Inference**:
```bash
# From CloudWatch Logs:
aws logs tail /aws/sagemaker/Endpoints/<endpoint-name> --follow
```

### Inspecting Container Locally

```bash
# Pull DLC image
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.2-sagemaker

# Run interactively
docker run -it --gpus all \
  -e SM_VLLM_MODEL=meta-llama/Llama-3-8b-hf \
  -e SM_VLLM_TENSOR_PARALLEL_SIZE=1 \
  763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.2-sagemaker \
  /bin/bash

# Inside container:
ls /opt/ml/model/
env | grep SM_
cat /usr/local/bin/sagemaker_entrypoint.sh
```

### Common Issues

**Issue**: Model not loading from /opt/ml/model/

**Solution**: Check model.tar.gz structure
```bash
tar -tzf model.tar.gz | head -20
# Should show: config.json, model.safetensors, tokenizer.json, etc.
# NOT: llama3-70b/config.json (extra directory layer)
```

**Issue**: Out of memory during inference

**Solution**: Reduce GPU memory utilization
```python
env = {
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.85",  # Lower from 0.95
    "SM_VLLM_MAX_MODEL_LEN": "2048",  # Reduce max length
}
```

**Issue**: Multi-node training fails with NCCL timeout

**Solution**: Check EFA setup
```bash
# Inside container:
/opt/amazon/efa/bin/fi_info -p efa  # Should list EFA devices
```

---

## Summary

### Key Takeaways

1. **Standardized Paths**: SageMaker uses consistent paths across all containers
   - `/opt/ml/model/` for models
   - `/opt/ml/input/data/` for training data
   - `/opt/ml/checkpoints/` for checkpoints

2. **Environment-Driven Configuration**: Most inference containers use env vars
   - vLLM: `SM_VLLM_*`
   - SGLang: `SM_SGLANG_*`
   - Pattern: Declarative, easy to version control

3. **Entry Points Bridge APIs**: Scripts transform SageMaker conventions to framework CLIs
   - Environment variables → CLI arguments
   - Allows SageMaker integration without modifying framework code

4. **Model Loading Flexibility**: Multiple mechanisms
   - Pre-download to S3 (faster startup)
   - Hub download at runtime (simpler deployment)
   - Lazy loading (for very large models)

5. **Lifecycle Hooks**: Understand what runs when
   - Training: hostname fix → SageMaker module → user script
   - Inference: entrypoint → model server → health check → serving

**Next Steps**: See deployment templates for concrete examples of using these configurations in production.
