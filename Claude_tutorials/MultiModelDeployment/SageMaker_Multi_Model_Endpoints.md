# SageMaker Multi-Model Endpoints (MME) for Foundation Models

## Overview

SageMaker Multi-Model Endpoints (MME) allow you to host multiple models on a single endpoint, dramatically reducing costs for scenarios with many models.

**Key Benefits**:
- **Cost Savings**: 50-90% reduction vs individual endpoints
- **Simplified Management**: Single endpoint for multiple models
- **Dynamic Loading**: Models loaded on-demand from S3
- **Auto-Scaling**: Shared compute resources

## Architecture

```
┌────────────────────────────────────────────┐
│ Client Application                         │
└────────────┬───────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│ SageMaker Multi-Model Endpoint             │
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │ Model Serving Container               │ │
│  │                                       │ │
│  │  Memory Cache                         │ │
│  │  ├─ Llama-3-8B (active)              │ │
│  │  ├─ Mistral-7B (active)              │ │
│  │  └─ CodeLlama-7B (cached)            │ │
│  │                                       │ │
│  │  S3 Model Store                       │ │
│  │  ├─ s3://models/llama-3-8b.tar.gz    │ │
│  │  ├─ s3://models/mistral-7b.tar.gz    │ │
│  │  ├─ s3://models/codellama-7b.tar.gz  │ │
│  │  └─ ... (50+ models)                 │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

**How it works**:
1. Client specifies model in TargetModel header
2. SageMaker checks if model is in memory
3. If not, downloads from S3 and loads
4. Model inference runs
5. Least-recently-used models evicted when memory full

## When to Use MME

### Perfect For
- **Customer-specific models**: SaaS with per-customer fine-tuned models
- **A/B testing**: Testing multiple model variants
- **Multi-task deployment**: Different models for different tasks (summarization, QA, code)
- **Low-traffic models**: Many models with sporadic usage

### Not Suitable For
- **High-throughput single model**: Use dedicated endpoint instead
- **Ultra-low latency**: Cold starts add 5-15s latency
- **Large models (>100GB)**: Memory constraints make loading impractical
- **Real-time streaming**: MME doesn't support streaming well

## Cost Comparison

### Scenario: 20 Models, Each 7B params

**Individual Endpoints** (20 × ml.g5.xlarge):
```
20 models × $1.01/hr × 730 hrs/month = $14,746/month
```

**Multi-Model Endpoint** (4 × ml.g5.xlarge):
```
4 instances × $1.01/hr × 730 hrs/month = $2,949/month
```

**Savings**: $11,797/month (80% reduction)

### When Does MME Save Money?

| # Models | Avg Traffic/Model | Individual Cost | MME Cost | Savings |
|----------|------------------|-----------------|----------|---------|
| 5 | Low | $3,687 | $2,949 | 20% |
| 10 | Low | $7,373 | $2,949 | 60% |
| 50 | Low | $36,865 | $2,949 | 92% |
| 5 | High | $3,687 | $7,373 | -100% ❌ |

**Rule of Thumb**: MME saves money when you have 5+ models with low to medium traffic each.

## Implementation Guide

### Step 1: Prepare Models for MME

Each model must be packaged as a `.tar.gz` archive with `model.safetensors` or `model.bin`.

```python
# prepare_models_for_mme.py
import os
import tarfile
from transformers import AutoModelForCausalLM, AutoTokenizer

def package_model_for_mme(model_id, output_dir):
    """Package model for MME deployment"""
    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Save locally
    temp_dir = f"/tmp/{model_id.replace('/', '_')}"
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    # Create tar.gz
    output_file = f"{output_dir}/{model_id.replace('/', '_')}.tar.gz"
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(temp_dir, arcname=".")

    print(f"✓ Packaged {model_id} → {output_file}")
    return output_file

# Package multiple models
models = [
    "meta-llama/Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "codellama/CodeLlama-7b-Instruct-hf",
]

for model_id in models:
    package_model_for_mme(model_id, "/tmp/mme_models")
```

### Step 2: Upload to S3

```python
# upload_to_s3.py
import boto3
import os

s3_client = boto3.client('s3')
bucket = "my-mme-models"
prefix = "foundation-models"

# Upload all packaged models
for filename in os.listdir("/tmp/mme_models"):
    if filename.endswith(".tar.gz"):
        local_path = f"/tmp/mme_models/{filename}"
        s3_key = f"{prefix}/{filename}"

        s3_client.upload_file(local_path, bucket, s3_key)
        print(f"✓ Uploaded s3://{bucket}/{s3_key}")
```

### Step 3: Create Multi-Model Endpoint

```python
# deploy_mme.py
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.huggingface import HuggingFaceModel
import boto3

role = "arn:aws:iam::123456789012:role/SageMakerRole"

# Create base model (defines the container)
base_model = HuggingFaceModel(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.0-tgi2.4.1-gpu-py311-cu121-ubuntu22.04",
    role=role,
)

# Create multi-model
mme = MultiDataModel(
    name="foundation-models-mme",
    model_data_prefix="s3://my-mme-models/foundation-models/",
    model=base_model,
)

# Deploy
mme.deploy(
    initial_instance_count=2,
    instance_type="ml.g5.2xlarge",
    endpoint_name="foundation-models-mme",
)

print("✓ Multi-Model Endpoint deployed!")
```

### Step 4: Invoke with Different Models

```python
# invoke_mme.py
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

def invoke_model(model_name, prompt):
    """Invoke specific model on MME"""
    response = runtime.invoke_endpoint(
        EndpointName="foundation-models-mme",
        ContentType="application/json",
        TargetModel=f"{model_name}.tar.gz",  # ⭐ Specify model
        Body=json.dumps({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 100}
        })
    )

    result = json.loads(response['Body'].read())
    return result[0]['generated_text']

# Invoke different models
print(invoke_model("meta-llama_Llama-3-8B-Instruct", "What is AI?"))
print(invoke_model("mistralai_Mistral-7B-Instruct-v0.3", "Explain quantum computing"))
print(invoke_model("codellama_CodeLlama-7b-Instruct-hf", "Write a quicksort in Python"))
```

## Advanced Patterns

### Pattern 1: Customer-Specific Fine-Tuned Models

Deploy a base model plus customer-specific LoRA adapters.

```python
# customer_specific_mme.py

# S3 structure:
# s3://models/
#   ├── base-llama-3-8b.tar.gz (base model)
#   ├── customer-123-adapter.tar.gz (LoRA for customer 123)
#   ├── customer-456-adapter.tar.gz (LoRA for customer 456)
#   └── ...

def invoke_for_customer(customer_id, prompt):
    """Invoke customer-specific model"""
    model_name = f"customer-{customer_id}-adapter"

    response = runtime.invoke_endpoint(
        EndpointName="customer-models-mme",
        ContentType="application/json",
        TargetModel=f"{model_name}.tar.gz",
        Body=json.dumps({"inputs": prompt})
    )

    return json.loads(response['Body'].read())

# SaaS application
@app.route("/api/generate", methods=["POST"])
def generate_text():
    customer_id = request.headers.get("X-Customer-ID")
    prompt = request.json["prompt"]

    result = invoke_for_customer(customer_id, prompt)
    return jsonify(result)
```

### Pattern 2: Task-Specific Model Routing

Route requests to specialized models based on task.

```python
# task_router.py

TASK_TO_MODEL = {
    "summarization": "bart-large-cnn",
    "question_answering": "roberta-large-squad2",
    "code_generation": "codellama-7b",
    "translation": "marian-mt-en-de",
    "sentiment": "distilbert-sentiment",
}

def route_to_model(task, input_text):
    """Route to appropriate model based on task"""
    model_name = TASK_TO_MODEL.get(task)

    if not model_name:
        raise ValueError(f"Unknown task: {task}")

    response = runtime.invoke_endpoint(
        EndpointName="task-models-mme",
        ContentType="application/json",
        TargetModel=f"{model_name}.tar.gz",
        Body=json.dumps({"inputs": input_text})
    )

    return json.loads(response['Body'].read())

# Usage
summary = route_to_model("summarization", long_article)
answer = route_to_model("question_answering", {"question": "...", "context": "..."})
code = route_to_model("code_generation", "write quicksort")
```

### Pattern 3: A/B Testing Multiple Variants

Test multiple model versions simultaneously.

```python
# ab_testing.py
import random

MODEL_VARIANTS = {
    "control": "llama-3-8b-base.tar.gz",
    "variant_a": "llama-3-8b-finetuned-v1.tar.gz",
    "variant_b": "llama-3-8b-finetuned-v2.tar.gz",
}

# Traffic allocation
TRAFFIC_SPLIT = {
    "control": 0.80,
    "variant_a": 0.10,
    "variant_b": 0.10,
}

def select_variant():
    """Select model variant based on traffic split"""
    rand = random.random()
    cumulative = 0

    for variant, prob in TRAFFIC_SPLIT.items():
        cumulative += prob
        if rand < cumulative:
            return variant

    return "control"

def generate_with_ab_test(prompt):
    """Generate with A/B testing"""
    variant = select_variant()
    model_file = MODEL_VARIANTS[variant]

    response = runtime.invoke_endpoint(
        EndpointName="ab-test-mme",
        ContentType="application/json",
        TargetModel=model_file,
        Body=json.dumps({"inputs": prompt})
    )

    result = json.loads(response['Body'].read())

    # Log for analytics
    log_ab_test_result(variant, prompt, result)

    return result

# Usage
output = generate_with_ab_test("What is machine learning?")
```

## Performance Considerations

### Cold Start Latency

**Model Loading Times** (ml.g5.2xlarge):

| Model Size | First Request | Subsequent Requests |
|-----------|---------------|---------------------|
| 7B (14GB) | 8-12 seconds | 300-500ms |
| 13B (26GB) | 15-20 seconds | 400-600ms |
| 34B (68GB) | 30-45 seconds | 600-900ms |

**Mitigation Strategies**:

1. **Pre-warming**:
```python
def prewarm_models():
    """Send dummy requests to load models into memory"""
    models = ["llama-3-8b", "mistral-7b", "codellama-7b"]

    for model in models:
        try:
            runtime.invoke_endpoint(
                EndpointName="foundation-models-mme",
                ContentType="application/json",
                TargetModel=f"{model}.tar.gz",
                Body=json.dumps({"inputs": "test"})
            )
            print(f"✓ Pre-warmed {model}")
        except Exception as e:
            print(f"✗ Failed to pre-warm {model}: {e}")

# Run after deployment or periodically
prewarm_models()
```

2. **Sticky routing** (keep users on same instance):
```python
import hashlib

def get_instance_hint(user_id):
    """Route same user to same instance"""
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return hash_val % NUM_INSTANCES

# In application
instance_hint = get_instance_hint(customer_id)
# Pass as custom header (requires ALB configuration)
```

### Memory Management

**Calculate models per instance**:

```python
def calculate_mme_capacity(instance_type, model_size_gb):
    """Calculate how many models fit in memory"""
    # Instance memory (GB)
    INSTANCE_MEMORY = {
        "ml.g5.xlarge": 24,
        "ml.g5.2xlarge": 24,
        "ml.g5.12xlarge": 96,
        "ml.p4d.24xlarge": 320,
    }

    available_memory = INSTANCE_MEMORY[instance_type]

    # Reserve 20% for overhead
    usable_memory = available_memory * 0.80

    # Number of models that fit
    models_in_memory = int(usable_memory / model_size_gb)

    return models_in_memory

# Example
capacity = calculate_mme_capacity("ml.g5.2xlarge", 14)  # 7B model ≈ 14GB
print(f"Can hold {capacity} models in memory simultaneously")  # Output: 1 model
```

**Recommendation**: Use larger instances (g5.12xlarge, p4d.24xlarge) for MME to fit more models in memory and reduce cold starts.

### Throughput Optimization

**Instance sizing**:

| Instance Type | GPU Memory | Recommended Max Models | Concurrent Models in Memory |
|--------------|-----------|----------------------|---------------------------|
| ml.g5.xlarge | 24GB | 10-20 small models | 1 × 7B |
| ml.g5.2xlarge | 24GB | 10-20 small models | 1 × 7B |
| ml.g5.12xlarge | 96GB | 50-100 models | 3-4 × 7B or 1 × 70B |
| ml.p4d.24xlarge | 320GB | 100-500 models | 10 × 7B or 2-3 × 70B |

## Monitoring and Debugging

### Key Metrics

```python
# monitor_mme.py
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')

def get_mme_metrics(endpoint_name):
    """Get MME-specific metrics"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    # Model loading time
    loading_time = cloudwatch.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='ModelLoadingWaitTime',
        Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=['Average', 'Maximum']
    )

    # Model cache hit rate
    cache_hits = cloudwatch.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='ModelCacheHit',
        Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=['Sum']
    )

    return {
        "avg_loading_time": loading_time['Datapoints'][0]['Average'] if loading_time['Datapoints'] else 0,
        "cache_hits": sum(d['Sum'] for d in cache_hits['Datapoints']),
    }

metrics = get_mme_metrics("foundation-models-mme")
print(f"Avg loading time: {metrics['avg_loading_time']:.1f}s")
print(f"Cache hit rate: {metrics['cache_hits']}")
```

### Logging Model Invocations

```python
# Track which models are being used
import json
from datetime import datetime

def log_model_invocation(model_name, latency_ms):
    """Log model usage for analytics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "latency_ms": latency_ms,
    }

    # Send to CloudWatch Logs
    logs = boto3.client('logs')
    logs.put_log_events(
        logGroupName='/aws/sagemaker/mme',
        logStreamName='model-invocations',
        logEvents=[{
            'timestamp': int(datetime.now().timestamp() * 1000),
            'message': json.dumps(log_entry)
        }]
    )
```

## Best Practices

1. **Instance Sizing**: Use larger instances (g5.12xlarge or p4d.24xlarge) to fit more models in memory
2. **Model Packaging**: Keep models small (<50GB each) for faster loading
3. **Pre-warming**: Pre-load frequently used models after deployment
4. **Monitoring**: Track ModelLoadingWaitTime and ModelCacheHit metrics
5. **Auto-Scaling**: Configure based on aggregate traffic, not per-model
6. **S3 Optimization**: Use S3 Transfer Acceleration for faster model downloads
7. **Versioning**: Include version in model filename (e.g., `llama-3-8b-v2.tar.gz`)

## Limitations

- **No Streaming**: MME doesn't support streaming responses well
- **Cold Starts**: First request to a model takes 5-45 seconds
- **Memory Constraints**: Large models (>100GB) problematic
- **Single Container**: All models must use same container image

## Migration from Individual Endpoints

```python
# migrate_to_mme.py

# Before: Individual endpoints
endpoints = [
    "llama-3-8b-endpoint",
    "mistral-7b-endpoint",
    "codellama-7b-endpoint",
]

# After: Single MME
mme_endpoint = "all-models-mme"

# Update client code
def invoke_endpoint_old(model_name, prompt):
    endpoint_name = f"{model_name}-endpoint"
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"inputs": prompt})
    )
    return json.loads(response['Body'].read())

def invoke_endpoint_new(model_name, prompt):
    response = runtime.invoke_endpoint(
        EndpointName="all-models-mme",  # ✓ Single endpoint
        ContentType="application/json",
        TargetModel=f"{model_name}.tar.gz",  # ✓ Specify model
        Body=json.dumps({"inputs": prompt})
    )
    return json.loads(response['Body'].read())
```

## Cost Optimization Example

**Scenario**: SaaS company with 50 customer-specific fine-tuned models

**Before (Individual Endpoints)**:
- 50 endpoints × ml.g5.xlarge
- Cost: 50 × $1.01/hr × 730 hrs = **$36,865/month**

**After (MME)**:
- 1 MME with 8 × ml.g5.12xlarge instances
- Cost: 8 × $7.09/hr × 730 hrs = **$41,406/month**

Wait, that's more expensive! 🤔

**Better MME Configuration**:
- 1 MME with 3 × ml.g5.12xlarge instances
- Cost: 3 × $7.09/hr × 730 hrs = **$15,527/month**
- **Savings: $21,338/month (58%)**

**Key Insight**: Right-size based on concurrent usage, not total models!

## Next Steps

- Review [Monitoring & Observability](../Monitoring/) for tracking MME performance
- See [Cost Calculator](../CostCalculator/) to compare MME vs individual endpoints
- Check [Security](../Security/) for best practices

## Resources

- [SageMaker MME Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)
- [MME Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Multi-Model Benchmarks](../Benchmarks/)
