# Migration Guides for AWS DLC Containers

Comprehensive upgrade guides to safely migrate between container versions.

## Available Guides

### 1. [PyTorch 2.7 → 2.8 GPU Training](./PyTorch_2.7_to_2.8_GPU.md)
Migrate GPU training workloads to PyTorch 2.8 with improved torch.compile and FSDP2.

**Key Changes**:
- torch.compile backend defaults to `inductor-aot`
- FSDP state dict format updated
- Flash Attention 2.5 now default
- 15-20% faster training for transformers

**When to Use**: Upgrading training containers for better performance

### 2. [vLLM 0.10 → 0.11 Breaking Changes](./vLLM_0.10_to_0.11_Breaking_Changes.md)
Critical breaking changes in vLLM inference containers.

**Key Changes**:
- OpenAI API response format changed
- `tensor_parallel_size` moved to environment variable
- Default `top_p` changed from 1.0 to 0.95
- Native FP8 quantization for H100

**When to Use**: Upgrading vLLM inference endpoints

### 3. [NeuronSDK 2.24 → 2.26 Upgrade](./NeuronSDK_2.24_to_2.26_Upgrade.md)
Upgrade Trainium (trn1) and Inferentia2 (inf2) containers for AWS Neuron.

**Key Changes**:
- Flash Attention support on Inferentia2 (2.5x faster)
- Enhanced tensor parallelism (TP=32 on trn1.32xlarge)
- 15% lower HBM memory usage
- 30-40% faster compilation

**When to Use**: Upgrading Neuron-based training or inference

### 4. [HuggingFace Training Version Matrix](./HuggingFace_Training_Version_Matrix.md)
Compatibility matrix for HuggingFace libraries across all DLC versions.

**Covers**:
- transformers, accelerate, peft, trl version compatibility
- Model-specific requirements (Llama 3.3, Mixtral, Qwen 2.5)
- Fine-tuning method support (LoRA, QLoRA, DPO, KTO)
- Distributed training (DeepSpeed, FSDP)

**When to Use**: Planning upgrades or troubleshooting compatibility issues

## Quick Start

### Step 1: Identify Your Current Version

```bash
# For GPU containers
docker run --rm <current-image> python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# For vLLM containers
docker run --rm <current-image> python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# For Neuron containers
docker run --rm <current-image> python -c "import torch_neuronx; print(torch_neuronx.__version__)"
```

### Step 2: Choose the Right Guide

| Current Container | Target Container | Guide to Use |
|------------------|------------------|--------------|
| PyTorch 2.7 GPU Training | PyTorch 2.8 GPU Training | [PyTorch 2.7→2.8](./PyTorch_2.7_to_2.8_GPU.md) |
| vLLM 0.10 Inference | vLLM 0.11 Inference | [vLLM 0.10→0.11](./vLLM_0.10_to_0.11_Breaking_Changes.md) |
| NeuronSDK 2.24 | NeuronSDK 2.26 | [NeuronSDK 2.24→2.26](./NeuronSDK_2.24_to_2.26_Upgrade.md) |
| Any HuggingFace Training | Any HuggingFace Training | [Version Matrix](./HuggingFace_Training_Version_Matrix.md) |

### Step 3: Test in Staging

```python
# Example: Test new container in SageMaker
from sagemaker.pytorch import PyTorch

# Deploy to staging endpoint
estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_type="ml.p4d.24xlarge",
    instance_count=1,
    framework_version="2.8.0",  # New version
    py_version="py311",
)

estimator.fit(inputs={"training": "s3://bucket/data"})
```

### Step 4: Validate Performance

```python
# benchmark_migration.py
import time
import boto3

def compare_endpoints(old_endpoint, new_endpoint, n_runs=100):
    """Compare performance between old and new versions"""
    runtime = boto3.client('sagemaker-runtime')

    def benchmark(endpoint_name):
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body='{"inputs": "test prompt", "parameters": {"max_new_tokens": 100}}'
            )
            latencies.append(time.time() - start)
        return latencies

    old_latencies = benchmark(old_endpoint)
    new_latencies = benchmark(new_endpoint)

    import numpy as np
    print(f"Old version P95: {np.percentile(old_latencies, 95)*1000:.1f}ms")
    print(f"New version P95: {np.percentile(new_latencies, 95)*1000:.1f}ms")
    print(f"Speedup: {np.mean(old_latencies)/np.mean(new_latencies):.2f}x")

compare_endpoints("llama-3-v2.7", "llama-3-v2.8")
```

## Migration Strategies

### Blue/Green Deployment (Recommended)

Deploy new version alongside old version, gradually shift traffic.

```python
# 1. Deploy new version to separate endpoint
new_endpoint = model_v2.deploy(
    endpoint_name="model-v2-green",
    instance_type="ml.g5.12xlarge",
)

# 2. Route 10% traffic to new version (using application logic)
# 3. Monitor metrics for 24-48 hours
# 4. Gradually increase to 100%
# 5. Decommission old endpoint
```

**Pros**: Safe, easy rollback
**Cons**: Runs two endpoints (2x cost during migration)

### Canary Deployment

Deploy new version, route small percentage of traffic.

```python
# Use weighted routing in application layer
import random

def get_endpoint():
    if random.random() < 0.1:  # 10% canary
        return "model-v2-canary"
    else:
        return "model-v1-stable"

response = runtime.invoke_endpoint(EndpointName=get_endpoint(), ...)
```

**Pros**: Lower cost than blue/green
**Cons**: Requires application changes

### In-Place Update (Not Recommended for Production)

Update existing endpoint directly.

```python
# Update endpoint in-place
model_v2.deploy(
    endpoint_name="model-production",  # Same name
    update_endpoint=True,  # ⚠️ Triggers update
)
```

**Pros**: Simple, no routing changes
**Cons**: Downtime during update, difficult rollback

## Common Migration Patterns

### Pattern 1: Test → Staging → Production

```
1. Test in local notebook / SageMaker Studio
2. Deploy to staging endpoint
3. Run automated tests
4. Deploy to production (blue/green)
5. Monitor for 48 hours
6. Decommission old version
```

### Pattern 2: Model-by-Model Migration

If running multiple models:

```
Week 1: Migrate Model A (lowest traffic)
Week 2: Migrate Model B (medium traffic)
Week 3: Migrate Model C (highest traffic)
```

**Benefit**: Reduces blast radius of issues

### Pattern 3: Feature-Gated Rollout

Use feature flags to control which users see new version:

```python
# Application layer
def get_model_endpoint(user_id):
    if is_beta_user(user_id):
        return "model-v2-beta"
    else:
        return "model-v1-stable"
```

## Validation Checklist

Before migrating to production:

### Functional Validation
- [ ] Model loads successfully
- [ ] Inference produces expected outputs
- [ ] All API endpoints work
- [ ] Streaming (if used) functions correctly
- [ ] Error handling works as expected

### Performance Validation
- [ ] Latency P50/P95/P99 measured
- [ ] Throughput (requests/sec) measured
- [ ] Memory usage checked
- [ ] GPU/Neuron utilization monitored
- [ ] Performance meets SLAs

### Accuracy Validation
- [ ] Run on test set
- [ ] Compare outputs with old version
- [ ] Check for regressions in key metrics
- [ ] Validate on edge cases

### Integration Validation
- [ ] Load balancer configuration updated
- [ ] Monitoring dashboards show metrics
- [ ] Alerts configured
- [ ] Logging works correctly
- [ ] Security policies enforced

### Rollback Validation
- [ ] Rollback procedure documented
- [ ] Old container image saved
- [ ] Old model artifacts available in S3
- [ ] Rollback tested in staging

## Troubleshooting Migration Issues

### Issue: Performance Regression

**Symptoms**: New version slower than old version

**Debug Steps**:
1. Check instance type (same as before?)
2. Review configuration (batch size, TP degree, etc.)
3. Compare CPU/GPU utilization
4. Check for compilation issues (Neuron)
5. Review release notes for known performance issues

**Solution**:
```python
# Benchmark both versions side-by-side
from Claude_tutorials.Benchmarks.scripts.benchmark_runner import BenchmarkRunner

old_results = BenchmarkRunner("old-endpoint").run()
new_results = BenchmarkRunner("new-endpoint").run()

print(f"Throughput: {old_results.throughput_rps} → {new_results.throughput_rps}")
print(f"Latency P95: {old_results.latency_p95} → {new_results.latency_p95}")
```

### Issue: Accuracy Degradation

**Symptoms**: Model outputs differ significantly

**Debug Steps**:
1. Check for quantization changes
2. Review Flash Attention settings
3. Verify model checkpoint compatibility
4. Check for tokenizer changes
5. Test with `temperature=0.0` for determinism

**Solution**:
```python
# Force deterministic generation
response = model.generate(
    input_ids,
    temperature=0.0,  # Greedy
    do_sample=False,
    seed=42,
)
```

### Issue: Out of Memory (OOM)

**Symptoms**: Container crashes with OOM error

**Debug Steps**:
1. Check memory usage vs previous version
2. Review batch size configuration
3. Check for memory leaks (long-running tests)
4. Verify instance type has enough memory

**Solution**:
```python
# Reduce memory usage
model = LLM(
    model="llama-3-70b",
    gpu_memory_utilization=0.85,  # Reduce from 0.9
    max_model_len=2048,  # Limit sequence length
)
```

## Monitoring During Migration

### Key Metrics to Track

```python
# CloudWatch metrics to monitor
metrics = [
    "ModelLatency",  # P50, P95, P99
    "Invocations",  # Requests/sec
    "ModelSetupTime",  # Cold start time
    "MemoryUtilization",  # Memory usage
    "CPUUtilization",  # CPU usage
    "GPUUtilization",  # GPU usage (if applicable)
]

# Set up alarms
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_alarm(
    AlarmName='ModelLatencyP95-High',
    MetricName='ModelLatency',
    Namespace='AWS/SageMaker',
    Statistic='p95',
    Period=300,
    EvaluationPeriods=2,
    Threshold=1000,  # 1 second
    ComparisonOperator='GreaterThanThreshold',
)
```

### Comparison Dashboard

Create a dashboard comparing old vs new:

```
┌─────────────────────────────────────────┐
│ Latency P95 (ms)                        │
│ Old Version: 520ms ████████             │
│ New Version: 450ms ███████ ✓ 13% faster│
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Throughput (req/sec)                    │
│ Old Version: 24.5 ████████              │
│ New Version: 28.3 █████████ ✓ 15% gain │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Memory Usage (GB)                       │
│ Old Version: 18.2 ████████              │
│ New Version: 19.1 ████████▌ ⚠️ 5% more │
└─────────────────────────────────────────┘
```

## Cost Analysis

Before migrating, estimate cost impact:

```python
from Claude_tutorials.CostCalculator.cost_calculator import CostCalculator

# Compare costs
old_version_cost = 1.01 * 730  # ml.g5.xlarge for 1 month
new_version_cost = 1.01 * 730  # Same instance type

# But check if performance gain allows smaller instance
calculator = CostCalculator(
    model_size_gb=70,
    requests_per_day=50000,
    max_latency_p95_ms=600
)

recommendations = calculator.find_optimal_deployment()
print(f"Recommended instance: {recommendations[0].instance_type}")
print(f"Monthly cost: ${recommendations[0].monthly_cost:.2f}")
```

## Next Steps

1. Choose migration guide based on your container type
2. Test migration in development environment
3. Deploy to staging and validate
4. Create rollback plan
5. Execute production migration using blue/green deployment
6. Monitor for 48 hours
7. Document any issues and learnings

## Additional Resources

- [AWS DLC Release Notes](https://github.com/aws/deep-learning-containers/releases)
- [SageMaker Deployment Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)
- [Performance Benchmark Suite](../Benchmarks/README.md)
- [Cost Calculator](../CostCalculator/README.md)
- [Testing Framework](../Testing/)
