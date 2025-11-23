# Cost Optimization Calculator for DLC Deployments

Interactive tool to find the most cost-effective container and instance combination for your workload.

## Overview

This calculator helps you choose optimal deployments by:
- Analyzing model size requirements
- Estimating throughput and latency
- Calculating total monthly costs
- Computing cost per million tokens
- Recommending the best container + instance combinations

## Quick Start

### Basic Usage

```bash
python cost_calculator.py \
  --model-size-gb 70 \
  --requests-per-day 50000 \
  --max-latency-ms 600
```

### Example Output

```
================================================================================
COST CALCULATOR INPUT
================================================================================
Model Size:          70 GB
Expected Traffic:    50,000 requests/day
Max Latency (P95):   600ms
================================================================================

================================================================================
COST OPTIMIZATION RECOMMENDATIONS
================================================================================

1. NEURONX on ml.inf2.xlarge (1 instance(s))
   Monthly Cost:        $554.80
   Cost/1M tokens:      $0.41
   Est. Throughput:     2.5 req/sec
   Est. P95 Latency:    600ms
   💡 BEST CHOICE ⭐

2. VLLM on ml.g5.xlarge (1 instance(s))
   Monthly Cost:        $737.30
   Cost/1M tokens:      $1.64
   Est. Throughput:     6.0 req/sec
   Est. P95 Latency:    500ms

3. SGLANG on ml.g5.xlarge (1 instance(s))
   Monthly Cost:        $737.30
   Cost/1M tokens:      $1.79
   Est. Throughput:     5.5 req/sec
   Est. P95 Latency:    500ms

================================================================================
```

## Use Cases

### 1. Small Model, Low Traffic

```bash
python cost_calculator.py \
  --model-size-gb 14 \
  --requests-per-day 10000 \
  --max-latency-ms 500
```

**Recommendation**: vLLM on ml.g5.xlarge ($737/month)
- Single GPU handles low traffic efficiently
- Fast latency (<500ms)
- Simple deployment

### 2. Large Model, High Traffic

```bash
python cost_calculator.py \
  --model-size-gb 140 \
  --requests-per-day 100000 \
  --max-latency-ms 1000
```

**Recommendation**: vLLM on ml.g5.12xlarge (2 instances, $10,351/month)
- Multi-GPU instances for large models
- Horizontal scaling for high throughput
- Meets latency SLA

### 3. Cost-Sensitive, Moderate Traffic

```bash
python cost_calculator.py \
  --model-size-gb 70 \
  --requests-per-day 25000 \
  --max-latency-ms 800
```

**Recommendation**: NeuronX on ml.inf2.xlarge ($555/month)
- Lowest cost per token
- Acceptable latency for non-real-time use cases
- AWS-optimized hardware

### 4. ARM64 for Maximum Efficiency

```bash
python cost_calculator.py \
  --model-size-gb 14 \
  --requests-per-day 50000 \
  --max-latency-ms 600
```

**Recommendation**: vLLM-ARM64 on ml.g5g.xlarge ($446/month)
- Lowest cost for small models
- Good balance of performance and cost

## Advanced Options

### Show More Recommendations

```bash
python cost_calculator.py \
  --model-size-gb 70 \
  --requests-per-day 50000 \
  --max-latency-ms 600 \
  --top-n 10
```

### Strict Latency Requirements

```bash
python cost_calculator.py \
  --model-size-gb 70 \
  --requests-per-day 100000 \
  --max-latency-ms 400  # Strict SLA
```

## How It Works

### 1. Instance Compatibility Check

The calculator filters out incompatible combinations:
- NeuronX containers require Inferentia2 instances
- vLLM-ARM64 requires Graviton instances
- Models must fit in instance GPU memory

### 2. Throughput Estimation

Estimates requests/second based on:
- Container type (vLLM > SGLang > NeuronX)
- GPU count (more GPUs = higher throughput)
- Model size (larger models = slower inference)

**Example**:
- vLLM on ml.g5.12xlarge (4 GPUs) = ~24 req/sec
- NeuronX on ml.inf2.xlarge = ~2.5 req/sec

### 3. Latency Estimation

P95 latency varies by:
- Hardware (Multi-GPU < Single GPU < Inferentia2)
- Container optimization
- Model complexity

**Typical ranges**:
- Multi-GPU (8+ GPUs): 400ms
- Single GPU: 500ms
- ARM64: 550ms
- Inferentia2: 600ms

### 4. Cost Calculation

**Monthly Cost**:
```
monthly_cost = hourly_rate × 730 hours × instance_count
```

**Cost per Million Tokens**:
```
tokens_per_hour = throughput_rps × 3600 × 2148 tokens/request
cost_per_1M_tokens = (hourly_cost / tokens_per_hour) × 1,000,000
```

## Instance Pricing Database

The calculator includes pricing for:

### GPU Instances
- `ml.g5.xlarge`: $1.01/hr (1 GPU, 24GB)
- `ml.g5.2xlarge`: $1.52/hr (1 GPU, 24GB)
- `ml.g5.12xlarge`: $7.09/hr (4 GPUs, 96GB)
- `ml.p4d.24xlarge`: $32.77/hr (8 GPUs, 320GB)

### ARM64 + GPU
- `ml.g5g.xlarge`: $0.61/hr (1 GPU, 16GB)
- `ml.g5g.16xlarge`: $4.20/hr (1 GPU, 24GB)

### Inferentia2
- `ml.inf2.xlarge`: $0.76/hr (32GB)
- `ml.inf2.48xlarge`: $12.98/hr (384GB)

## Assumptions and Limitations

### Assumptions
- Average request: 2,048 input tokens + 100 output tokens = 2,148 tokens
- 24/7 operation (730 hours/month)
- No spot instance discounts applied
- Simplified throughput models (use benchmarks for production)

### Limitations
- Throughput estimates are conservative approximations
- Actual latency depends on model architecture, batch size, sequence length
- Network and cold start overhead not included
- Does not account for auto-scaling costs

## Production Recommendations

### Before Deployment
1. **Run Benchmarks**: Use the Benchmark Suite to measure actual performance
2. **Load Test**: Validate throughput under realistic traffic patterns
3. **Monitor Latency**: Track P50/P95/P99 in production
4. **Enable Auto-Scaling**: Handle traffic spikes efficiently

### Cost Optimization Tips
1. **Use Spot Instances**: 50-70% savings for non-critical workloads
2. **Right-Size**: Start small, scale based on metrics
3. **Quantization**: AWQ/GPTQ can reduce instance size requirements
4. **Batch Processing**: Use Batch Transform for offline workloads
5. **Multi-Model Endpoints**: Share instances across models

## Integration with Other Tools

### With Benchmark Suite
```bash
# 1. Get baseline performance
python ../Benchmarks/scripts/benchmark_runner.py \
  --container vllm \
  --endpoint-name test-endpoint

# 2. Use actual metrics in calculator
python cost_calculator.py \
  --model-size-gb 70 \
  --requests-per-day 50000 \
  --max-latency-ms 600
```

### With CloudWatch Metrics
```python
# Extract actual request rate from CloudWatch
import boto3

cloudwatch = boto3.client('cloudwatch')
metrics = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='Invocations',
    Dimensions=[{'Name': 'EndpointName', 'Value': 'my-endpoint'}],
    StartTime=datetime.now() - timedelta(days=7),
    EndTime=datetime.now(),
    Period=86400,
    Statistics=['Sum']
)

requests_per_day = int(metrics['Datapoints'][0]['Sum'])
```

## Troubleshooting

### "No configurations found"
- **Cause**: Latency requirement too strict or model too large
- **Solution**: Increase `--max-latency-ms` or use larger instances

### Unexpected high costs
- **Check**: Verify instance count calculation
- **Solution**: Consider quantization or batch processing

### Latency higher than estimated
- **Cause**: Estimates are conservative
- **Solution**: Run actual benchmarks with your model

## Next Steps

After finding optimal configuration:
1. Review [Performance Benchmarks](../Benchmarks/README.md)
2. Deploy using [Deployment Guides](../FoundationModel_Deployment_Guides/)
3. Set up [Monitoring](../Monitoring/) (coming in Tier 2)
4. Implement [Testing](../Testing/) suite

## Support

For questions or issues:
- Check [Container Annotations](../../Annotated_DLC_Containers/)
- Review [Security Best Practices](../Security/)
- See [Integration Examples](../IntegrationExamples/)
