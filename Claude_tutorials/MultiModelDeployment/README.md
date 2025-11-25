# Multi-Model Deployment Patterns

Advanced patterns for deploying and managing multiple foundation models cost-effectively.

## Overview

When deploying multiple models, choosing the right architecture can save 50-90% on costs while maintaining performance. This guide covers:

1. **Multi-Model Endpoints (MME)**: Host many models on shared infrastructure
2. **Routing Strategies**: Intelligently route requests to optimal models

## Quick Decision Matrix

### When to Use Multi-Model Endpoints

| Scenario | Use MME? | Alternative |
|----------|----------|-------------|
| 50+ low-traffic models | ✅ Yes | Individual endpoints waste money |
| 5 high-traffic models | ❌ No | Individual endpoints perform better |
| Customer-specific fine-tuned models | ✅ Yes | Perfect use case |
| A/B testing 3-5 variants | ✅ Maybe | Both MME and individual work |
| Single model, high throughput | ❌ No | Dedicated endpoint optimized |

### Cost Savings Calculator

**Formula**:
```
Individual Endpoints Cost = N_models × Instance_Cost × 730 hrs
MME Cost = N_instances × Instance_Cost × 730 hrs

Savings = (Individual - MME) / Individual × 100%
```

**Example** (20 models, each on ml.g5.xlarge):
- Individual: 20 × $1.01 × 730 = $14,746/month
- MME (4 instances): 4 × $1.01 × 730 = $2,949/month
- **Savings: 80%** ($11,797/month)

## Architecture Patterns

### Pattern 1: Pure Multi-Model Endpoint

**Use Case**: Many low-traffic models (e.g., customer-specific fine-tuned models)

```
Single Endpoint → 50+ models on S3 → Load on demand
```

**Pros**:
- Lowest cost (50-90% savings)
- Infinite scalability (add models to S3)
- Simple management

**Cons**:
- Cold start latency (5-15s first request)
- Limited by instance memory

**See**: [Multi-Model Endpoints Guide](./SageMaker_Multi_Model_Endpoints.md)

### Pattern 2: Hybrid (MME + Dedicated Endpoints)

**Use Case**: Mix of high-traffic and low-traffic models

```
High Traffic Models → Dedicated Endpoints (3-5 models)
Low Traffic Models → MME (45+ models)
```

**Example**:
- Dedicated: Llama-3-70B, Llama-3-8B, CodeLlama-34B (80% of traffic)
- MME: 47 specialized/customer models (20% of traffic)

**Cost**:
- Dedicated: 3 × $7.09 × 730 = $15,527
- MME: 2 × $7.09 × 730 = $10,351
- **Total: $25,878** vs **$36,865** (30% savings)

### Pattern 3: Task-Specific Routing

**Use Case**: Different tasks require different specialized models

```
Request → Task Classifier → Route to Specialist Model
    ├─ Code Generation → CodeLlama-34B
    ├─ Summarization → Llama-3-70B
    ├─ Translation → NLLB-200
    └─ General Chat → Llama-3-8B
```

**Benefits**:
- Better quality (specialized models)
- Cost-effective (right-sized for task)
- Optimal latency (task-appropriate model)

**See**: [Routing Strategies Guide](./Model_Routing_Strategies.md)

### Pattern 4: Tier-Based Deployment

**Use Case**: SaaS with multiple customer tiers

```
Free Tier → Llama-3-8B (quantized, shared)
Standard Tier → Llama-3-8B (full precision, shared)
Premium Tier → Llama-3-70B (shared)
Enterprise Tier → Dedicated Llama-3-405B (isolated)
```

**Pricing**:
| Tier | Model | Endpoint | Monthly Cost (per customer) |
|------|-------|----------|---------------------------|
| Free | 8B Quant | Shared MME | $0 (included in freemium) |
| Standard | 8B Full | Shared | $0.50 |
| Premium | 70B | Shared | $5.00 |
| Enterprise | 405B | Dedicated | $500+ |

## Implementation Examples

### Example 1: SaaS with Custom Models Per Customer

**Scenario**: 100 customers, each with fine-tuned Llama-3-8B

```python
# deploy_saas_mme.py
from sagemaker.multidatamodel import MultiDataModel

# Upload customer models to S3
# s3://models/customers/
#   ├── customer-001.tar.gz
#   ├── customer-002.tar.gz
#   └── ...

mme = MultiDataModel(
    name="customer-models",
    model_data_prefix="s3://models/customers/",
    model=base_model,
)

mme.deploy(
    initial_instance_count=5,  # Handle 100 customers
    instance_type="ml.g5.12xlarge",  # Fit 3-4 models in memory
)

# In application
@app.route("/api/generate")
def generate():
    customer_id = request.headers["X-Customer-ID"]
    model_file = f"customer-{customer_id}.tar.gz"

    response = runtime.invoke_endpoint(
        EndpointName="customer-models",
        TargetModel=model_file,  # Customer-specific model
        ContentType="application/json",
        Body=json.dumps({"inputs": request.json["prompt"]})
    )

    return jsonify(json.loads(response['Body'].read()))
```

**Cost Analysis**:
- Individual endpoints: 100 × $1.01 × 730 = $73,730/month
- MME (5 instances): 5 × $7.09 × 730 = $25,878/month
- **Savings: $47,852/month (65%)**

### Example 2: Task Router with Fallbacks

**Scenario**: Route to specialized models with fallback to general model

```python
# task_router_with_fallback.py
from enum import Enum

class Task(Enum):
    CODE = "code"
    MATH = "math"
    CHAT = "chat"

TASK_ENDPOINTS = {
    Task.CODE: ("codellama-34b", "llama-3-70b"),  # primary, fallback
    Task.MATH: ("llama-3-70b", "llama-3-8b"),
    Task.CHAT: ("llama-3-8b", "mistral-7b"),
}

def invoke_with_fallback(task: Task, prompt: str) -> dict:
    """Invoke task-specific model with fallback"""
    primary, fallback = TASK_ENDPOINTS[task]

    try:
        return invoke_endpoint(primary, prompt)
    except Exception as e:
        print(f"Primary {primary} failed: {e}, trying fallback...")
        return invoke_endpoint(fallback, prompt)

# Usage
result = invoke_with_fallback(Task.CODE, "Write quicksort in Python")
```

### Example 3: Complexity-Based Auto-Scaling

**Scenario**: Route simple queries to small model, complex to large model

```python
# complexity_router.py
def estimate_complexity(prompt: str) -> str:
    """Estimate query complexity"""
    tokens = tokenizer.encode(prompt)

    if len(tokens) > 1000:
        return "high"
    elif len(tokens) > 500 or has_code_or_math(prompt):
        return "medium"
    else:
        return "low"

def route_by_complexity(prompt: str) -> str:
    complexity = estimate_complexity(prompt)

    if complexity == "high":
        return "llama-3-70b"
    elif complexity == "medium":
        return "llama-3-8b"
    else:
        return "llama-3-8b-quantized"

# Result: 80% queries → cheap model, 20% → expensive model
# Cost savings: ~60% vs all queries on expensive model
```

## Performance Optimization

### Reducing Cold Starts in MME

```python
# prewarm_mme.py
def prewarm_top_models():
    """Pre-load frequently used models"""
    top_models = [
        "customer-001.tar.gz",  # Most active customer
        "customer-042.tar.gz",
        "customer-099.tar.gz",
    ]

    for model in top_models:
        try:
            runtime.invoke_endpoint(
                EndpointName="customer-models-mme",
                TargetModel=model,
                ContentType="application/json",
                Body=json.dumps({"inputs": "warmup"})
            )
            print(f"✓ Pre-warmed {model}")
        except Exception as e:
            print(f"✗ Failed to pre-warm {model}: {e}")

# Run after deployment and every 30 minutes
```

### Sticky Routing for Cache Hits

```python
# sticky_routing.py
import hashlib

def get_instance_for_customer(customer_id: str, num_instances: int) -> int:
    """Hash customer to specific instance for cache hits"""
    hash_val = int(hashlib.md5(customer_id.encode()).hexdigest(), 16)
    return hash_val % num_instances

# Application load balancer routes customer to same instance
# → Model stays in memory → No cold starts
```

## Monitoring Multi-Model Deployments

### Key Metrics to Track

```python
# monitor_multi_model.py
import boto3

cloudwatch = boto3.client('cloudwatch')

def get_mme_health_metrics(endpoint_name):
    """Get MME-specific health metrics"""
    metrics = {}

    # Model loading time (cold starts)
    metrics['model_load_time'] = get_metric(
        'ModelLoadingWaitTime', endpoint_name
    )

    # Model cache hit rate
    metrics['cache_hit_rate'] = get_metric(
        'ModelCacheHit', endpoint_name
    )

    # Models loaded (memory usage indicator)
    metrics['models_loaded'] = get_metric(
        'LoadedModelCount', endpoint_name
    )

    return metrics

# Alert on high cold start rate
if metrics['model_load_time']['p95'] > 10_000:  # >10 seconds
    send_alert("High cold start latency detected")
```

### Cost Attribution Per Model

```python
# cost_per_model.py
from collections import defaultdict

model_invocations = defaultdict(int)

def log_invocation(model_name):
    """Track invocations per model"""
    model_invocations[model_name] += 1

def calculate_cost_per_model(total_endpoint_cost: float) -> dict:
    """Allocate endpoint cost to models by usage"""
    total_invocations = sum(model_invocations.values())

    costs = {}
    for model, count in model_invocations.items():
        proportion = count / total_invocations
        costs[model] = total_endpoint_cost * proportion

    return costs

# Example
total_cost = 5 * 7.09 * 730  # 5 instances × $7.09/hr × 730 hrs = $25,878
cost_breakdown = calculate_cost_per_model(total_cost)

# Output:
# {
#   "customer-001": $5,200 (20% of usage),
#   "customer-042": $3,900 (15% of usage),
#   ...
# }
```

## Migration Guides

### Migrating from Individual Endpoints to MME

```python
# migration_plan.py

# BEFORE: Individual endpoints
endpoints = {
    "model-a": "ml.g5.xlarge",
    "model-b": "ml.g5.xlarge",
    "model-c": "ml.g5.xlarge",
}

# Cost: 3 × $1.01 × 730 = $2,211/month

# AFTER: Multi-Model Endpoint
mme_config = {
    "models": ["model-a", "model-b", "model-c"],
    "instance_type": "ml.g5.12xlarge",
    "instance_count": 1,
}

# Cost: 1 × $7.09 × 730 = $5,176/month
# Wait, that's more expensive! 🤔

# BETTER: Use smaller instance for MME
mme_config = {
    "models": ["model-a", "model-b", "model-c"],
    "instance_type": "ml.g5.xlarge",  # Same as before
    "instance_count": 1,  # Shared instance
}

# Cost: 1 × $1.01 × 730 = $737/month
# Savings: $1,474/month (67%)
```

**Migration Steps**:
1. Package models for MME (create `.tar.gz` files)
2. Upload to S3
3. Deploy MME endpoint
4. Update application to use `TargetModel` header
5. A/B test (route 10% traffic to MME)
6. Monitor cold start latency
7. Gradually shift 100% traffic
8. Decommission individual endpoints

## Decision Tree

```
Do you have 5+ models?
├─ No → Use individual endpoints
└─ Yes → Continue
    │
    Are most models low-traffic?
    ├─ No → Use individual endpoints
    └─ Yes → Continue
        │
        Is cold start latency acceptable (5-15s)?
        ├─ No → Hybrid (MME + dedicated for hot models)
        └─ Yes → Use Multi-Model Endpoint ✓
```

## Cost-Benefit Analysis Template

```python
# cost_analysis.py

def compare_deployment_options(
    num_models: int,
    requests_per_model_per_day: int,
    model_size_gb: float
):
    """Compare individual endpoints vs MME"""

    # Individual endpoints
    individual_cost = num_models * 1.01 * 730  # ml.g5.xlarge

    # MME (estimate instances needed)
    models_per_instance = int(24 / model_size_gb)  # 24GB GPU
    instances_needed = max(1, num_models // models_per_instance)
    mme_cost = instances_needed * 1.01 * 730

    print(f"Individual Endpoints: ${individual_cost:,.2f}/month")
    print(f"Multi-Model Endpoint: ${mme_cost:,.2f}/month")
    print(f"Savings: ${individual_cost - mme_cost:,.2f}/month ({(individual_cost - mme_cost)/individual_cost*100:.1f}%)")

# Example
compare_deployment_options(
    num_models=20,
    requests_per_model_per_day=100,
    model_size_gb=14,  # 7B model
)
```

## Best Practices

### Multi-Model Endpoints
1. **Right-size instances**: Use larger instances to fit more models in memory
2. **Pre-warm frequently used models**: Reduce cold starts
3. **Monitor cache hit rate**: Aim for >80% cache hits
4. **Use sticky routing**: Route same customer to same instance
5. **Set memory limits**: Prevent OOM with too many models loaded

### Routing Strategies
1. **Start simple**: Task-based routing is easiest
2. **Add complexity gradually**: Complexity → Tier → Load-based
3. **Monitor routing decisions**: Track which models are selected
4. **Implement fallbacks**: Always have backup endpoints
5. **A/B test routing logic**: Validate before full rollout

### Cost Optimization
1. **Profile usage patterns**: Identify low-traffic models
2. **Consolidate where possible**: Use MME for long-tail models
3. **Use quantized models**: 4-bit models reduce memory, enable more models per instance
4. **Auto-scale based on aggregate load**: Not per-model
5. **Use Spot instances for MME**: 50-70% savings for non-critical workloads

## Troubleshooting

### Problem: High cold start latency

**Symptoms**: First request to model takes >15 seconds

**Solutions**:
1. Pre-warm frequently used models
2. Use larger instances (more models fit in memory)
3. Implement sticky routing (cache hits)
4. Consider hybrid approach (dedicated endpoints for hot models)

### Problem: Out of memory errors

**Symptoms**: Endpoint crashes with OOM

**Solutions**:
1. Reduce number of models loaded simultaneously
2. Use larger instance type (more GPU memory)
3. Switch to quantized models (4-bit uses 4x less memory)
4. Decrease `max_model_len` parameter

### Problem: Uneven load distribution

**Symptoms**: Some instances overloaded, others idle

**Solutions**:
1. Implement load-based routing (least connections)
2. Use consistent hashing for sticky routing
3. Enable auto-scaling
4. Monitor per-instance CloudWatch metrics

## Next Steps

1. **Read the guides**:
   - [Multi-Model Endpoints](./SageMaker_Multi_Model_Endpoints.md)
   - [Routing Strategies](./Model_Routing_Strategies.md)

2. **Estimate costs**:
   - Use [Cost Calculator](../CostCalculator/) to compare options

3. **Deploy**:
   - Start with single MME for low-traffic models
   - Add routing logic incrementally

4. **Monitor**:
   - Set up [Monitoring](../Monitoring/) (next section)
   - Track costs, latency, cache hit rate

## Resources

- [SageMaker MME Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)
- [Cost Calculator](../CostCalculator/README.md)
- [Performance Benchmarks](../Benchmarks/README.md)
- [Security Best Practices](../Security/)
