# Model Routing and Load Balancing Strategies

## Overview

Intelligent routing strategies to optimize cost, latency, and quality when deploying multiple foundation models.

## Routing Strategies

### 1. Task-Based Routing

Route requests to specialized models based on task type.

```python
# task_based_router.py
from enum import Enum
from dataclasses import dataclass

class TaskType(Enum):
    CODE_GENERATION = "code"
    SUMMARIZATION = "summary"
    QUESTION_ANSWERING = "qa"
    TRANSLATION = "translation"
    GENERAL_CHAT = "chat"

@dataclass
class ModelConfig:
    endpoint_name: str
    cost_per_1k_tokens: float
    avg_latency_ms: float
    quality_score: float  # 0-1

# Model registry
TASK_MODELS = {
    TaskType.CODE_GENERATION: [
        ModelConfig("codellama-34b", 0.60, 800, 0.95),
        ModelConfig("codellama-13b", 0.25, 400, 0.88),
        ModelConfig("codellama-7b", 0.15, 250, 0.82),
    ],
    TaskType.SUMMARIZATION: [
        ModelConfig("llama-3-70b", 0.90, 650, 0.94),
        ModelConfig("llama-3-8b", 0.20, 300, 0.87),
        ModelConfig("bart-large-cnn", 0.05, 150, 0.80),
    ],
    TaskType.GENERAL_CHAT: [
        ModelConfig("llama-3-70b-instruct", 0.90, 600, 0.96),
        ModelConfig("mistral-8x7b", 0.65, 550, 0.93),
        ModelConfig("llama-3-8b-instruct", 0.20, 300, 0.89),
    ],
}

def route_by_task(task: TaskType, priority: str = "balanced") -> ModelConfig:
    """
    Route to best model for task based on priority

    priority:
        - "quality": Choose highest quality model
        - "cost": Choose cheapest model
        - "latency": Choose fastest model
        - "balanced": Balance all factors
    """
    models = TASK_MODELS[task]

    if priority == "quality":
        return max(models, key=lambda m: m.quality_score)
    elif priority == "cost":
        return min(models, key=lambda m: m.cost_per_1k_tokens)
    elif priority == "latency":
        return min(models, key=lambda m: m.avg_latency_ms)
    elif priority == "balanced":
        # Score: 0.4*quality + 0.3*cost + 0.3*latency (normalized)
        def balanced_score(m):
            norm_quality = m.quality_score
            norm_cost = 1 - (m.cost_per_1k_tokens / 1.0)  # Inverse
            norm_latency = 1 - (m.avg_latency_ms / 1000)  # Inverse
            return 0.4*norm_quality + 0.3*norm_cost + 0.3*norm_latency

        return max(models, key=balanced_score)

# Usage
model = route_by_task(TaskType.CODE_GENERATION, priority="quality")
print(f"Selected: {model.endpoint_name}")

model = route_by_task(TaskType.CODE_GENERATION, priority="cost")
print(f"Selected: {model.endpoint_name}")
```

### 2. Complexity-Based Routing

Route simple queries to small models, complex queries to large models.

```python
# complexity_router.py
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

def estimate_complexity(prompt: str) -> str:
    """Estimate query complexity"""
    # Token count
    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)

    # Heuristics
    has_code = bool(re.search(r'```|def |class |import ', prompt))
    has_math = bool(re.search(r'\d+\s*[\+\-\*/]\s*\d+|\^|sqrt|integral', prompt))
    is_multi_step = prompt.lower().count("then") > 1 or prompt.lower().count("step") > 1

    # Classify
    if token_count > 1000 or is_multi_step:
        return "high"
    elif has_code or has_math or token_count > 500:
        return "medium"
    else:
        return "low"

def route_by_complexity(prompt: str) -> str:
    """Route to appropriate model based on complexity"""
    complexity = estimate_complexity(prompt)

    if complexity == "high":
        return "llama-3-70b-endpoint"  # Most capable
    elif complexity == "medium":
        return "llama-3-8b-endpoint"  # Balanced
    else:
        return "llama-3-8b-quantized-endpoint"  # Fast & cheap

# Usage
prompt1 = "What is the capital of France?"
prompt2 = "Explain quantum entanglement and write code to simulate it"
prompt3 = """
First, analyze the market trends.
Then, create a financial model.
Next, generate forecasts for 5 years.
Finally, summarize the findings.
"""

print(route_by_complexity(prompt1))  # → llama-3-8b-quantized
print(route_by_complexity(prompt2))  # → llama-3-8b
print(route_by_complexity(prompt3))  # → llama-3-70b
```

### 3. Latency-SLA Routing

Route based on user's latency requirements.

```python
# sla_router.py
from dataclasses import dataclass
from typing import List

@dataclass
class EndpointSLA:
    name: str
    p95_latency_ms: float
    cost_per_request: float

# Endpoint inventory sorted by latency
ENDPOINTS = [
    EndpointSLA("llama-3-8b-inf2", 350, 0.002),  # Fastest, Inferentia2
    EndpointSLA("llama-3-8b-gpu-fp8", 420, 0.003),  # Fast, FP8 quantized
    EndpointSLA("llama-3-8b-gpu", 500, 0.004),  # Standard GPU
    EndpointSLA("llama-3-70b-gpu-tp8", 650, 0.015),  # Large model
]

def route_by_sla(max_latency_ms: float, prioritize_cost: bool = True) -> EndpointSLA:
    """
    Select endpoint meeting latency SLA

    Args:
        max_latency_ms: Maximum acceptable P95 latency
        prioritize_cost: If True, choose cheapest option within SLA
    """
    # Filter endpoints meeting SLA
    candidates = [ep for ep in ENDPOINTS if ep.p95_latency_ms <= max_latency_ms]

    if not candidates:
        raise ValueError(f"No endpoints meet {max_latency_ms}ms SLA")

    if prioritize_cost:
        return min(candidates, key=lambda ep: ep.cost_per_request)
    else:
        return min(candidates, key=lambda ep: ep.p95_latency_ms)

# Usage: Premium tier (strict SLA, don't care about cost)
premium_endpoint = route_by_sla(max_latency_ms=400, prioritize_cost=False)
print(f"Premium: {premium_endpoint.name}")  # → llama-3-8b-inf2

# Usage: Standard tier (relaxed SLA, optimize cost)
standard_endpoint = route_by_sla(max_latency_ms=600, prioritize_cost=True)
print(f"Standard: {standard_endpoint.name}")  # → llama-3-8b-inf2 (cheapest within SLA)
```

### 4. Tier-Based Routing

Different models for different customer tiers.

```python
# tier_router.py
from enum import Enum

class CustomerTier(Enum):
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

TIER_CONFIG = {
    CustomerTier.FREE: {
        "endpoint": "llama-3-8b-quantized",
        "max_requests_per_day": 100,
        "max_tokens_per_request": 512,
        "rate_limit_per_minute": 5,
    },
    CustomerTier.STANDARD: {
        "endpoint": "llama-3-8b",
        "max_requests_per_day": 10000,
        "max_tokens_per_request": 2048,
        "rate_limit_per_minute": 60,
    },
    CustomerTier.PREMIUM: {
        "endpoint": "llama-3-70b",
        "max_requests_per_day": 100000,
        "max_tokens_per_request": 4096,
        "rate_limit_per_minute": 600,
    },
    CustomerTier.ENTERPRISE: {
        "endpoint": "dedicated-llama-3-405b",  # Dedicated endpoint
        "max_requests_per_day": None,  # Unlimited
        "max_tokens_per_request": 8192,
        "rate_limit_per_minute": None,  # Unlimited
    },
}

def get_endpoint_for_tier(tier: CustomerTier) -> str:
    """Get appropriate endpoint for customer tier"""
    return TIER_CONFIG[tier]["endpoint"]

def check_rate_limit(tier: CustomerTier, requests_in_window: int) -> bool:
    """Check if customer is within rate limits"""
    limit = TIER_CONFIG[tier]["rate_limit_per_minute"]
    if limit is None:
        return True  # Unlimited
    return requests_in_window < limit

# Usage in API
@app.route("/api/generate", methods=["POST"])
def generate():
    # Get customer tier from database
    customer_id = request.headers["X-Customer-ID"]
    tier = get_customer_tier(customer_id)

    # Check rate limit
    requests_in_window = get_request_count(customer_id, window=60)
    if not check_rate_limit(tier, requests_in_window):
        return jsonify({"error": "Rate limit exceeded"}), 429

    # Route to appropriate endpoint
    endpoint_name = get_endpoint_for_tier(tier)

    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(request.json)
    )

    return jsonify(json.loads(response['Body'].read()))
```

### 5. Load-Based Routing

Route to least-loaded endpoint for better throughput.

```python
# load_balancer.py
import boto3
from datetime import datetime, timedelta
from collections import defaultdict

cloudwatch = boto3.client('cloudwatch')

class LoadBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.request_counts = defaultdict(int)

    def get_endpoint_load(self, endpoint_name: str) -> float:
        """Get current load (invocations per minute)"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=1)

        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName='Invocations',
            Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Sum']
        )

        if response['Datapoints']:
            return response['Datapoints'][0]['Sum']
        return 0

    def select_endpoint(self) -> str:
        """Select least-loaded endpoint"""
        loads = {ep: self.get_endpoint_load(ep) for ep in self.endpoints}
        return min(loads, key=loads.get)

# Usage
lb = LoadBalancer([
    "llama-3-8b-endpoint-1",
    "llama-3-8b-endpoint-2",
    "llama-3-8b-endpoint-3",
])

endpoint = lb.select_endpoint()
print(f"Routing to: {endpoint}")
```

### 6. Fallback Routing

Primary endpoint with fallback to backup.

```python
# fallback_router.py
import time
from typing import Optional

class FallbackRouter:
    def __init__(self, primary: str, fallbacks: List[str]):
        self.primary = primary
        self.fallbacks = fallbacks
        self.failure_count = defaultdict(int)
        self.last_failure = defaultdict(float)

    def should_skip_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint is in circuit breaker state"""
        failures = self.failure_count[endpoint]
        last_fail_time = self.last_failure[endpoint]

        # Circuit breaker: skip if >3 failures in last 60 seconds
        if failures >= 3:
            if time.time() - last_fail_time < 60:
                return True  # Still in cooldown
            else:
                # Reset after cooldown
                self.failure_count[endpoint] = 0
                return False

        return False

    def invoke_with_fallback(self, payload: dict) -> dict:
        """Invoke with automatic fallback"""
        endpoints_to_try = [self.primary] + self.fallbacks

        for endpoint in endpoints_to_try:
            if self.should_skip_endpoint(endpoint):
                continue

            try:
                response = runtime.invoke_endpoint(
                    EndpointName=endpoint,
                    ContentType="application/json",
                    Body=json.dumps(payload)
                )

                # Success - reset failure count
                self.failure_count[endpoint] = 0
                return json.loads(response['Body'].read())

            except Exception as e:
                # Record failure
                self.failure_count[endpoint] += 1
                self.last_failure[endpoint] = time.time()

                print(f"✗ {endpoint} failed: {e}")
                continue  # Try next endpoint

        raise Exception("All endpoints failed")

# Usage
router = FallbackRouter(
    primary="llama-3-70b-primary",
    fallbacks=["llama-3-70b-backup", "llama-3-8b-emergency"]
)

try:
    result = router.invoke_with_fallback({"inputs": "Hello"})
except Exception as e:
    # All endpoints down
    return {"error": "Service temporarily unavailable"}, 503
```

## Load Balancing Patterns

### Pattern 1: Round-Robin

Simple rotation through endpoints.

```python
# round_robin.py
class RoundRobinBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.current_index = 0

    def next_endpoint(self) -> str:
        """Get next endpoint in rotation"""
        endpoint = self.endpoints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.endpoints)
        return endpoint

# Usage
lb = RoundRobinBalancer([
    "llama-3-8b-1",
    "llama-3-8b-2",
    "llama-3-8b-3",
])

for i in range(10):
    print(lb.next_endpoint())  # Cycles through 1 → 2 → 3 → 1 → ...
```

### Pattern 2: Weighted Round-Robin

Distribute based on endpoint capacity.

```python
# weighted_round_robin.py
from collections import Counter

class WeightedRoundRobin:
    def __init__(self, endpoints: dict):
        """
        endpoints: {name: weight}
        e.g., {"ep-1": 3, "ep-2": 2, "ep-3": 1}
        """
        self.endpoints = []
        for name, weight in endpoints.items():
            self.endpoints.extend([name] * weight)

        self.current_index = 0

    def next_endpoint(self) -> str:
        endpoint = self.endpoints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.endpoints)
        return endpoint

# Usage: ep-1 is 3x more powerful than ep-3
lb = WeightedRoundRobin({
    "llama-3-70b-p4d": 3,  # Powerful instance
    "llama-3-8b-g5": 2,    # Medium instance
    "llama-3-8b-inf2": 1,  # Light instance
})

# Distribution: 50% to p4d, 33% to g5, 17% to inf2
requests = [lb.next_endpoint() for _ in range(100)]
print(Counter(requests))
```

### Pattern 3: Least Connections

Route to endpoint with fewest active requests.

```python
# least_connections.py
from threading import Lock

class LeastConnectionsBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.active_connections = {ep: 0 for ep in endpoints}
        self.lock = Lock()

    def get_endpoint(self) -> str:
        """Get endpoint with least active connections"""
        with self.lock:
            return min(self.active_connections, key=self.active_connections.get)

    def acquire(self, endpoint: str):
        """Mark connection as active"""
        with self.lock:
            self.active_connections[endpoint] += 1

    def release(self, endpoint: str):
        """Mark connection as complete"""
        with self.lock:
            self.active_connections[endpoint] -= 1

# Usage
lb = LeastConnectionsBalancer(["ep-1", "ep-2", "ep-3"])

def make_request(prompt):
    endpoint = lb.get_endpoint()
    lb.acquire(endpoint)

    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=json.dumps({"inputs": prompt})
        )
        return json.loads(response['Body'].read())
    finally:
        lb.release(endpoint)
```

### Pattern 4: Consistent Hashing

Sticky routing - same input always goes to same endpoint.

```python
# consistent_hashing.py
import hashlib

class ConsistentHashRouter:
    def __init__(self, endpoints: List[str]):
        self.endpoints = sorted(endpoints)  # Consistent ordering

    def get_endpoint(self, key: str) -> str:
        """Route based on hash of key"""
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        index = hash_val % len(self.endpoints)
        return self.endpoints[index]

# Usage: Same user always routed to same endpoint
router = ConsistentHashRouter(["ep-1", "ep-2", "ep-3"])

# Customer-specific routing
customer_id = "customer-12345"
endpoint = router.get_endpoint(customer_id)

# Benefit: Model cache hit (same customer = same endpoint)
```

## Advanced Routing Logic

### Multi-Factor Routing Decision

Combine multiple factors for optimal routing.

```python
# multi_factor_router.py
from dataclasses import dataclass
from typing import List

@dataclass
class RoutingContext:
    task: str
    customer_tier: str
    max_latency_ms: float
    prioritize_cost: bool
    user_id: str

@dataclass
class Endpoint:
    name: str
    tasks: List[str]
    min_tier: str
    latency_p95_ms: float
    cost_per_request: float

TIER_RANK = {"free": 0, "standard": 1, "premium": 2, "enterprise": 3}

class SmartRouter:
    def __init__(self, endpoints: List[Endpoint]):
        self.endpoints = endpoints

    def route(self, context: RoutingContext) -> Endpoint:
        """Multi-factor routing decision"""
        # Filter: Task compatibility
        candidates = [ep for ep in self.endpoints if context.task in ep.tasks]

        # Filter: Tier access
        user_tier_rank = TIER_RANK[context.customer_tier]
        candidates = [ep for ep in candidates
                     if TIER_RANK[ep.min_tier] <= user_tier_rank]

        # Filter: Latency SLA
        candidates = [ep for ep in candidates
                     if ep.latency_p95_ms <= context.max_latency_ms]

        if not candidates:
            raise ValueError("No endpoint matches criteria")

        # Sort by priority
        if context.prioritize_cost:
            return min(candidates, key=lambda ep: ep.cost_per_request)
        else:
            return min(candidates, key=lambda ep: ep.latency_p95_ms)

# Setup
endpoints = [
    Endpoint("codellama-7b", ["code"], "free", 300, 0.001),
    Endpoint("codellama-34b", ["code"], "premium", 700, 0.005),
    Endpoint("llama-3-8b", ["chat", "qa"], "free", 350, 0.002),
    Endpoint("llama-3-70b", ["chat", "qa", "code"], "premium", 600, 0.008),
]

router = SmartRouter(endpoints)

# Route: Free tier user, code task, strict latency
context = RoutingContext(
    task="code",
    customer_tier="free",
    max_latency_ms=400,
    prioritize_cost=True,
    user_id="user-123"
)

endpoint = router.route(context)
print(f"Selected: {endpoint.name}")  # → codellama-7b
```

### Canary Deployment Router

Gradually shift traffic to new model version.

```python
# canary_router.py
import random

class CanaryRouter:
    def __init__(self, stable_endpoint: str, canary_endpoint: str):
        self.stable = stable_endpoint
        self.canary = canary_endpoint
        self.canary_percentage = 0.0  # Start at 0%

    def set_canary_traffic(self, percentage: float):
        """Set canary traffic percentage (0.0 to 1.0)"""
        self.canary_percentage = max(0.0, min(1.0, percentage))

    def get_endpoint(self) -> str:
        """Select endpoint based on canary percentage"""
        if random.random() < self.canary_percentage:
            return self.canary
        return self.stable

# Deployment strategy
router = CanaryRouter("llama-3-70b-v1", "llama-3-70b-v2")

# Day 1: 5% canary
router.set_canary_traffic(0.05)

# Day 2: 25% canary (if metrics good)
router.set_canary_traffic(0.25)

# Day 3: 50% canary
router.set_canary_traffic(0.50)

# Day 4: 100% canary (promote to stable)
router.set_canary_traffic(1.00)
```

## Monitoring Routing Decisions

```python
# routing_metrics.py
import boto3
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

def log_routing_decision(endpoint: str, latency_ms: float, cost: float, reason: str):
    """Log routing decision for analysis"""
    cloudwatch.put_metric_data(
        Namespace='CustomApp/Routing',
        MetricData=[
            {
                'MetricName': 'EndpointSelection',
                'Dimensions': [
                    {'Name': 'Endpoint', 'Value': endpoint},
                    {'Name': 'Reason', 'Value': reason}
                ],
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.now()
            },
            {
                'MetricName': 'RoutingLatency',
                'Dimensions': [{'Name': 'Endpoint', 'Value': endpoint}],
                'Value': latency_ms,
                'Unit': 'Milliseconds',
                'Timestamp': datetime.now()
            }
        ]
    )

# Usage
endpoint = route_by_complexity(prompt)
log_routing_decision(endpoint, latency_ms=450, cost=0.003, reason="complexity-based")
```

## Best Practices

1. **Start Simple**: Begin with task-based routing, add complexity as needed
2. **Monitor Metrics**: Track which routing decisions are made and their outcomes
3. **A/B Test**: Compare routing strategies to find what works best
4. **Fallback Plans**: Always have backup endpoints
5. **Cache Routing Decisions**: For same inputs, cache endpoint selection
6. **Gradual Rollouts**: Use canary deployments for new routing logic
7. **Cost Tracking**: Log costs per routing decision to optimize

## Next Steps

- Implement [Multi-Model Endpoints](./SageMaker_Multi_Model_Endpoints.md) for cost savings
- Set up [Monitoring](../Monitoring/) to track routing effectiveness
- Use [Cost Calculator](../CostCalculator/) to evaluate routing strategies
