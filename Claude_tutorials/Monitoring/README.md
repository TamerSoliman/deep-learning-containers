# Monitoring & Observability for Foundation Model Deployments

Comprehensive monitoring setup for production LLM endpoints using CloudWatch, X-Ray, and custom metrics.

## Overview

Production monitoring for foundation models requires tracking:
1. **Infrastructure Metrics**: CPU, GPU, memory utilization
2. **Application Metrics**: Latency, throughput, error rates
3. **Business Metrics**: Token usage, cost, quality scores
4. **Distributed Traces**: End-to-end request flow

## Quick Start

### 1. Enable Basic Monitoring (5 minutes)

```python
# quick_setup.py
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create critical alarm
cloudwatch.put_metric_alarm(
    AlarmName='LLM-High-Error-Rate',
    MetricName='Invocation5XXErrors',
    Namespace='AWS/SageMaker',
    Statistic='Sum',
    Period=300,
    EvaluationPeriods=2,
    Threshold=10,
    ComparisonOperator='GreaterThanThreshold',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': 'your-endpoint-name'}
    ],
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:alerts']
)

print("✓ Basic monitoring enabled")
```

### 2. Create Dashboard (10 minutes)

```python
# create_basic_dashboard.py
import json

dashboard = {
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "Invocations", {"stat": "Sum"}],
                    [".", "ModelLatency", {"stat": "Average"}]
                ],
                "period": 300,
                "region": "us-west-2",
                "title": "Endpoint Health"
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName='LLM-Endpoint-Monitor',
    DashboardBody=json.dumps(dashboard)
)
```

### 3. Enable Tracing (15 minutes)

See [Distributed Tracing Guide](./Distributed_Tracing.md)

## Monitoring Layers

### Layer 1: Infrastructure (Always On)

**Metrics**:
- CPU/GPU/Memory utilization
- Disk I/O
- Network throughput

**SLOs**:
- GPU utilization: 60-90% (too low = waste, too high = queue builds up)
- Memory utilization: < 95% (prevent OOM)
- Disk utilization: < 80%

**Tools**: CloudWatch built-in metrics

### Layer 2: Application (Critical)

**Metrics**:
- Request throughput (req/sec)
- Latency (P50, P95, P99)
- Error rates (4XX, 5XX)
- Model loading time (for MME)

**SLOs**:
- P95 latency: < 1000ms (adjust per use case)
- Error rate: < 1%
- Availability: > 99.9%

**Tools**: CloudWatch + Custom Metrics

### Layer 3: Business (Important)

**Metrics**:
- Token usage (input/output)
- Cost per request
- Quality scores (user feedback)
- Customer-specific usage

**SLOs**:
- Cost per 1M tokens: < $X (varies by model)
- Quality score: > 0.85

**Tools**: Custom CloudWatch Metrics

### Layer 4: Traces (Debugging)

**Purpose**: Debug performance issues, understand request flow

**Tools**: AWS X-Ray, OpenTelemetry

## Monitoring Strategies by Deployment Pattern

### Single Endpoint

**Focus**: Latency, utilization, cost

```python
# single_endpoint_monitoring.py
CRITICAL_ALARMS = [
    "High Latency (P95 > 2s)",
    "High Error Rate (> 1%)",
    "Low GPU Utilization (< 30%)",  # Waste of money
]

CUSTOM_METRICS = [
    "Tokens per second",
    "Cost per request",
]
```

### Multi-Model Endpoint (MME)

**Focus**: Model loading time, cache hit rate, memory

```python
# mme_monitoring.py
CRITICAL_ALARMS = [
    "High Model Loading Wait Time (> 10s)",
    "Low Model Cache Hit Rate (< 60%)",
    "Memory Saturation (> 95%)",
]

CUSTOM_METRICS = [
    "Models loaded simultaneously",
    "Cache evictions per minute",
    "Cost attribution per model",
]
```

### Auto-Scaled Deployment

**Focus**: Scaling metrics, queue depth, instance health

```python
# autoscaling_monitoring.py
CRITICAL_ALARMS = [
    "High Invocations Per Instance (> 500/min)",
    "Scaling Too Slow (queue > 100)",
    "Instance Health Check Failed",
]

CUSTOM_METRICS = [
    "Scaling events per hour",
    "Average instance lifetime",
    "Cost per scaling event",
]
```

## Essential Dashboards

### 1. Executive Dashboard (For Leadership)

```
┌─────────────────────────────────────────────────┐
│ LLM Platform - Executive Summary                │
├─────────────────────────────────────────────────┤
│ Total Requests Today: 1.2M           ↑ 15%      │
│ Avg Latency: 580ms                   ↓ 8%       │
│ Error Rate: 0.12%                    ✓ Good     │
│ Daily Cost: $2,450                   → Stable   │
│                                                  │
│ Cost Breakdown:                                  │
│ ███████████████ Llama-3-70B    $1,800 (73%)    │
│ ████ Llama-3-8B                $450 (18%)       │
│ ██ CodeLlama-34B               $200 (9%)        │
└─────────────────────────────────────────────────┘
```

### 2. Operations Dashboard (For SRE)

```
┌─────────────────────────────────────────────────┐
│ LLM Endpoints - Operations                      │
├─────────────────────────────────────────────────┤
│ Endpoint Health:                                 │
│ ✓ llama-3-70b-prod      InService  GPU: 78%    │
│ ✓ llama-3-8b-prod       InService  GPU: 65%    │
│ ⚠ codellama-34b-dev     Updating   GPU: N/A    │
│                                                  │
│ Latency (last hour):                            │
│ P50: ████████ 420ms                             │
│ P95: ███████████████ 850ms                      │
│ P99: ██████████████████ 1,240ms                 │
│                                                  │
│ Active Alarms:                                   │
│ ⚠ 1 Warning: High latency on llama-3-70b       │
└─────────────────────────────────────────────────┘
```

### 3. Developer Dashboard (For ML Engineers)

```
┌─────────────────────────────────────────────────┐
│ Model Performance Metrics                        │
├─────────────────────────────────────────────────┤
│ Throughput:                                      │
│ Requests/sec:  24.5  (target: 20-30)            │
│ Tokens/sec:    3,622 (↑ 12% vs yesterday)       │
│                                                  │
│ Model Quality (last 1000 requests):              │
│ Avg Score: 0.87  ████████▌ (↑ 0.02)            │
│ User Feedback: 👍 92% 👎 8%                     │
│                                                  │
│ Resource Usage:                                  │
│ GPU Memory: 42GB / 80GB (52%)                    │
│ KV Cache Hit Rate: 78%                           │
└─────────────────────────────────────────────────┘
```

## Alerting Strategy

### Alert Severity Levels

| Level | Response Time | Escalation | Examples |
|-------|--------------|------------|----------|
| P0 - Critical | Immediate | PagerDuty | Endpoint down, >10% error rate |
| P1 - High | 15 minutes | Slack + Email | Latency > 3s, Memory > 95% |
| P2 - Medium | 1 hour | Slack | Latency > 2s, Error rate > 1% |
| P3 - Low | Next day | Email | Inefficient usage, cost anomalies |

### Alert Frequency Management

```python
# prevent_alarm_fatigue.py
class AlertManager:
    def __init__(self):
        self.alert_history = {}
        self.cooldown_period = 300  # 5 minutes

    def should_send_alert(self, alarm_name: str) -> bool:
        """Prevent duplicate alerts within cooldown period"""
        last_sent = self.alert_history.get(alarm_name, 0)
        current_time = time.time()

        if current_time - last_sent > self.cooldown_period:
            self.alert_history[alarm_name] = current_time
            return True

        return False

# Usage
manager = AlertManager()

if metric_exceeds_threshold and manager.should_send_alert("HighLatency"):
    send_alert("High latency detected")
```

## Cost Monitoring

### Track Endpoint Costs

```python
# cost_tracking.py
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce')  # Cost Explorer

def get_endpoint_cost(endpoint_name: str, days: int = 7):
    """Get total cost for endpoint over last N days"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date.isoformat(),
            'End': end_date.isoformat()
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        Filter={
            'And': [
                {'Dimensions': {'Key': 'SERVICE', 'Values': ['Amazon SageMaker']}},
                {'Tags': {'Key': 'EndpointName', 'Values': [endpoint_name]}}
            ]
        }
    )

    total_cost = sum(
        float(day['Total']['UnblendedCost']['Amount'])
        for day in response['ResultsByTime']
    )

    return total_cost

# Usage
cost = get_endpoint_cost("llama-3-70b-prod", days=30)
print(f"Monthly cost: ${cost:.2f}")

# Alert if cost exceeds budget
if cost > 10000:  # $10K budget
    send_alert(f"Endpoint cost ${cost:.2f} exceeds budget")
```

### Cost Anomaly Detection

```python
# cost_anomaly.py
ce = boto3.client('ce')

# Create cost anomaly monitor
ce.create_anomaly_monitor(
    AnomalyMonitor={
        'MonitorName': 'LLM-Endpoint-Cost-Monitor',
        'MonitorType': 'CUSTOM',
        'MonitorSpecification': {
            'And': [
                {'Dimensions': {'Key': 'SERVICE', 'Values': ['Amazon SageMaker']}},
                {'Tags': {'Key': 'Project', 'Values': ['LLM-Platform']}}
            ]
        }
    }
)

# Create anomaly subscription
ce.create_anomaly_subscription(
    AnomalySubscription={
        'SubscriptionName': 'LLM-Cost-Anomaly-Alerts',
        'MonitorArnList': ['arn:aws:ce::123456789012:anomalymonitor/...'],
        'Subscribers': [
            {
                'Type': 'SNS',
                'Address': 'arn:aws:sns:us-west-2:123456789012:cost-alerts'
            }
        ],
        'Threshold': 500,  # Alert on $500+ anomaly
        'Frequency': 'IMMEDIATE'
    }
)
```

## Quality Monitoring

### Track Model Output Quality

```python
# quality_monitoring.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class QualityMetric:
    timestamp: datetime
    model: str
    task: str
    score: float  # 0-1
    feedback: Optional[str] = None

class QualityMonitor:
    def __init__(self):
        self.metrics = []

    def record_generation(
        self,
        model: str,
        task: str,
        prompt: str,
        completion: str,
        user_feedback: Optional[str] = None
    ):
        """Record quality metric for generation"""
        # Calculate automated score (e.g., BLEU, ROUGE, perplexity)
        score = self.calculate_score(prompt, completion, task)

        metric = QualityMetric(
            timestamp=datetime.now(),
            model=model,
            task=task,
            score=score,
            feedback=user_feedback
        )

        self.metrics.append(metric)

        # Publish to CloudWatch
        cloudwatch.put_metric_data(
            Namespace='CustomApp/Quality',
            MetricData=[
                {
                    'MetricName': 'QualityScore',
                    'Dimensions': [
                        {'Name': 'Model', 'Value': model},
                        {'Name': 'Task', 'Value': task}
                    ],
                    'Value': score,
                    'Timestamp': metric.timestamp
                }
            ]
        )

        # Alert on quality degradation
        avg_score = self.get_avg_score_last_hour(model, task)
        if avg_score < 0.75:  # Below threshold
            send_alert(f"Quality degradation detected: {model} ({task})")

    def calculate_score(self, prompt, completion, task):
        """Calculate quality score (simplified)"""
        # In production, use actual metrics:
        # - BLEU/ROUGE for summarization
        # - Exact match for QA
        # - Syntax validation for code
        # - Human feedback
        return 0.85  # Placeholder

# Usage
monitor = QualityMonitor()

monitor.record_generation(
    model="llama-3-70b",
    task="summarization",
    prompt=article,
    completion=summary,
    user_feedback="positive"
)
```

## Monitoring Checklist

### Initial Setup
- [ ] Enable CloudWatch Container Insights
- [ ] Create SNS topics for alerts
- [ ] Set up CloudWatch Dashboard
- [ ] Configure IAM roles for metrics publishing
- [ ] Enable X-Ray tracing
- [ ] Tag all resources appropriately

### Critical Alarms
- [ ] High error rate (> 1%)
- [ ] High latency (P95 > 2s)
- [ ] Low endpoint availability
- [ ] GPU memory saturation (> 95%)
- [ ] Endpoint not receiving traffic

### Custom Metrics
- [ ] Token usage (input/output)
- [ ] Cost per request
- [ ] Throughput (tokens/sec)
- [ ] Quality scores
- [ ] User feedback

### Dashboards
- [ ] Executive dashboard (business metrics)
- [ ] Operations dashboard (health, alarms)
- [ ] Developer dashboard (performance, quality)
- [ ] Cost dashboard (spend, attribution)

### Processes
- [ ] On-call rotation defined
- [ ] Runbooks for common issues
- [ ] Escalation paths documented
- [ ] Post-incident review process

## Troubleshooting Playbook

### High Latency

**Symptoms**: P95 latency > 2s

**Investigation**:
1. Check CloudWatch GPU utilization (if >95%, endpoint overloaded)
2. Review X-Ray traces to find bottleneck
3. Check model loading time (for MME)
4. Verify instance type is appropriate for model size

**Solutions**:
- Scale up (more instances) if GPU saturated
- Scale up (larger instance) if model doesn't fit in memory
- Enable auto-scaling
- Pre-warm models (for MME)

### High Error Rate

**Symptoms**: 5XX errors > 1%

**Investigation**:
1. Check CloudWatch Logs for error messages
2. Review X-Ray traces for failed requests
3. Check memory utilization (OOM errors)
4. Verify model artifacts are accessible (S3)

**Solutions**:
- Restart endpoint if transient issue
- Increase memory if OOM errors
- Fix model artifacts if S3 access errors
- Check IAM permissions

### Cost Spike

**Symptoms**: Daily cost 2x higher than baseline

**Investigation**:
1. Check request volume (legitimate spike?)
2. Review instance count (auto-scaling gone wild?)
3. Check for new endpoints deployed
4. Look for inefficient usage patterns

**Solutions**:
- Adjust auto-scaling policies
- Shut down unused endpoints
- Move low-traffic models to MME
- Enable Spot instances

## Best Practices

1. **Start with Basics**: Don't over-engineer monitoring initially
2. **Alert on Symptoms, Not Causes**: Alert on high latency, not high CPU
3. **Reduce Noise**: Tune thresholds to avoid false positives
4. **Automate Remediation**: Auto-scale, auto-restart where possible
5. **Monitor Cost**: Set budgets and anomaly detection
6. **Track Quality**: Don't just monitor infrastructure
7. **Use Traces for Debugging**: Enable for complex issues
8. **Dashboard for Each Audience**: Executive, Ops, Dev dashboards
9. **Document Runbooks**: What to do when alarm fires
10. **Review Metrics**: Weekly review of trends and anomalies

## Advanced Topics

### Multi-Region Monitoring

```python
# multi_region_monitoring.py
regions = ['us-west-2', 'us-east-1', 'eu-west-1']

for region in regions:
    cloudwatch = boto3.client('cloudwatch', region_name=region)

    # Create cross-region dashboard
    cloudwatch.put_dashboard(
        DashboardName='Global-LLM-Endpoints',
        DashboardBody=json.dumps({
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invocations",
                             {"region": r, "stat": "Sum"}]
                            for r in regions
                        ],
                        "title": "Global Request Volume"
                    }
                }
            ]
        })
    )
```

### Synthetic Monitoring

```python
# synthetic_monitoring.py
import boto3
import json
import time

def synthetic_check():
    """Proactive health check"""
    runtime = boto3.client('sagemaker-runtime')

    try:
        start = time.time()
        response = runtime.invoke_endpoint(
            EndpointName='llama-3-8b-prod',
            ContentType='application/json',
            Body=json.dumps({"inputs": "Health check"})
        )
        latency = time.time() - start

        # Publish synthetic metric
        cloudwatch.put_metric_data(
            Namespace='Synthetic/LLM',
            MetricData=[
                {
                    'MetricName': 'SyntheticCheckLatency',
                    'Value': latency * 1000,
                    'Unit': 'Milliseconds'
                },
                {
                    'MetricName': 'SyntheticCheckSuccess',
                    'Value': 1
                }
            ]
        )

    except Exception as e:
        # Alert on failure
        cloudwatch.put_metric_data(
            Namespace='Synthetic/LLM',
            MetricData=[
                {'MetricName': 'SyntheticCheckSuccess', 'Value': 0}
            ]
        )
        send_alert(f"Synthetic check failed: {e}")

# Run every 5 minutes via Lambda
```

## Next Steps

1. **Read Detailed Guides**:
   - [CloudWatch Metrics & Alarms](./CloudWatch_Metrics_and_Alarms.md)
   - [Distributed Tracing](./Distributed_Tracing.md)

2. **Deploy Monitoring**:
   - Start with critical alarms
   - Create basic dashboard
   - Enable tracing for debugging

3. **Iterate**:
   - Add custom metrics as needed
   - Tune alert thresholds
   - Expand dashboards

## Resources

- [SageMaker Monitoring Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-overview.html)
- [CloudWatch Best Practices](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Best_Practice_Recommended_Alarms_AWS_Services.html)
- [X-Ray Developer Guide](https://docs.aws.amazon.com/xray/latest/devguide/)
- [Cost Optimization](../CostCalculator/README.md)
