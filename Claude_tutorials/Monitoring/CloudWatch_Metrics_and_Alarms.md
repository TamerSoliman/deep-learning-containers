# CloudWatch Metrics and Alarms for SageMaker Endpoints

## Overview

Comprehensive monitoring setup for foundation model endpoints using CloudWatch.

## Built-in SageMaker Metrics

### Endpoint-Level Metrics

| Metric | Description | Normal Range | Alert Threshold |
|--------|-------------|--------------|-----------------|
| `Invocations` | Total requests | Varies | N/A |
| `InvocationsPerInstance` | Requests per instance | < 100/min | > 200/min |
| `ModelLatency` | Processing time (ms) | 200-800ms | > 2000ms |
| `OverheadLatency` | SageMaker overhead | 5-20ms | > 100ms |
| `Invocation4XXErrors` | Client errors | 0-1% | > 5% |
| `Invocation5XXErrors` | Server errors | < 0.1% | > 1% |
| `CPUUtilization` | CPU usage | 30-70% | > 90% |
| `MemoryUtilization` | Memory usage | 40-80% | > 95% |
| `GPUUtilization` | GPU usage | 50-90% | < 20% or > 98% |
| `GPUMemoryUtilization` | GPU memory | 60-85% | > 95% |
| `DiskUtilization` | Disk usage | < 50% | > 80% |

## Setup CloudWatch Alarms

### Critical Alarms

#### 1. High Error Rate Alarm

```python
# create_error_alarm.py
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='LLM-Endpoint-High-Error-Rate',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='Invocation5XXErrors',
    Namespace='AWS/SageMaker',
    Period=300,  # 5 minutes
    Statistic='Sum',
    Threshold=10,  # > 10 errors in 5 min
    ActionsEnabled=True,
    AlarmActions=[
        'arn:aws:sns:us-west-2:123456789012:sagemaker-alerts'
    ],
    AlarmDescription='Alert on high 5XX error rate',
    Dimensions=[
        {
            'Name': 'EndpointName',
            'Value': 'llama-3-70b-endpoint'
        },
    ]
)

print("✓ High error rate alarm created")
```

#### 2. High Latency Alarm

```python
# latency_alarm.py
cloudwatch.put_metric_alarm(
    AlarmName='LLM-Endpoint-High-Latency',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=3,
    MetricName='ModelLatency',
    Namespace='AWS/SageMaker',
    Period=300,
    Statistic='Average',  # Could also use ExtendedStatistic='p95'
    Threshold=2000,  # 2 seconds
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:sagemaker-alerts'],
    AlarmDescription='Alert when latency exceeds 2s',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': 'llama-3-70b-endpoint'},
    ]
)
```

#### 3. Low Invocation Rate (Endpoint Not Used)

```python
# low_invocation_alarm.py
cloudwatch.put_metric_alarm(
    AlarmName='LLM-Endpoint-Not-Used',
    ComparisonOperator='LessThanThreshold',
    EvaluationPeriods=6,  # 30 minutes
    MetricName='Invocations',
    Namespace='AWS/SageMaker',
    Period=300,
    Statistic='Sum',
    Threshold=1,  # < 1 request in 30 min
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:cost-alerts'],
    AlarmDescription='Endpoint receiving no traffic (potential waste)',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': 'llama-3-70b-endpoint'},
    ]
)
```

#### 4. GPU Memory Saturation

```python
# gpu_memory_alarm.py
cloudwatch.put_metric_alarm(
    AlarmName='LLM-Endpoint-GPU-Memory-Full',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='GPUMemoryUtilization',
    Namespace='/aws/sagemaker/Endpoints',
    Period=60,
    Statistic='Average',
    Threshold=95,  # 95% GPU memory
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:sagemaker-alerts'],
    AlarmDescription='GPU memory near saturation (OOM risk)',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': 'llama-3-70b-endpoint'},
    ]
)
```

### Warning Alarms

#### 5. Increasing Latency Trend

```python
# latency_anomaly.py
cloudwatch.put_metric_alarm(
    AlarmName='LLM-Endpoint-Latency-Anomaly',
    ComparisonOperator='GreaterThanUpperThreshold',
    EvaluationPeriods=2,
    ThresholdMetricId='ad1',
    Metrics=[
        {
            'Id': 'm1',
            'ReturnData': True,
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/SageMaker',
                    'MetricName': 'ModelLatency',
                    'Dimensions': [
                        {'Name': 'EndpointName', 'Value': 'llama-3-70b-endpoint'}
                    ]
                },
                'Period': 300,
                'Stat': 'Average'
            }
        },
        {
            'Id': 'ad1',
            'Expression': 'ANOMALY_DETECTION_BAND(m1, 2)',  # 2 std dev
            'Label': 'Latency Anomaly'
        }
    ],
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:sagemaker-warnings'],
    AlarmDescription='Latency deviating from normal pattern',
)
```

## Custom Metrics

### Publish Application-Level Metrics

```python
# custom_metrics.py
import boto3
import time
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

class MetricsPublisher:
    def __init__(self, namespace: str = "CustomApp/LLM"):
        self.namespace = namespace

    def publish_generation_metrics(
        self,
        endpoint_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: float
    ):
        """Publish detailed generation metrics"""
        cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    'MetricName': 'InputTokens',
                    'Dimensions': [
                        {'Name': 'EndpointName', 'Value': endpoint_name}
                    ],
                    'Value': input_tokens,
                    'Unit': 'Count',
                    'Timestamp': datetime.now()
                },
                {
                    'MetricName': 'OutputTokens',
                    'Dimensions': [
                        {'Name': 'EndpointName', 'Value': endpoint_name}
                    ],
                    'Value': output_tokens,
                    'Unit': 'Count',
                    'Timestamp': datetime.now()
                },
                {
                    'MetricName': 'TotalTokens',
                    'Dimensions': [
                        {'Name': 'EndpointName', 'Value': endpoint_name}
                    ],
                    'Value': input_tokens + output_tokens,
                    'Unit': 'Count',
                    'Timestamp': datetime.now()
                },
                {
                    'MetricName': 'TokensPerSecond',
                    'Dimensions': [
                        {'Name': 'EndpointName', 'Value': endpoint_name}
                    ],
                    'Value': output_tokens / (latency_ms / 1000),
                    'Unit': 'Count/Second',
                    'Timestamp': datetime.now()
                },
                {
                    'MetricName': 'CostPerRequest',
                    'Dimensions': [
                        {'Name': 'EndpointName', 'Value': endpoint_name}
                    ],
                    'Value': cost,
                    'Unit': 'None',  # Dollars
                    'Timestamp': datetime.now()
                },
            ]
        )

# Usage
metrics = MetricsPublisher()

# After each generation
metrics.publish_generation_metrics(
    endpoint_name="llama-3-70b",
    input_tokens=2048,
    output_tokens=256,
    latency_ms=850,
    cost=0.0023
)
```

### Track Model Quality Metrics

```python
# quality_metrics.py
class QualityMetrics:
    def publish_quality_score(
        self,
        endpoint_name: str,
        task_type: str,
        quality_score: float,  # 0-1
        user_feedback: str = None
    ):
        """Track model quality over time"""
        metrics = [
            {
                'MetricName': 'QualityScore',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'TaskType', 'Value': task_type}
                ],
                'Value': quality_score,
                'Unit': 'None',
                'Timestamp': datetime.now()
            }
        ]

        if user_feedback:
            feedback_value = 1 if user_feedback == 'positive' else 0
            metrics.append({
                'MetricName': 'UserFeedback',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'Feedback', 'Value': user_feedback}
                ],
                'Value': feedback_value,
                'Unit': 'None',
                'Timestamp': datetime.now()
            })

        cloudwatch.put_metric_data(
            Namespace='CustomApp/Quality',
            MetricData=metrics
        )

# Usage
quality = QualityMetrics()

# After evaluation or user feedback
quality.publish_quality_score(
    endpoint_name="llama-3-70b",
    task_type="summarization",
    quality_score=0.89,
    user_feedback="positive"
)
```

## Create CloudWatch Dashboard

```python
# create_dashboard.py
import json

dashboard_body = {
    "widgets": [
        # Row 1: Invocations and Errors
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "Invocations", {"stat": "Sum"}],
                    [".", "Invocation4XXErrors", {"stat": "Sum"}],
                    [".", "Invocation5XXErrors", {"stat": "Sum"}]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "us-west-2",
                "title": "Endpoint Invocations & Errors",
                "yAxis": {"left": {"min": 0}}
            }
        },

        # Row 2: Latency
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency", {"stat": "Average", "label": "Avg"}],
                    ["...", {"stat": "p50", "label": "P50"}],
                    ["...", {"stat": "p95", "label": "P95"}],
                    ["...", {"stat": "p99", "label": "P99"}]
                ],
                "period": 300,
                "region": "us-west-2",
                "title": "Model Latency",
                "yAxis": {"left": {"label": "ms", "min": 0}}
            }
        },

        # Row 3: Resource Utilization
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "GPUUtilization", {"stat": "Average"}],
                    [".", "GPUMemoryUtilization", {"stat": "Average"}],
                    [".", "CPUUtilization", {"stat": "Average"}],
                    [".", "MemoryUtilization", {"stat": "Average"}]
                ],
                "period": 60,
                "region": "us-west-2",
                "title": "Resource Utilization",
                "yAxis": {"left": {"label": "%", "min": 0, "max": 100}}
            }
        },

        # Row 4: Custom Metrics - Tokens
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["CustomApp/LLM", "InputTokens", {"stat": "Sum"}],
                    [".", "OutputTokens", {"stat": "Sum"}],
                    [".", "TokensPerSecond", {"stat": "Average"}]
                ],
                "period": 300,
                "region": "us-west-2",
                "title": "Token Metrics",
                "yAxis": {"left": {"min": 0}}
            }
        },

        # Row 5: Cost Tracking
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["CustomApp/LLM", "CostPerRequest", {"stat": "Average"}],
                ],
                "period": 3600,
                "region": "us-west-2",
                "title": "Cost Per Request",
                "yAxis": {"left": {"label": "$", "min": 0}}
            }
        }
    ]
}

# Create dashboard
cloudwatch.put_dashboard(
    DashboardName='LLM-Endpoint-Monitoring',
    DashboardBody=json.dumps(dashboard_body)
)

print("✓ Dashboard created: LLM-Endpoint-Monitoring")
```

## Metric Math for Advanced Queries

### Calculate Effective Throughput

```python
# throughput_metric.py
cloudwatch.put_metric_alarm(
    AlarmName='LLM-Low-Throughput',
    Metrics=[
        {
            'Id': 'invocations',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/SageMaker',
                    'MetricName': 'Invocations',
                    'Dimensions': [
                        {'Name': 'EndpointName', 'Value': 'llama-3-70b'}
                    ]
                },
                'Period': 60,
                'Stat': 'Sum'
            },
            'ReturnData': False
        },
        {
            'Id': 'throughput',
            'Expression': 'invocations / 60',  # Requests per second
            'Label': 'Throughput (req/sec)',
            'ReturnData': True
        }
    ],
    EvaluationPeriods=5,
    ComparisonOperator='LessThanThreshold',
    Threshold=1,  # < 1 req/sec
    AlarmDescription='Throughput below expected'
)
```

### Calculate Error Rate Percentage

```python
# error_rate.py
cloudwatch.put_metric_alarm(
    AlarmName='LLM-High-Error-Rate-Percentage',
    Metrics=[
        {
            'Id': 'errors',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/SageMaker',
                    'MetricName': 'Invocation5XXErrors',
                },
                'Period': 300,
                'Stat': 'Sum'
            },
            'ReturnData': False
        },
        {
            'Id': 'invocations',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/SageMaker',
                    'MetricName': 'Invocations',
                },
                'Period': 300,
                'Stat': 'Sum'
            },
            'ReturnData': False
        },
        {
            'Id': 'error_rate',
            'Expression': '(errors / invocations) * 100',
            'Label': 'Error Rate %',
            'ReturnData': True
        }
    ],
    EvaluationPeriods=2,
    ComparisonOperator='GreaterThanThreshold',
    Threshold=5,  # > 5% error rate
)
```

### Calculate Cost Per Million Tokens

```python
# cost_per_million_tokens.py
metrics_query = [
    {
        'Id': 'total_cost',
        'MetricStat': {
            'Metric': {
                'Namespace': 'CustomApp/LLM',
                'MetricName': 'CostPerRequest',
            },
            'Period': 3600,
            'Stat': 'Sum'  # Total cost in hour
        },
        'ReturnData': False
    },
    {
        'Id': 'total_tokens',
        'MetricStat': {
            'Metric': {
                'Namespace': 'CustomApp/LLM',
                'MetricName': 'TotalTokens',
            },
            'Period': 3600,
            'Stat': 'Sum'
        },
        'ReturnData': False
    },
    {
        'Id': 'cost_per_mil',
        'Expression': '(total_cost / total_tokens) * 1000000',
        'Label': 'Cost per 1M tokens',
        'ReturnData': True
    }
]
```

## Alerting Best Practices

### 1. Multi-Stage Alerting

```python
# multi_stage_alerts.py

# Stage 1: Warning (Slack)
cloudwatch.put_metric_alarm(
    AlarmName='Latency-Warning',
    Threshold=1500,  # 1.5s
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:slack-warnings'],
)

# Stage 2: Critical (PagerDuty)
cloudwatch.put_metric_alarm(
    AlarmName='Latency-Critical',
    Threshold=3000,  # 3s
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:pagerduty-oncall'],
)
```

### 2. Composite Alarms

```python
# composite_alarm.py
cloudwatch.put_composite_alarm(
    AlarmName='LLM-Endpoint-Unhealthy',
    AlarmRule='(ALARM(High-Latency) OR ALARM(High-Error-Rate)) AND ALARM(Low-Throughput)',
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:pagerduty-oncall'],
    AlarmDescription='Endpoint is unhealthy (multiple symptoms)'
)
```

### 3. Auto-Remediation

```python
# auto_remediation.py

# Create SNS topic
sns = boto3.client('sns')
topic_arn = sns.create_topic(Name='sagemaker-auto-remediation')['TopicArn']

# Lambda function triggered by alarm
lambda_client = boto3.client('lambda')

# Subscribe Lambda to SNS
sns.subscribe(
    TopicArn=topic_arn,
    Protocol='lambda',
    Endpoint='arn:aws:lambda:us-west-2:123456789012:function:RestartEndpoint'
)

# Lambda function code (restart_endpoint.py)
def lambda_handler(event, context):
    """Auto-restart endpoint on high error rate"""
    import boto3

    sagemaker = boto3.client('sagemaker')

    # Parse alarm details
    message = json.loads(event['Records'][0]['Sns']['Message'])
    endpoint_name = message['Trigger']['Dimensions'][0]['value']

    # Update endpoint (triggers restart)
    sagemaker.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_name + '-config',
        RetainAllVariantProperties=True
    )

    return {'statusCode': 200, 'body': f'Restarted {endpoint_name}'}
```

## Monitoring Checklist

- [ ] Enable CloudWatch Container Insights
- [ ] Create alarms for critical metrics (latency, errors, utilization)
- [ ] Set up CloudWatch Dashboard
- [ ] Configure SNS topics for notifications
- [ ] Publish custom metrics (tokens, cost, quality)
- [ ] Set up log aggregation (CloudWatch Logs)
- [ ] Enable anomaly detection for latency trends
- [ ] Create composite alarms for complex conditions
- [ ] Test alarm notifications (send test alerts)
- [ ] Document runbooks for each alarm

## Next Steps

- Set up [Distributed Tracing](./Distributed_Tracing.md)
- Configure [Custom Logging](./Custom_Logging_Strategy.md)
- Review [Dashboard Templates](./Dashboard_Templates.md)

## Resources

- [CloudWatch Metrics Reference](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html)
- [CloudWatch Alarms Best Practices](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Best_Practice_Recommended_Alarms_AWS_Services.html)
