# Distributed Tracing for LLM Applications

## Overview

Implement end-to-end tracing to understand request flow through your LLM application using AWS X-Ray and OpenTelemetry.

## Why Distributed Tracing?

**Challenges without tracing**:
- Requests span multiple services (API Gateway → Lambda → SageMaker → S3)
- Hard to identify bottlenecks
- Difficult to debug failures
- No visibility into component latencies

**With distributed tracing**:
```
Request ID: abc-123
├─ API Gateway: 15ms
├─ Lambda (preprocessing): 45ms
│  ├─ DynamoDB (user context): 12ms
│  └─ S3 (prompt template): 8ms
├─ SageMaker Inference: 850ms ⚠️ Bottleneck!
└─ Lambda (postprocessing): 25ms
Total: 935ms
```

## AWS X-Ray Setup

### 1. Enable X-Ray on SageMaker Endpoint

```python
# deploy_with_xray.py
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.0-tgi2.4.1-gpu-py311-cu121-ubuntu22.04",
    role=role,
    env={
        "AWS_XRAY_DAEMON_ADDRESS": "xray-daemon:2000",  # Enable X-Ray
    }
)

endpoint = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    endpoint_name="llama-3-8b-traced",
)
```

### 2. Instrument Lambda Functions

```python
# lambda_function.py
import boto3
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch all supported libraries (boto3, requests, etc.)
patch_all()

runtime = boto3.client('sagemaker-runtime')

@xray_recorder.capture('invoke_sagemaker')
def invoke_llm(prompt):
    """Traced SageMaker invocation"""
    with xray_recorder.in_subsegment('preprocess') as subsegment:
        # Add metadata
        subsegment.put_metadata('prompt_length', len(prompt))
        processed_prompt = preprocess(prompt)

    with xray_recorder.in_subsegment('sagemaker_invoke'):
        response = runtime.invoke_endpoint(
            EndpointName='llama-3-8b-traced',
            ContentType='application/json',
            Body=json.dumps({"inputs": processed_prompt})
        )

    with xray_recorder.in_subsegment('postprocess'):
        result = postprocess(response['Body'].read())

    return result

def lambda_handler(event, context):
    # X-Ray automatically creates trace for Lambda
    prompt = event['prompt']
    result = invoke_llm(prompt)

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### 3. Trace API Gateway Requests

```python
# api_gateway_tracing.tf (Terraform)
resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.llm_api.id
  name        = "prod"
  auto_deploy = true

  # Enable X-Ray tracing
  default_route_settings {
    detailed_metrics_enabled = true
    throttling_burst_limit   = 5000
    throttling_rate_limit    = 10000
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
      xrayTraceId    = "$context.xrayTraceId"  # Trace ID
    })
  }
}

resource "aws_apigatewayv2_api" "llm_api" {
  name          = "llm-api"
  protocol_type = "HTTP"

  # Enable X-Ray
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET"]
  }

  # X-Ray tracing
  tags = {
    Tracing = "active"
  }
}
```

## OpenTelemetry Integration

### 1. Instrument Application with OpenTelemetry

```python
# app_with_otel.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.boto3sagemaker import Boto3SageMakerInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure exporter (sends to AWS X-Ray via OTLP)
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",  # OTEL collector
    insecure=True
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument libraries
Boto3SageMakerInstrumentor().instrument()
RequestsInstrumentor().instrument()

# Manual instrumentation
@tracer.start_as_current_span("generate_text")
def generate_text(prompt):
    span = trace.get_current_span()

    # Add attributes
    span.set_attribute("llm.model", "llama-3-8b")
    span.set_attribute("llm.prompt_tokens", count_tokens(prompt))

    with tracer.start_as_current_span("invoke_endpoint"):
        response = runtime.invoke_endpoint(
            EndpointName="llama-3-8b",
            ContentType="application/json",
            Body=json.dumps({"inputs": prompt})
        )

    result = json.loads(response['Body'].read())

    # Record output tokens
    span.set_attribute("llm.completion_tokens", count_tokens(result[0]['generated_text']))

    return result

# Usage
text = generate_text("What is machine learning?")
```

### 2. Deploy OpenTelemetry Collector Sidecar

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

exporters:
  awsxray:
    region: us-west-2
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [awsxray, logging]
```

```dockerfile
# Dockerfile (add OTEL collector to container)
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.0-tgi2.4.1-gpu-py311-cu121-ubuntu22.04

# Install OTEL collector
RUN wget https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.91.0/otelcol_0.91.0_linux_amd64.tar.gz && \
    tar -xvf otelcol_0.91.0_linux_amd64.tar.gz && \
    mv otelcol /usr/local/bin/

# Copy config
COPY otel-collector-config.yaml /etc/otel/config.yaml

# Start both OTEL collector and model server
CMD otelcol --config /etc/otel/config.yaml & \
    python -m sagemaker_huggingface_inference_toolkit.serving
```

## Custom Span Attributes for LLMs

### Semantic Conventions

```python
# llm_tracing.py
from opentelemetry import trace

class LLMTracer:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)

    def trace_generation(
        self,
        model: str,
        prompt: str,
        **kwargs
    ):
        """Context manager for tracing LLM generation"""
        span = self.tracer.start_span("llm.generation")

        # Standard attributes
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.request.temperature", kwargs.get("temperature", 1.0))
        span.set_attribute("llm.request.max_tokens", kwargs.get("max_tokens", 256))
        span.set_attribute("llm.request.top_p", kwargs.get("top_p", 1.0))

        # Prompt metadata
        span.set_attribute("llm.prompt.length", len(prompt))
        span.set_attribute("llm.prompt.tokens", count_tokens(prompt))

        return span

    def record_completion(
        self,
        span,
        completion: str,
        latency_ms: float,
        cost: float
    ):
        """Record completion metadata"""
        span.set_attribute("llm.completion.length", len(completion))
        span.set_attribute("llm.completion.tokens", count_tokens(completion))
        span.set_attribute("llm.latency_ms", latency_ms)
        span.set_attribute("llm.cost", cost)

        # Calculate tokens/second
        tokens_per_sec = count_tokens(completion) / (latency_ms / 1000)
        span.set_attribute("llm.tokens_per_second", tokens_per_sec)

        span.end()

# Usage
tracer = LLMTracer()

span = tracer.trace_generation(
    model="llama-3-8b",
    prompt="Explain quantum computing",
    temperature=0.7,
    max_tokens=500
)

start = time.time()
result = generate(prompt)
latency = (time.time() - start) * 1000

tracer.record_completion(
    span=span,
    completion=result,
    latency_ms=latency,
    cost=0.0023
)
```

## Analyzing Traces

### Query Traces with CloudWatch Insights

```sql
-- Find slow requests (>2s)
fields @timestamp, @message, @xrayTraceId
| filter @message like /SageMaker/
| filter duration > 2000
| sort @timestamp desc
| limit 100
```

### X-Ray Service Map Analysis

```python
# analyze_traces.py
import boto3
from datetime import datetime, timedelta

xray = boto3.client('xray')

def get_service_statistics():
    """Get latency statistics per service"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    response = xray.get_service_graph(
        StartTime=start_time,
        EndTime=end_time
    )

    for service in response['Services']:
        stats = service['SummaryStatistics']
        print(f"""
Service: {service['Name']}
  Total Requests: {stats['TotalCount']}
  OK: {stats['OkCount']}
  Errors: {stats['ErrorStatistics']['TotalCount']}
  Fault Rate: {stats['FaultStatistics']['TotalCount'] / stats['TotalCount'] * 100:.2f}%
  Avg Latency: {stats['TotalResponseTime'] / stats['TotalCount']:.0f}ms
        """)

get_service_statistics()
```

### Trace Sampling

```python
# sampling_rules.py
xray = boto3.client('xray')

# Create sampling rule
xray.create_sampling_rule(
    SamplingRule={
        'RuleName': 'LLM-High-Value-Requests',
        'Priority': 100,
        'FixedRate': 0.05,  # Sample 5% of requests
        'ReservoirSize': 10,  # Always sample first 10 req/sec
        'ServiceName': 'llm-api',
        'ServiceType': '*',
        'Host': '*',
        'HTTPMethod': '*',
        'URLPath': '/api/generate',
        'Version': 1,
        'Attributes': {
            'customer_tier': 'premium'  # Sample all premium requests
        }
    }
)

# Sample 100% of errors
xray.create_sampling_rule(
    SamplingRule={
        'RuleName': 'LLM-All-Errors',
        'Priority': 1,  # Highest priority
        'FixedRate': 1.0,  # 100% sampling
        'ReservoirSize': 100,
        'ServiceName': 'llm-api',
        'ServiceType': '*',
        'Host': '*',
        'HTTPMethod': '*',
        'URLPath': '*',
        'Version': 1,
        'Attributes': {
            'error': 'true'
        }
    }
)
```

## Trace-Based Alerts

### Alert on Slow Traces

```python
# trace_based_alarm.py
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='LLM-Slow-Request-Traces',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='Duration',
    Namespace='AWS/XRay',
    Period=300,
    Statistic='p95',
    Threshold=3000,  # 3 seconds
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:xray-alerts'],
    Dimensions=[
        {'Name': 'ServiceName', 'Value': 'sagemaker-endpoint'}
    ]
)
```

### Alert on High Error Rates

```python
# error_rate_alarm.py
cloudwatch.put_metric_alarm(
    AlarmName='LLM-High-Trace-Error-Rate',
    Metrics=[
        {
            'Id': 'faults',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/XRay',
                    'MetricName': 'FaultCount',
                    'Dimensions': [
                        {'Name': 'ServiceName', 'Value': 'sagemaker-endpoint'}
                    ]
                },
                'Period': 300,
                'Stat': 'Sum'
            },
            'ReturnData': False
        },
        {
            'Id': 'total',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/XRay',
                    'MetricName': 'CallCount',
                    'Dimensions': [
                        {'Name': 'ServiceName', 'Value': 'sagemaker-endpoint'}
                    ]
                },
                'Period': 300,
                'Stat': 'Sum'
            },
            'ReturnData': False
        },
        {
            'Id': 'error_rate',
            'Expression': '(faults / total) * 100',
            'Label': 'Error Rate %',
            'ReturnData': True
        }
    ],
    EvaluationPeriods=2,
    ComparisonOperator='GreaterThanThreshold',
    Threshold=5,  # 5% error rate
)
```

## Best Practices

1. **Sample Strategically**: Don't trace 100% of requests (high cost)
   - Sample errors: 100%
   - Sample slow requests: 100%
   - Sample normal requests: 5-10%

2. **Add Context**: Include useful attributes
   - Model name, version
   - Customer/user ID
   - Input/output token counts
   - Cost

3. **Correlate Traces with Logs**: Include trace ID in logs
   ```python
   import logging
   from aws_xray_sdk.core import xray_recorder

   logger = logging.getLogger()

   def log_with_trace(message):
       trace_id = xray_recorder.get_trace_entity().trace_id
       logger.info(f"[Trace: {trace_id}] {message}")
   ```

4. **Monitor Trace Metrics**: Track trace count, latency, error rate

5. **Use Service Map**: Visualize dependencies and bottlenecks

## Cost Optimization

X-Ray Pricing:
- First 100K traces/month: Free
- After: $5 per 1M traces
- Trace storage: $0.50 per 1M traces per month

**Example**:
- 1M requests/day
- 5% sampling rate
- Monthly traces: 1M × 30 × 0.05 = 1.5M traces
- Cost: $5 × 1.5 = **$7.50/month**

## Troubleshooting with Traces

### Example: Debug Slow Request

```python
# analyze_slow_trace.py
trace_id = "1-67891234-abcdef1234567890abcdef12"

response = xray.get_trace_summaries(
    StartTime=start_time,
    EndTime=end_time,
    FilterExpression=f'id("{trace_id}")'
)

trace_summary = response['TraceSummaries'][0]
duration = trace_summary['Duration']

# Get detailed trace
trace_detail = xray.batch_get_traces(
    TraceIds=[trace_id]
)

# Analyze segments
for segment in trace_detail['Traces'][0]['Segments']:
    doc = json.loads(segment['Document'])
    print(f"""
Segment: {doc['name']}
  Duration: {doc.get('end_time', 0) - doc.get('start_time', 0):.3f}s
  Subsegments:
    """)

    for subseg in doc.get('subsegments', []):
        subseg_duration = subseg.get('end_time', 0) - subseg.get('start_time', 0)
        print(f"    - {subseg['name']}: {subseg_duration:.3f}s")
```

Output:
```
Segment: API Gateway
  Duration: 0.015s

Segment: Lambda (preprocessing)
  Duration: 0.045s
  Subsegments:
    - DynamoDB query: 0.012s
    - S3 get: 0.008s

Segment: SageMaker Endpoint
  Duration: 2.850s  ⚠️ BOTTLENECK
  Subsegments:
    - Model inference: 2.820s
    - Serialization: 0.030s

Segment: Lambda (postprocessing)
  Duration: 0.025s
```

## Next Steps

- Configure [CloudWatch Metrics](./CloudWatch_Metrics_and_Alarms.md)
- Set up [Dashboard Templates](./Dashboard_Templates.md)
- Implement [Custom Logging](./Custom_Logging_Strategy.md)

## Resources

- [AWS X-Ray Developer Guide](https://docs.aws.amazon.com/xray/latest/devguide/)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/instrumentation/python/)
- [OTEL Semantic Conventions](https://github.com/open-telemetry/semantic-conventions)
