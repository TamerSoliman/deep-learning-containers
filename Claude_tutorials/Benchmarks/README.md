# Performance Benchmark Suite

## Quick Start

```bash
# Run benchmark
python scripts/benchmark_runner.py \
  --endpoint vllm-llama3-70b \
  --container vllm \
  --model meta-llama/Llama-3-70b-hf \
  --instance-type ml.p4d.24xlarge \
  --num-requests 100

# Compare multiple containers
python scripts/compare_benchmarks.py \
  --results results/vllm_results.json \
  --results results/sglang_results.json \
  --results results/neuron_results.json
```

## Benchmark Metrics

- **Throughput**: Requests/sec, Tokens/sec
- **Latency**: P50, P95, P99 (milliseconds)
- **Cost**: $/hour, $/million tokens
- **Memory**: GPU/Neuron memory usage

## Sample Results

### Llama 3 70B Inference

| Container | Instance | Throughput | P50 Latency | Cost/1M tokens |
|-----------|----------|------------|-------------|----------------|
| **vLLM** | ml.p4d.24xlarge | 50 req/sec | 400ms | $1.64 |
| **SGLang** | ml.p4d.24xlarge | 45 req/sec | 450ms | $1.82 |
| **NeuronX** | ml.inf2.48xlarge | 30 req/sec | 600ms | **$0.41** ⭐ |
| **vLLM ARM64** | ml.g5g.16xlarge | 16 req/sec | 550ms | $0.72 |

## Usage Guide

See `scripts/benchmark_runner.py --help` for full options.
