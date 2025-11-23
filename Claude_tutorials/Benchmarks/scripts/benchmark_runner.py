#!/usr/bin/env python3
"""
Automated Performance Benchmark Suite for DLC Containers

Benchmarks:
- Throughput (requests/sec, tokens/sec)
- Latency (P50, P95, P99)
- Cost per million tokens
- Memory usage

Supported containers:
- vLLM (GPU, ARM64)
- SGLang
- HuggingFace TGI
- Neuron (Inferentia2)
"""

import boto3
import json
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    endpoint_name: str
    container_type: str  # vllm, sglang, tgi, neuron
    instance_type: str
    instance_count: int
    model_name: str
    input_tokens: int = 2048
    output_tokens: int = 100
    num_requests: int = 100
    concurrent_requests: int = 10


@dataclass
class BenchmarkResults:
    """Benchmark results"""
    container_type: str
    model_name: str
    instance_type: str
    throughput_rps: float  # Requests per second
    throughput_tps: float  # Tokens per second
    latency_p50: float
    latency_p95: float
    latency_p99: float
    cost_per_hour: float
    cost_per_million_tokens: float
    memory_usage_gb: float


class PerformanceBenchmark:
    """Performance benchmark runner"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.runtime = boto3.client('sagemaker-runtime')
        self.cloudwatch = boto3.client('cloudwatch')

    def run_latency_test(self) -> List[float]:
        """Measure latency distribution"""
        logger.info(f"Running latency test ({self.config.num_requests} requests)...")

        latencies = []
        prompt = "test " * (self.config.input_tokens // 2)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config.output_tokens,
                "temperature": 0.0,
            }
        }

        for i in range(self.config.num_requests):
            start = time.time()

            try:
                response = self.runtime.invoke_endpoint(
                    EndpointName=self.config.endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(payload)
                )
                latency = time.time() - start
                latencies.append(latency)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Completed {i+1}/{self.config.num_requests} requests")

            except Exception as e:
                logger.error(f"Request failed: {e}")

        return latencies

    def run_throughput_test(self, duration_seconds: int = 60) -> float:
        """Measure sustained throughput"""
        logger.info(f"Running throughput test ({duration_seconds}s)...")

        import concurrent.futures

        def make_request():
            payload = {
                "inputs": "test " * 100,
                "parameters": {"max_new_tokens": self.config.output_tokens}
            }

            try:
                self.runtime.invoke_endpoint(
                    EndpointName=self.config.endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(payload)
                )
                return 1
            except:
                return 0

        start_time = time.time()
        completed_requests = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.concurrent_requests) as executor:
            while time.time() - start_time < duration_seconds:
                futures = [executor.submit(make_request) for _ in range(self.config.concurrent_requests)]
                results = [f.result() for f in futures]
                completed_requests += sum(results)

        elapsed = time.time() - start_time
        throughput = completed_requests / elapsed

        logger.info(f"  Completed {completed_requests} requests in {elapsed:.2f}s")
        logger.info(f"  Throughput: {throughput:.2f} req/sec")

        return throughput

    def get_memory_usage(self) -> float:
        """Get memory usage from CloudWatch"""
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='MemoryUtilization',
                Dimensions=[
                    {'Name': 'EndpointName', 'Value': self.config.endpoint_name},
                    {'Name': 'VariantName', 'Value': 'AllTraffic'},
                ],
                StartTime=time.time() - 300,
                EndTime=time.time(),
                Period=300,
                Statistics=['Average'],
            )

            if response['Datapoints']:
                return response['Datapoints'][0]['Average']
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")

        return 0.0

    def calculate_cost(self, throughput_rps: float) -> Dict[str, float]:
        """Calculate cost metrics"""
        # Instance pricing (examples - update with actual pricing)
        instance_pricing = {
            "ml.g5.xlarge": 1.01,
            "ml.g5.2xlarge": 1.52,
            "ml.g5.12xlarge": 7.09,
            "ml.p4d.24xlarge": 32.77,
            "ml.inf2.xlarge": 0.76,
            "ml.inf2.48xlarge": 12.98,
            "ml.g5g.xlarge": 0.61,
        }

        cost_per_hour = instance_pricing.get(self.config.instance_type, 0) * self.config.instance_count

        # Tokens per request
        tokens_per_request = self.config.input_tokens + self.config.output_tokens

        # Requests per hour
        requests_per_hour = throughput_rps * 3600

        # Tokens per hour
        tokens_per_hour = requests_per_hour * tokens_per_request

        # Cost per million tokens
        if tokens_per_hour > 0:
            cost_per_million_tokens = (cost_per_hour / tokens_per_hour) * 1_000_000
        else:
            cost_per_million_tokens = 0

        return {
            "cost_per_hour": cost_per_hour,
            "cost_per_million_tokens": cost_per_million_tokens,
        }

    def run(self) -> BenchmarkResults:
        """Run complete benchmark suite"""
        logger.info("="*80)
        logger.info("STARTING BENCHMARK")
        logger.info("="*80)
        logger.info(f"Endpoint: {self.config.endpoint_name}")
        logger.info(f"Container: {self.config.container_type}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Instance: {self.config.instance_type}")

        # Latency test
        latencies = self.run_latency_test()

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        logger.info(f"\nLatency Results:")
        logger.info(f"  P50: {p50*1000:.2f}ms")
        logger.info(f"  P95: {p95*1000:.2f}ms")
        logger.info(f"  P99: {p99*1000:.2f}ms")

        # Throughput test
        throughput_rps = self.run_throughput_test(duration_seconds=60)
        throughput_tps = throughput_rps * (self.config.input_tokens + self.config.output_tokens)

        logger.info(f"\nThroughput Results:")
        logger.info(f"  {throughput_rps:.2f} requests/sec")
        logger.info(f"  {throughput_tps:.2f} tokens/sec")

        # Memory usage
        memory_usage = self.get_memory_usage()

        # Cost calculations
        cost_metrics = self.calculate_cost(throughput_rps)

        logger.info(f"\nCost Analysis:")
        logger.info(f"  ${cost_metrics['cost_per_hour']:.2f}/hour")
        logger.info(f"  ${cost_metrics['cost_per_million_tokens']:.2f} per 1M tokens")

        # Create results
        results = BenchmarkResults(
            container_type=self.config.container_type,
            model_name=self.config.model_name,
            instance_type=self.config.instance_type,
            throughput_rps=throughput_rps,
            throughput_tps=throughput_tps,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            cost_per_hour=cost_metrics['cost_per_hour'],
            cost_per_million_tokens=cost_metrics['cost_per_million_tokens'],
            memory_usage_gb=memory_usage,
        )

        logger.info("="*80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("="*80)

        return results


def save_results(results: BenchmarkResults, output_file: str):
    """Save results to JSON"""
    import json

    results_dict = {
        "container_type": results.container_type,
        "model_name": results.model_name,
        "instance_type": results.instance_type,
        "throughput": {
            "requests_per_sec": results.throughput_rps,
            "tokens_per_sec": results.throughput_tps,
        },
        "latency_ms": {
            "p50": results.latency_p50 * 1000,
            "p95": results.latency_p95 * 1000,
            "p99": results.latency_p99 * 1000,
        },
        "cost": {
            "per_hour": results.cost_per_hour,
            "per_million_tokens": results.cost_per_million_tokens,
        },
        "memory_usage_gb": results.memory_usage_gb,
    }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument("--endpoint", required=True, help="SageMaker endpoint name")
    parser.add_argument("--container", required=True, choices=["vllm", "sglang", "tgi", "neuron"])
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--instance-type", required=True, help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=100)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--output", default="benchmark_results.json")

    args = parser.parse_args()

    config = BenchmarkConfig(
        endpoint_name=args.endpoint,
        container_type=args.container,
        model_name=args.model,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        num_requests=args.num_requests,
        concurrent_requests=args.concurrent,
    )

    benchmark = PerformanceBenchmark(config)
    results = benchmark.run()

    save_results(results, args.output)


if __name__ == "__main__":
    main()
