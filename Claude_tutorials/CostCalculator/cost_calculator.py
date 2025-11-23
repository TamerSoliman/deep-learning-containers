#!/usr/bin/env python3
"""
Interactive Cost Optimization Calculator for DLC Deployments

Helps choose the most cost-effective container + instance combination
"""

import json
from dataclasses import dataclass
from typing import List, Tuple
import argparse


@dataclass
class InstancePricing:
    """Instance pricing information"""
    instance_type: str
    gpus: int
    gpu_memory_gb: int
    cost_per_hour: float
    supports_spot: bool = True


@dataclass
class Recommendation:
    """Cost optimization recommendation"""
    container: str
    instance_type: str
    instance_count: int
    estimated_throughput: float  # req/sec
    estimated_latency_p95: float  # ms
    monthly_cost: float
    cost_per_million_tokens: float
    notes: str


# Instance pricing database
INSTANCE_PRICING = {
    # GPU instances
    "ml.g5.xlarge": InstancePricing("ml.g5.xlarge", 1, 24, 1.01),
    "ml.g5.2xlarge": InstancePricing("ml.g5.2xlarge", 1, 24, 1.52),
    "ml.g5.12xlarge": InstancePricing("ml.g5.12xlarge", 4, 96, 7.09),
    "ml.p4d.24xlarge": InstancePricing("ml.p4d.24xlarge", 8, 320, 32.77),

    # ARM64 + GPU
    "ml.g5g.xlarge": InstancePricing("ml.g5g.xlarge", 1, 16, 0.61),
    "ml.g5g.16xlarge": InstancePricing("ml.g5g.16xlarge", 1, 24, 4.20),

    # Inferentia2
    "ml.inf2.xlarge": InstancePricing("ml.inf2.xlarge", 0, 32, 0.76),
    "ml.inf2.48xlarge": InstancePricing("ml.inf2.48xlarge", 0, 384, 12.98),
}


class CostCalculator:
    """Calculate and recommend cost-optimal deployments"""

    def __init__(self, model_size_gb: float, requests_per_day: int, max_latency_p95_ms: float):
        self.model_size_gb = model_size_gb
        self.requests_per_day = requests_per_day
        self.max_latency_p95_ms = max_latency_p95_ms

    def estimate_throughput(self, container: str, instance_type: str) -> float:
        """Estimate throughput (requests/sec)"""
        # Simplified model - in production, use actual benchmarks
        pricing = INSTANCE_PRICING.get(instance_type)

        if not pricing:
            return 1.0

        # Base throughput on GPUs/cores
        if container == "vllm":
            base_throughput = pricing.gpus * 6.0  # ~6 req/sec per GPU
        elif container == "sglang":
            base_throughput = pricing.gpus * 5.5
        elif container == "neuronx":
            base_throughput = 2.5  # Conservative for Neuron
        else:
            base_throughput = pricing.gpus * 4.0

        # Adjust for model size (larger models = slower)
        if self.model_size_gb > 100:
            base_throughput *= 0.6
        elif self.model_size_gb > 50:
            base_throughput *= 0.8

        return base_throughput

    def estimate_latency(self, container: str, instance_type: str) -> float:
        """Estimate P95 latency (ms)"""
        pricing = INSTANCE_PRICING.get(instance_type)

        if container == "neuronx":
            return 600  # Neuron typically slower
        elif "g5g" in instance_type:
            return 550  # ARM64 slightly slower
        elif pricing.gpus >= 8:
            return 400  # Fast multi-GPU
        else:
            return 500  # Default

    def calculate_cost(self, container: str, instance_type: str, instance_count: int) -> Tuple[float, float]:
        """Calculate monthly cost and cost per million tokens"""
        pricing = INSTANCE_PRICING.get(instance_type)

        # Monthly cost (24/7 operation)
        monthly_cost = pricing.cost_per_hour * 730 * instance_count  # 730 hrs/month

        # Throughput and token calculations
        throughput_rps = self.estimate_throughput(container, instance_type) * instance_count
        requests_per_hour = throughput_rps * 3600
        tokens_per_request = 2148  # Average (2K input + 100 output)
        tokens_per_hour = requests_per_hour * tokens_per_request

        if tokens_per_hour > 0:
            cost_per_million_tokens = (pricing.cost_per_hour * instance_count / tokens_per_hour) * 1_000_000
        else:
            cost_per_million_tokens = 0

        return monthly_cost, cost_per_million_tokens

    def find_optimal_deployment(self) -> List[Recommendation]:
        """Find optimal deployment configurations"""
        recommendations = []

        # Try different container + instance combinations
        for container in ["vllm", "sglang", "neuronx", "vllm-arm64"]:
            for instance_type in INSTANCE_PRICING.keys():
                # Skip incompatible combinations
                if container == "neuronx" and "inf2" not in instance_type:
                    continue
                if container == "vllm-arm64" and "g5g" not in instance_type:
                    continue
                if container in ["vllm", "sglang"] and ("inf2" in instance_type or "g5g" in instance_type):
                    continue

                # Skip if model doesn't fit
                pricing = INSTANCE_PRICING[instance_type]
                if self.model_size_gb > pricing.gpu_memory_gb and pricing.gpus == 1:
                    continue

                # Calculate instances needed
                throughput_per_instance = self.estimate_throughput(container, instance_type)
                required_throughput = self.requests_per_day / 86400  # req/sec

                instance_count = max(1, int(required_throughput / throughput_per_instance) + 1)

                # Estimate latency
                latency = self.estimate_latency(container, instance_type)

                # Skip if latency too high
                if latency > self.max_latency_p95_ms:
                    continue

                # Calculate costs
                monthly_cost, cost_per_mil_tokens = self.calculate_cost(container, instance_type, instance_count)

                # Create recommendation
                rec = Recommendation(
                    container=container,
                    instance_type=instance_type,
                    instance_count=instance_count,
                    estimated_throughput=throughput_per_instance * instance_count,
                    estimated_latency_p95=latency,
                    monthly_cost=monthly_cost,
                    cost_per_million_tokens=cost_per_mil_tokens,
                    notes=f"{instance_count}x {instance_type}",
                )

                recommendations.append(rec)

        # Sort by cost (cheapest first)
        recommendations.sort(key=lambda x: x.monthly_cost)

        return recommendations


def print_recommendations(recommendations: List[Recommendation], top_n: int = 3):
    """Print top recommendations"""
    print("\n" + "="*100)
    print("COST OPTIMIZATION RECOMMENDATIONS")
    print("="*100)

    for i, rec in enumerate(recommendations[:top_n], 1):
        print(f"\n{i}. {rec.container.upper()} on {rec.instance_type} ({rec.instance_count} instance(s))")
        print(f"   Monthly Cost:        ${rec.monthly_cost:,.2f}")
        print(f"   Cost/1M tokens:      ${rec.cost_per_million_tokens:.2f}")
        print(f"   Est. Throughput:     {rec.estimated_throughput:.1f} req/sec")
        print(f"   Est. P95 Latency:    {rec.estimated_latency_p95:.0f}ms")

        if i == 1:
            print(f"   💡 BEST CHOICE ⭐")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="Cost optimization calculator")
    parser.add_argument("--model-size-gb", type=float, required=True, help="Model size in GB")
    parser.add_argument("--requests-per-day", type=int, required=True, help="Expected requests/day")
    parser.add_argument("--max-latency-ms", type=float, default=1000, help="Max acceptable P95 latency (ms)")
    parser.add_argument("--top-n", type=int, default=5, help="Show top N recommendations")

    args = parser.parse_args()

    print("\n" + "="*100)
    print("COST CALCULATOR INPUT")
    print("="*100)
    print(f"Model Size:          {args.model_size_gb} GB")
    print(f"Expected Traffic:    {args.requests_per_day:,} requests/day")
    print(f"Max Latency (P95):   {args.max_latency_ms}ms")
    print("="*100)

    calculator = CostCalculator(
        model_size_gb=args.model_size_gb,
        requests_per_day=args.requests_per_day,
        max_latency_p95_ms=args.max_latency_ms,
    )

    recommendations = calculator.find_optimal_deployment()

    if recommendations:
        print_recommendations(recommendations, top_n=args.top_n)
    else:
        print("\n⚠️  No configurations found meeting your requirements.")
        print("Try relaxing the latency constraint or reducing expected traffic.")


if __name__ == "__main__":
    # Example: python cost_calculator.py --model-size-gb 140 --requests-per-day 100000 --max-latency-ms 1000
    main()
