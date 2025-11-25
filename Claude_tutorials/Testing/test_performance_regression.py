#!/usr/bin/env python3
"""
Performance Regression Tests

Ensure new versions don't regress on performance
"""

import boto3
import pytest
import json
import time
import numpy as np


class TestPerformanceRegression:
    """Test performance metrics"""

    @pytest.fixture
    def runtime_client(self):
        return boto3.client('sagemaker-runtime')

    def measure_latency(self, endpoint, runtime_client, n_runs=10):
        """Measure average latency"""
        latencies = []

        payload = {
            "inputs": "Test prompt for latency measurement",
            "parameters": {"max_new_tokens": 100, "temperature": 0.0}
        }

        for _ in range(n_runs):
            start = time.time()
            runtime_client.invoke_endpoint(
                EndpointName=endpoint,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            latencies.append(time.time() - start)

        return {
            "mean": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
        }

    def test_latency_regression(self, runtime_client):
        """Test latency hasn't regressed"""
        # Baseline metrics (from previous version)
        baseline = {
            "mean": 0.5,  # seconds
            "p95": 0.8,
            "p99": 1.0,
        }

        # Current metrics
        current = self.measure_latency("test-endpoint", runtime_client)

        # Allow 10% regression
        tolerance = 1.1

        assert current["mean"] <= baseline["mean"] * tolerance, \
            f"Mean latency regressed: {current['mean']} > {baseline['mean']}"

        assert current["p95"] <= baseline["p95"] * tolerance, \
            f"P95 latency regressed: {current['p95']} > {baseline['p95']}"

    def test_throughput_regression(self, runtime_client):
        """Test throughput hasn't regressed"""
        # Baseline throughput
        baseline_throughput = 100  # tokens/sec

        # Measure current throughput
        payload = {
            "inputs": "Test prompt",
            "parameters": {"max_new_tokens": 1000, "temperature": 0.0}
        }

        start = time.time()
        response = runtime_client.invoke_endpoint(
            EndpointName="test-endpoint",
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        elapsed = time.time() - start

        current_throughput = 1000 / elapsed

        # Allow 10% regression
        assert current_throughput >= baseline_throughput * 0.9, \
            f"Throughput regressed: {current_throughput} < {baseline_throughput}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
