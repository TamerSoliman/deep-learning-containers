#!/usr/bin/env python3
"""
Automated Container Functionality Tests

Tests:
- Container health checks
- Model loading
- Inference correctness
- Performance benchmarks
"""

import boto3
import pytest
import json
import time


class TestContainerFunctionality:
    """Test suite for DLC containers"""

    @pytest.fixture
    def sagemaker_endpoint(self):
        """Fixture for SageMaker endpoint"""
        return "test-endpoint-vllm"

    @pytest.fixture
    def runtime_client(self):
        return boto3.client('sagemaker-runtime')

    def test_endpoint_is_in_service(self, sagemaker_endpoint):
        """Test endpoint is running"""
        sm = boto3.client('sagemaker')
        response = sm.describe_endpoint(EndpointName=sagemaker_endpoint)

        assert response['EndpointStatus'] == 'InService', \
            f"Endpoint not in service: {response['EndpointStatus']}"

    def test_health_check(self, sagemaker_endpoint, runtime_client):
        """Test endpoint health check"""
        try:
            response = runtime_client.invoke_endpoint(
                EndpointName=sagemaker_endpoint,
                ContentType='application/json',
                Body=json.dumps({"inputs": "test"})
            )
            assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")

    def test_simple_inference(self, sagemaker_endpoint, runtime_client):
        """Test basic inference"""
        payload = {
            "inputs": "The capital of France is",
            "parameters": {"max_new_tokens": 10}
        }

        response = runtime_client.invoke_endpoint(
            EndpointName=sagemaker_endpoint,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        result = json.loads(response['Body'].read())
        assert len(result) > 0
        assert 'generated_text' in result[0]
        assert 'Paris' in result[0]['generated_text']

    def test_batch_inference(self, sagemaker_endpoint, runtime_client):
        """Test batch processing"""
        prompts = [
            "What is 2+2?",
            "What is the capital of Germany?",
            "Who wrote Hamlet?",
        ]

        for prompt in prompts:
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 20}
            }

            response = runtime_client.invoke_endpoint(
                EndpointName=sagemaker_endpoint,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            result = json.loads(response['Body'].read())
            assert len(result) > 0, f"Empty response for: {prompt}"

    def test_latency_sla(self, sagemaker_endpoint, runtime_client):
        """Test latency meets SLA"""
        payload = {
            "inputs": "Test prompt",
            "parameters": {"max_new_tokens": 100}
        }

        start = time.time()
        response = runtime_client.invoke_endpoint(
            EndpointName=sagemaker_endpoint,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        latency = time.time() - start

        # SLA: < 2 seconds for 100 tokens
        assert latency < 2.0, f"Latency {latency}s exceeds SLA (2s)"

    def test_concurrent_requests(self, sagemaker_endpoint, runtime_client):
        """Test concurrent request handling"""
        import concurrent.futures

        def make_request():
            payload = {
                "inputs": "Test",
                "parameters": {"max_new_tokens": 10}
            }
            response = runtime_client.invoke_endpoint(
                EndpointName=sagemaker_endpoint,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            return response['ResponseMetadata']['HTTPStatusCode']

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        assert all(r == 200 for r in results), "Some concurrent requests failed"

    def test_error_handling(self, sagemaker_endpoint, runtime_client):
        """Test error handling for invalid inputs"""
        # Test empty input
        with pytest.raises(Exception):
            runtime_client.invoke_endpoint(
                EndpointName=sagemaker_endpoint,
                ContentType='application/json',
                Body=json.dumps({"inputs": ""})
            )

        # Test oversized input
        large_input = "test " * 100000
        with pytest.raises(Exception):
            runtime_client.invoke_endpoint(
                EndpointName=sagemaker_endpoint,
                ContentType='application/json',
                Body=json.dumps({"inputs": large_input})
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
