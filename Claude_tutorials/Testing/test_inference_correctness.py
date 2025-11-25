#!/usr/bin/env python3
"""
Inference Correctness Tests

Validates model outputs are correct and consistent
"""

import boto3
import pytest
import json


class TestInferenceCorrectness:
    """Test inference correctness"""

    @pytest.fixture
    def runtime_client(self):
        return boto3.client('sagemaker-runtime')

    def test_math_reasoning(self, runtime_client):
        """Test mathematical reasoning"""
        test_cases = [
            ("What is 15 + 27?", "42"),
            ("If I have 10 apples and give away 3, how many do I have?", "7"),
        ]

        for prompt, expected in test_cases:
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 50, "temperature": 0.0}
            }

            response = runtime_client.invoke_endpoint(
                EndpointName="test-endpoint",
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            result = json.loads(response['Body'].read())
            output = result[0]['generated_text']

            assert expected in output, f"Expected '{expected}' in output: {output}"

    def test_factual_knowledge(self, runtime_client):
        """Test factual knowledge"""
        test_cases = [
            ("What is the capital of France?", "Paris"),
            ("Who wrote Romeo and Juliet?", "Shakespeare"),
            ("What is the boiling point of water in Celsius?", "100"),
        ]

        for prompt, expected in test_cases:
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 30, "temperature": 0.0}
            }

            response = runtime_client.invoke_endpoint(
                EndpointName="test-endpoint",
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            result = json.loads(response['Body'].read())
            output = result[0]['generated_text'].lower()

            assert expected.lower() in output, \
                f"Expected '{expected}' in output: {output}"

    def test_consistency(self, runtime_client):
        """Test output consistency (same input = same output)"""
        prompt = "The capital of France is"

        outputs = []
        for _ in range(5):
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 10, "temperature": 0.0}  # Greedy
            }

            response = runtime_client.invoke_endpoint(
                EndpointName="test-endpoint",
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            result = json.loads(response['Body'].read())
            outputs.append(result[0]['generated_text'])

        # All outputs should be identical (greedy decoding)
        assert all(o == outputs[0] for o in outputs), \
            f"Outputs not consistent: {outputs}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
