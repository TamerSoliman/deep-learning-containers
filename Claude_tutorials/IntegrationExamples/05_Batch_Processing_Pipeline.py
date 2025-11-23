#!/usr/bin/env python3
"""
Batch Processing Pipeline with SageMaker Batch Transform

For high-throughput, offline inference:
- Process millions of records
- Cost-effective (no always-on endpoint)
- Automatic scaling
- Output directly to S3
"""

import boto3
import sagemaker
from sagemaker.transformer import Transformer
from sagemaker import Model
import json


def create_batch_input_file(prompts: list, output_path: str):
    """Create JSONL input file for batch transform"""
    with open(output_path, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps({"inputs": prompt}) + '\n')


def run_batch_inference(
    model_s3: str,
    input_s3: str,
    output_s3: str,
    instance_type: str = "ml.g5.12xlarge",
    instance_count: int = 2,
):
    """
    Run batch inference using SageMaker Batch Transform

    Args:
        model_s3: S3 path to model
        input_s3: S3 path to input JSONL file
        output_s3: S3 output location
        instance_type: Instance type for batch job
        instance_count: Number of instances (parallel processing)
    """
    role = sagemaker.get_execution_role()

    # Create model
    model = Model(
        image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.11.2-gpu-sagemaker",
        model_data=model_s3,
        role=role,
        env={
            "SM_VLLM_TENSOR_PARALLEL_SIZE": "4",
            "SM_VLLM_MAX_MODEL_LEN": "4096",
        }
    )

    # Create transformer
    transformer = Transformer(
        model_name=model.name,
        instance_count=instance_count,
        instance_type=instance_type,
        strategy='MultiRecord',  # Process multiple records per request
        max_payload=1,          # MB per request
        max_concurrent_transforms=8,  # Parallel requests per instance
        output_path=output_s3,
    )

    # Start batch transform job
    transformer.transform(
        data=input_s3,
        data_type='S3Prefix',
        content_type='application/jsonl',
        split_type='Line',
    )

    # Wait for completion
    transformer.wait()

    print(f"✓ Batch transform complete! Output: {output_s3}")

    return output_s3


# Example: Process customer support tickets
def process_support_tickets():
    """Batch process support tickets for classification"""

    # Sample tickets
    tickets = [
        "My order hasn't arrived yet, tracking shows it's been stuck for a week",
        "I need to return this product, it doesn't work as described",
        "Can you help me reset my password? I forgot it",
        # ... thousands more
    ]

    # Create input file
    create_batch_input_file(tickets, "./batch_input.jsonl")

    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file("./batch_input.jsonl", "your-bucket", "batch-jobs/input.jsonl")

    # Run batch inference
    output_s3 = run_batch_inference(
        model_s3="s3://your-bucket/models/ticket-classifier.tar.gz",
        input_s3="s3://your-bucket/batch-jobs/input.jsonl",
        output_s3="s3://your-bucket/batch-jobs/output/",
        instance_count=5,  # Scale for high throughput
    )

    return output_s3


# Example: Generate product descriptions
def generate_product_descriptions():
    """Batch generate descriptions for e-commerce products"""

    # Product data
    products = [
        {"name": "Wireless Mouse", "specs": "Bluetooth, 1000 DPI, Ergonomic"},
        {"name": "USB-C Cable", "specs": "3ft, Fast Charging, Durable Nylon"},
        # ... thousands more
    ]

    # Create prompts
    prompts = [
        f"Write a compelling product description for: {p['name']} ({p['specs']})"
        for p in products
    ]

    create_batch_input_file(prompts, "./batch_input.jsonl")

    # Upload and process
    s3 = boto3.client('s3')
    s3.upload_file("./batch_input.jsonl", "your-bucket", "batch-jobs/descriptions-input.jsonl")

    output_s3 = run_batch_inference(
        model_s3="s3://your-bucket/models/llama-3-8b.tar.gz",
        input_s3="s3://your-bucket/batch-jobs/descriptions-input.jsonl",
        output_s3="s3://your-bucket/batch-jobs/descriptions-output/",
        instance_count=10,  # 10 instances for fast processing
    )

    return output_s3


# Cost Optimization: Use Spot Instances
def run_batch_with_spot_instances(input_s3: str, output_s3: str):
    """Use EC2 Spot instances for 70% cost savings"""

    transformer = Transformer(
        model_name="my-model",
        instance_count=5,
        instance_type="ml.g5.12xlarge",
        use_spot_instances=True,  # Use Spot!
        max_wait=7200,           # Wait up to 2 hours for Spot capacity
        output_path=output_s3,
    )

    transformer.transform(data=input_s3, content_type='application/jsonl')
    transformer.wait()


if __name__ == "__main__":
    # Example 1: Process support tickets
    output = process_support_tickets()
    print(f"Support tickets processed: {output}")

    # Example 2: Generate product descriptions
    output = generate_product_descriptions()
    print(f"Descriptions generated: {output}")
