"""
SageMaker Python SDK Deployment Template for Foundation Model Inference
=======================================================================

This template demonstrates deploying a Foundation Model (LLM) using vLLM DLC
to a SageMaker real-time endpoint with the SageMaker Python SDK.

Framework: vLLM 0.11.2
Model Example: Llama 3 70B
Instance: ml.p4d.24xlarge (8x A100 80GB)
Optimization: 8-way Tensor Parallelism, PagedAttention

REQUIREMENTS:
  pip install sagemaker>=2.200.0 boto3>=1.34.0

USAGE:
  1. Set AWS credentials (AWS_PROFILE or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
  2. Update PLACEHOLDERS below (IAM role, S3 paths, model name)
  3. Run: python deploy_sagemaker_sdk_template.py
"""

import json
import time
from typing import Dict, Any, Optional

import boto3
from sagemaker import Model, Session
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


# =============================================================================
# CONFIGURATION SECTION - UPDATE THESE PLACEHOLDERS
# =============================================================================

# AWS Configuration
AWS_REGION = "us-east-1"  # Update to your region
IAM_ROLE = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"  # REQUIRED: Update with your IAM role

# Model Configuration
MODEL_NAME = "llama-3-70b-vllm"  # Descriptive name for this deployment
HF_MODEL_ID = "meta-llama/Llama-3-70b-hf"  # HuggingFace model ID
HF_TOKEN = None  # Set to "hf_..." if model is gated, else None

# Option 1: Use model from HuggingFace Hub (simpler, slower first startup)
MODEL_DATA_S3_URI = None  # Set to None to download from Hub at runtime

# Option 2: Use pre-downloaded model from S3 (faster startup)
# MODEL_DATA_S3_URI = "s3://my-bucket/models/llama-3-70b.tar.gz"

# Inference Configuration
INSTANCE_TYPE = "ml.p4d.24xlarge"  # 8x A100 80GB - adjust based on model size
INSTANCE_COUNT = 1  # Number of instances (typically 1 for LLMs)
TENSOR_PARALLEL_SIZE = 8  # Must match GPU count on instance

# vLLM Optimization Settings
MAX_MODEL_LEN = 4096  # Maximum sequence length (input + output)
GPU_MEMORY_UTILIZATION = 0.95  # Fraction of GPU memory for KV cache (0.85-0.95)
ENABLE_PREFIX_CACHING = True  # Cache system prompts for faster inference

# Endpoint Configuration
ENDPOINT_NAME = f"{MODEL_NAME}-endpoint"  # Auto-generated, or set custom name
INITIAL_INSTANCE_COUNT = 1  # Instances to start with

# DLC Image Configuration
# Find latest images at: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
VLLM_IMAGE_URI = f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/vllm:0.11.2-sagemaker"


# =============================================================================
# DEPLOYMENT FUNCTIONS
# =============================================================================

def create_model_env_vars(
    model_id: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enable_prefix_caching: bool,
    hf_token: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create environment variable dictionary for vLLM container.

    All vLLM configuration is done via environment variables with SM_VLLM_ prefix.
    The entry point script transforms these to CLI arguments.

    Args:
        model_id: HuggingFace model ID or path (/opt/ml/model for S3-backed)
        tensor_parallel_size: Number of GPUs to parallelize across
        max_model_len: Maximum total sequence length (input + output tokens)
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        enable_prefix_caching: Whether to cache prompt prefixes
        hf_token: HuggingFace API token for gated models

    Returns:
        Dict of environment variables for SageMaker Model
    """
    env = {
        # Model specification
        "SM_VLLM_MODEL": model_id,  # HF model ID or /opt/ml/model

        # Performance: Tensor Parallelism (split model across GPUs)
        "SM_VLLM_TENSOR_PARALLEL_SIZE": str(tensor_parallel_size),

        # Memory management
        "SM_VLLM_MAX_MODEL_LEN": str(max_model_len),  # Longer = more memory
        "SM_VLLM_GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),  # KV cache size

        # Optimization: Prefix caching for repeated prompts (e.g., system prompts)
        "SM_VLLM_ENABLE_PREFIX_CACHING": "true" if enable_prefix_caching else "false",

        # Data type (bfloat16 recommended for A100/H100)
        "SM_VLLM_DTYPE": "bfloat16",

        # Trust remote code (required for some models like Qwen, MPT)
        "SM_VLLM_TRUST_REMOTE_CODE": "true",

        # HuggingFace Hub optimization
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Use fast Rust-based downloads
    }

    # Add HuggingFace token if provided (for gated models like Llama 3)
    if hf_token:
        env["HF_TOKEN"] = hf_token

    return env


def deploy_vllm_model(
    model_name: str,
    image_uri: str,
    role: str,
    model_data: Optional[str],
    env: Dict[str, str],
    instance_type: str,
    initial_instance_count: int = 1,
    endpoint_name: Optional[str] = None,
) -> Any:
    """
    Deploy vLLM model to SageMaker real-time endpoint.

    This function:
    1. Creates a SageMaker Model resource (container + env vars)
    2. Creates an Endpoint Configuration (instance type/count)
    3. Deploys an Endpoint (launches instances and loads model)

    Args:
        model_name: Name for the SageMaker Model resource
        image_uri: DLC image URI (vLLM container)
        role: IAM role ARN with SageMaker permissions
        model_data: S3 URI to model.tar.gz, or None to download from Hub
        env: Environment variables for container
        instance_type: EC2 instance type (e.g., ml.p4d.24xlarge)
        initial_instance_count: Number of instances to launch
        endpoint_name: Optional custom endpoint name

    Returns:
        SageMaker Predictor object for making inference requests
    """
    # Create SageMaker session
    sagemaker_session = Session(boto_session=boto3.Session(region_name=AWS_REGION))

    # Create Model resource
    print(f"Creating SageMaker Model: {model_name}")
    model = Model(
        name=model_name,
        image_uri=image_uri,
        model_data=model_data,  # None = download from Hub, or S3 URI
        role=role,
        env=env,
        sagemaker_session=sagemaker_session,
    )

    # Deploy to endpoint (this creates EndpointConfig and Endpoint)
    print(f"Deploying endpoint: {endpoint_name or 'auto-generated'}")
    print(f"  Instance type: {instance_type}")
    print(f"  Instance count: {initial_instance_count}")
    print("\nThis will take 10-30 minutes depending on model size...")
    print("  - Launching EC2 instances (~5 mins)")
    print("  - Downloading model weights (~5-25 mins)")
    print("  - Loading model into GPU memory (~2-5 mins)")

    predictor = model.deploy(
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        serializer=JSONSerializer(),  # Serialize requests as JSON
        deserializer=JSONDeserializer(),  # Deserialize responses as JSON
        wait=True,  # Block until endpoint is InService
    )

    print(f"\n✅ Endpoint deployed successfully: {predictor.endpoint_name}")
    return predictor


def test_inference(predictor: Any, test_prompts: list) -> None:
    """
    Test inference with sample prompts.

    vLLM implements OpenAI Chat Completions API.

    Args:
        predictor: SageMaker Predictor from deploy_vllm_model()
        test_prompts: List of test prompts
    """
    print("\n" + "="*80)
    print("TESTING INFERENCE")
    print("="*80)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")

        # OpenAI Chat Completions API format
        request_payload = {
            "model": HF_MODEL_ID,  # Model identifier (required by OpenAI API)
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 256,  # Maximum tokens to generate
            "temperature": 0.7,  # Sampling temperature (0.0 = greedy, 1.0 = diverse)
            "top_p": 0.9,  # Nucleus sampling parameter
        }

        # Send request
        start_time = time.time()
        response = predictor.predict(request_payload)
        elapsed = time.time() - start_time

        # Parse response
        generated_text = response["choices"][0]["message"]["content"]
        tokens_generated = response["usage"]["completion_tokens"]
        throughput = tokens_generated / elapsed

        print(f"Response: {generated_text}")
        print(f"Stats: {tokens_generated} tokens in {elapsed:.2f}s ({throughput:.1f} tokens/sec)")

        # Assert successful response
        assert response["choices"][0]["finish_reason"] in ["stop", "length"], \
            f"Unexpected finish reason: {response['choices'][0]['finish_reason']}"


def cleanup_endpoint(predictor: Any, delete_model: bool = True) -> None:
    """
    Delete SageMaker endpoint and optionally model/endpoint-config.

    Args:
        predictor: SageMaker Predictor
        delete_model: Whether to also delete Model and EndpointConfig
    """
    print("\n" + "="*80)
    print("CLEANING UP RESOURCES")
    print("="*80)

    endpoint_name = predictor.endpoint_name

    # Delete endpoint (stops instances, no more charges)
    print(f"Deleting endpoint: {endpoint_name}")
    predictor.delete_endpoint(delete_endpoint_config=delete_model)

    if delete_model:
        # Delete model
        print(f"Deleting model: {MODEL_NAME}")
        predictor.delete_model()
        print("✅ All resources deleted")
    else:
        print("✅ Endpoint deleted (model and config retained)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main deployment workflow:
    1. Create environment variables for vLLM
    2. Deploy model to SageMaker endpoint
    3. Test inference with sample prompts
    4. Clean up (optional)
    """

    # Validate configuration
    assert IAM_ROLE != "arn:aws:iam::123456789012:role/SageMakerExecutionRole", \
        "ERROR: Update IAM_ROLE placeholder with your actual IAM role ARN"

    print("="*80)
    print("SAGEMAKER VLLM DEPLOYMENT")
    print("="*80)
    print(f"Model: {HF_MODEL_ID}")
    print(f"Instance: {INSTANCE_TYPE} x{INSTANCE_COUNT}")
    print(f"Tensor Parallelism: {TENSOR_PARALLEL_SIZE} GPUs")
    print(f"Max Sequence Length: {MAX_MODEL_LEN}")
    print(f"Region: {AWS_REGION}")

    # Determine model path (Hub download vs S3)
    if MODEL_DATA_S3_URI:
        print(f"Model Source: S3 ({MODEL_DATA_S3_URI})")
        model_id = "/opt/ml/model"  # vLLM loads from local path
    else:
        print(f"Model Source: HuggingFace Hub (runtime download)")
        model_id = HF_MODEL_ID  # vLLM downloads from Hub

    # Step 1: Create environment variables
    env_vars = create_model_env_vars(
        model_id=model_id,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        hf_token=HF_TOKEN,
    )

    print("\nEnvironment Variables:")
    for key, value in env_vars.items():
        # Mask token for security
        display_value = value if key != "HF_TOKEN" else "***REDACTED***"
        print(f"  {key}={display_value}")

    # Step 2: Deploy model
    predictor = deploy_vllm_model(
        model_name=MODEL_NAME,
        image_uri=VLLM_IMAGE_URI,
        role=IAM_ROLE,
        model_data=MODEL_DATA_S3_URI,
        env=env_vars,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        endpoint_name=ENDPOINT_NAME,
    )

    # Step 3: Test inference
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about machine learning.",
    ]

    test_inference(predictor, test_prompts)

    # Step 4: Clean up (OPTIONAL - comment out to keep endpoint running)
    # WARNING: Endpoint costs ~$32/hour for ml.p4d.24xlarge
    cleanup_prompt = input("\nDelete endpoint? (y/N): ")
    if cleanup_prompt.lower() == 'y':
        cleanup_endpoint(predictor, delete_model=True)
    else:
        print(f"\n✅ Endpoint running: {predictor.endpoint_name}")
        print(f"   Estimated cost: ~$32/hour (ml.p4d.24xlarge)")
        print(f"   To delete later: predictor.delete_endpoint()")


if __name__ == "__main__":
    main()


# =============================================================================
# ADDITIONAL CONFIGURATION EXAMPLES
# =============================================================================

"""
# Example 1: Smaller Model (Llama 3 8B on single GPU)
# Instance: ml.g5.2xlarge (1x A10G 24GB)

model_id = "meta-llama/Llama-3-8b-hf"
instance_type = "ml.g5.2xlarge"
tensor_parallel_size = 1
max_model_len = 8192  # Support longer contexts
gpu_memory_utilization = 0.90


# Example 2: Quantized Model (AWQ - faster, less memory)
# Run AWQ quantization offline, upload to S3, then deploy

env = {
    "SM_VLLM_MODEL": "/opt/ml/model",
    "SM_VLLM_QUANTIZATION": "awq",  # Activate AWQ quantization
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "4",  # Smaller model, fewer GPUs
    "SM_VLLM_DTYPE": "float16",
}


# Example 3: Multi-Instance Deployment (for high throughput)
# SageMaker automatically load balances across instances

predictor = model.deploy(
    instance_type="ml.p4d.24xlarge",
    initial_instance_count=3,  # 3 instances = 3x throughput
    endpoint_name="llama-3-70b-high-throughput",
)


# Example 4: Auto-Scaling Configuration
# Add auto-scaling to handle variable traffic

import boto3

asg_client = boto3.client('application-autoscaling', region_name=AWS_REGION)

# Register endpoint as scalable target
asg_client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=5,  # Scale up to 5 instances
)

# Create target tracking scaling policy (based on invocations per instance)
asg_client.put_scaling_policy(
    PolicyName=f'{predictor.endpoint_name}-scaling-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 1000.0,  # Target 1000 invocations per instance
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance',
        },
        'ScaleInCooldown': 300,  # Wait 5 min before scaling in
        'ScaleOutCooldown': 60,  # Wait 1 min before scaling out
    }
)

print("✅ Auto-scaling configured: 1-5 instances")
"""
