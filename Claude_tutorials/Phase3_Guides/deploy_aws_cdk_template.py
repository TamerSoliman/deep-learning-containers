"""
AWS CDK Deployment Template for Foundation Model Inference
==========================================================

This template demonstrates deploying a Foundation Model (LLM) using vLLM DLC
to a SageMaker real-time endpoint using AWS Cloud Development Kit (CDK).

Framework: vLLM 0.11.2
Model Example: Llama 3 70B
Instance: ml.p4d.24xlarge (8x A100 80GB)
Infrastructure as Code: AWS CDK (Python)

REQUIREMENTS:
  pip install aws-cdk-lib>=2.100.0 constructs>=10.0.0

USAGE:
  1. Update PLACEHOLDERS in SageMakerVllmStack class
  2. Initialize CDK (first time only):
       cdk bootstrap aws://ACCOUNT-ID/REGION
  3. Deploy stack:
       cdk deploy
  4. Destroy stack (cleanup):
       cdk destroy

CDK COMMANDS:
  cdk synth        - Synthesize CloudFormation template (preview)
  cdk diff         - Show differences between deployed stack and local code
  cdk deploy       - Deploy stack to AWS
  cdk destroy      - Delete all resources
  cdk ls           - List all stacks in the app
"""

from typing import Dict, Optional
from aws_cdk import (
    App,
    Stack,
    CfnOutput,
    RemovalPolicy,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
)
from constructs import Construct


class SageMakerVllmStack(Stack):
    """
    CDK Stack to deploy vLLM inference endpoint on SageMaker.

    This stack creates:
    1. IAM Role for SageMaker execution
    2. SageMaker Model (container + configuration)
    3. SageMaker Endpoint Configuration (instance type/count)
    4. SageMaker Endpoint (real-time inference)
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # =====================================================================
        # CONFIGURATION - UPDATE THESE PLACEHOLDERS
        # =====================================================================

        # Model Configuration
        model_name = "llama-3-70b-vllm"
        hf_model_id = "meta-llama/Llama-3-70b-hf"
        hf_token = None  # Set to "hf_..." if gated model, else None

        # Model artifacts (choose one):
        # Option 1: Download from HuggingFace Hub at runtime
        model_data_url = None

        # Option 2: Use pre-downloaded model from S3
        # model_data_url = "s3://my-bucket/models/llama-3-70b.tar.gz"

        # Inference Configuration
        instance_type = "ml.p4d.24xlarge"  # 8x A100 80GB
        initial_instance_count = 1
        tensor_parallel_size = 8

        # vLLM Settings
        max_model_len = 4096
        gpu_memory_utilization = 0.95
        enable_prefix_caching = True

        # Container Image
        # Find latest at: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
        vllm_image_uri = f"763104351884.dkr.ecr.{self.region}.amazonaws.com/vllm:0.11.2-sagemaker"

        # Endpoint Configuration
        endpoint_name = f"{model_name}-endpoint"

        # =====================================================================
        # RESOURCE CREATION
        # =====================================================================

        # 1. Create IAM Role for SageMaker
        # ---------------------------------
        # This role allows SageMaker to:
        #   - Pull Docker images from ECR
        #   - Download model artifacts from S3
        #   - Write logs to CloudWatch
        #   - Access other AWS services as needed

        execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description=f"Execution role for {model_name} SageMaker endpoint",
            managed_policies=[
                # Provides full SageMaker access (can be scoped down for production)
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )

        # Add inline policy for ECR access (to pull DLC images)
        execution_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=["*"],  # ECR images
            )
        )

        # Add inline policy for S3 access (to download model artifacts)
        if model_data_url and model_data_url.startswith("s3://"):
            execution_role.add_to_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["s3:GetObject", "s3:ListBucket"],
                    resources=[
                        f"arn:aws:s3:::{model_data_url.split('/')[2]}/*",  # Bucket contents
                        f"arn:aws:s3:::{model_data_url.split('/')[2]}",    # Bucket itself
                    ],
                )
            )

        # 2. Create Environment Variables for vLLM Container
        # ---------------------------------------------------
        env_vars = self._create_vllm_env_vars(
            model_id=hf_model_id if not model_data_url else "/opt/ml/model",
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            hf_token=hf_token,
        )

        # 3. Create SageMaker Model
        # --------------------------
        # Defines the container image and environment configuration

        cfn_model = sagemaker.CfnModel(
            self,
            "VllmModel",
            model_name=model_name,
            execution_role_arn=execution_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=vllm_image_uri,
                model_data_url=model_data_url,  # None = download from Hub
                environment=env_vars,
            ),
        )

        # 4. Create Endpoint Configuration
        # ---------------------------------
        # Specifies instance type, count, and model variant

        endpoint_config_name = f"{model_name}-config"

        cfn_endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "VllmEndpointConfig",
            endpoint_config_name=endpoint_config_name,
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="AllTraffic",  # Name for this variant
                    model_name=cfn_model.model_name,
                    instance_type=instance_type,
                    initial_instance_count=initial_instance_count,
                    initial_variant_weight=1.0,  # Traffic weight (1.0 = 100%)

                    # Container startup health check settings
                    # Model loading can take 10-30 minutes for large models
                    container_startup_health_check_timeout_in_seconds=1800,  # 30 minutes
                )
            ],
        )

        # Ensure model is created before endpoint config
        cfn_endpoint_config.add_dependency(cfn_model)

        # 5. Create SageMaker Endpoint
        # -----------------------------
        # Launches instances and deploys the model

        cfn_endpoint = sagemaker.CfnEndpoint(
            self,
            "VllmEndpoint",
            endpoint_name=endpoint_name,
            endpoint_config_name=cfn_endpoint_config.endpoint_config_name,
        )

        # Ensure endpoint config is created before endpoint
        cfn_endpoint.add_dependency(cfn_endpoint_config)

        # =====================================================================
        # CLOUDFORMATION OUTPUTS
        # =====================================================================
        # These values are displayed after deployment and can be referenced
        # by other stacks or scripts

        CfnOutput(
            self,
            "EndpointName",
            value=cfn_endpoint.endpoint_name,
            description="SageMaker endpoint name for inference",
            export_name=f"{construct_id}-EndpointName",
        )

        CfnOutput(
            self,
            "ExecutionRoleArn",
            value=execution_role.role_arn,
            description="IAM role ARN for SageMaker execution",
            export_name=f"{construct_id}-ExecutionRoleArn",
        )

        CfnOutput(
            self,
            "ModelName",
            value=cfn_model.model_name,
            description="SageMaker model name",
            export_name=f"{construct_id}-ModelName",
        )

        # Output inference example
        CfnOutput(
            self,
            "InferenceExample",
            value=self._generate_inference_example(endpoint_name),
            description="Example Python code to invoke endpoint",
        )

    def _create_vllm_env_vars(
        self,
        model_id: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        enable_prefix_caching: bool,
        hf_token: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Create environment variable dictionary for vLLM container.

        All variables use SM_VLLM_ prefix and are transformed to CLI args
        by the container's entry point script.

        Args:
            model_id: HuggingFace model ID or /opt/ml/model
            tensor_parallel_size: Number of GPUs
            max_model_len: Max sequence length
            gpu_memory_utilization: GPU memory fraction (0.0-1.0)
            enable_prefix_caching: Enable prefix caching
            hf_token: HuggingFace token (optional)

        Returns:
            Environment variable dictionary
        """
        env = {
            "SM_VLLM_MODEL": model_id,
            "SM_VLLM_TENSOR_PARALLEL_SIZE": str(tensor_parallel_size),
            "SM_VLLM_MAX_MODEL_LEN": str(max_model_len),
            "SM_VLLM_GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),
            "SM_VLLM_ENABLE_PREFIX_CACHING": "true" if enable_prefix_caching else "false",
            "SM_VLLM_DTYPE": "bfloat16",
            "SM_VLLM_TRUST_REMOTE_CODE": "true",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }

        if hf_token:
            env["HF_TOKEN"] = hf_token

        return env

    def _generate_inference_example(self, endpoint_name: str) -> str:
        """Generate example Python code for invoking the endpoint."""
        return f"""
import boto3
import json

client = boto3.client('sagemaker-runtime')

response = client.invoke_endpoint(
    EndpointName='{endpoint_name}',
    ContentType='application/json',
    Body=json.dumps({{
        'messages': [{{'role': 'user', 'content': 'Hello!'}}],
        'max_tokens': 100
    }})
)

result = json.loads(response['Body'].read())
print(result['choices'][0]['message']['content'])
"""


# =============================================================================
# CDK APP DEFINITION
# =============================================================================

app = App()

SageMakerVllmStack(
    app,
    "SageMakerVllmStack",
    description="Deploy vLLM inference endpoint for Foundation Models",

    # Optional: Specify AWS account and region
    # env={
    #     "account": "123456789012",  # Your AWS account ID
    #     "region": "us-east-1",
    # },

    # Optional: Add tags to all resources
    tags={
        "Project": "FoundationModelInference",
        "Framework": "vLLM",
        "ManagedBy": "CDK",
    },
)

app.synth()


# =============================================================================
# DEPLOYMENT INSTRUCTIONS
# =============================================================================

"""
FIRST TIME SETUP (one-time per account/region):
-----------------------------------------------

1. Install AWS CDK CLI:
   npm install -g aws-cdk

2. Verify installation:
   cdk --version

3. Bootstrap CDK (creates S3 bucket for CDK assets):
   cdk bootstrap aws://ACCOUNT-ID/REGION

   Example:
   cdk bootstrap aws://123456789012/us-east-1


DEVELOPMENT WORKFLOW:
--------------------

1. Install Python dependencies:
   pip install aws-cdk-lib constructs

2. Preview CloudFormation template:
   cdk synth

   This outputs the generated CloudFormation template to cdk.out/

3. Compare local changes with deployed stack:
   cdk diff

   Shows what resources will be added/modified/removed

4. Deploy stack:
   cdk deploy

   This will:
   - Create IAM role (~30 seconds)
   - Create SageMaker Model (~30 seconds)
   - Create Endpoint Config (~30 seconds)
   - Create Endpoint and launch instances (~10-30 minutes)

   You'll be prompted to approve IAM changes before deployment.

5. View stack outputs:
   After deployment completes, outputs are displayed:
   - EndpointName: Use this for inference
   - ExecutionRoleArn: IAM role created
   - InferenceExample: Copy-paste Python code

6. Test inference:
   Copy the InferenceExample output and run it:

   import boto3
   import json

   client = boto3.client('sagemaker-runtime', region_name='us-east-1')
   response = client.invoke_endpoint(
       EndpointName='llama-3-70b-vllm-endpoint',
       ContentType='application/json',
       Body=json.dumps({
           'messages': [{'role': 'user', 'content': 'Explain AI'}],
           'max_tokens': 256
       })
   )
   result = json.loads(response['Body'].read())
   print(result['choices'][0]['message']['content'])


CLEANUP:
-------

1. Destroy all resources:
   cdk destroy

   This will:
   - Delete the endpoint (stops billing)
   - Delete endpoint config
   - Delete model
   - Delete IAM role

   WARNING: This is irreversible. The endpoint and its configuration
   will be permanently deleted.

2. Confirm deletion when prompted.


COST ESTIMATION:
---------------

Endpoint costs (while running):
  - ml.p4d.24xlarge: ~$32.77/hour
  - ml.g5.12xlarge: ~$7.09/hour
  - ml.g5.2xlarge: ~$1.52/hour

CDK infrastructure (negligible):
  - S3 storage for CDK assets: <$1/month
  - CloudFormation: Free

IMPORTANT: The endpoint runs 24/7 until deleted. Remember to run
`cdk destroy` when not in use to avoid charges.


ADVANCED CONFIGURATION:
----------------------

# Example 1: Multi-AZ Deployment (High Availability)
production_variants=[
    sagemaker.CfnEndpointConfig.ProductionVariantProperty(
        variant_name="Primary",
        model_name=cfn_model.model_name,
        instance_type="ml.p4d.24xlarge",
        initial_instance_count=2,  # 2 instances across AZs
        initial_variant_weight=1.0,
    )
]


# Example 2: A/B Testing with Multiple Model Variants
production_variants=[
    sagemaker.CfnEndpointConfig.ProductionVariantProperty(
        variant_name="ModelA",
        model_name=model_a.model_name,
        instance_type="ml.p4d.24xlarge",
        initial_instance_count=1,
        initial_variant_weight=0.8,  # 80% of traffic
    ),
    sagemaker.CfnEndpointConfig.ProductionVariantProperty(
        variant_name="ModelB",
        model_name=model_b.model_name,
        instance_type="ml.p4d.24xlarge",
        initial_instance_count=1,
        initial_variant_weight=0.2,  # 20% of traffic
    ),
]


# Example 3: Add Auto-Scaling (using L2 constructs)
from aws_cdk import aws_applicationautoscaling as appscaling

# After creating endpoint, add auto-scaling target
scaling_target = appscaling.ScalableTarget(
    self,
    "EndpointScalingTarget",
    service_namespace=appscaling.ServiceNamespace.SAGEMAKER,
    resource_id=f"endpoint/{cfn_endpoint.endpoint_name}/variant/AllTraffic",
    scalable_dimension="sagemaker:variant:DesiredInstanceCount",
    min_capacity=1,
    max_capacity=5,
)

# Add target tracking policy (scale based on invocations)
scaling_target.scale_to_track_metric(
    "InvocationScaling",
    target_value=1000.0,  # Target 1000 invocations per instance
    predefined_metric=appscaling.PredefinedMetric.SAGEMAKER_VARIANT_INVOCATIONS_PER_INSTANCE,
    scale_in_cooldown=Duration.minutes(5),
    scale_out_cooldown=Duration.minutes(1),
)


# Example 4: Add CloudWatch Alarms for Monitoring
from aws_cdk import aws_cloudwatch as cloudwatch

# Create alarm for high model latency
latency_alarm = cloudwatch.Alarm(
    self,
    "HighLatencyAlarm",
    metric=cloudwatch.Metric(
        namespace="AWS/SageMaker",
        metric_name="ModelLatency",
        dimensions_map={
            "EndpointName": cfn_endpoint.endpoint_name,
            "VariantName": "AllTraffic",
        },
        statistic="Average",
        period=Duration.minutes(5),
    ),
    threshold=5000,  # 5 seconds
    evaluation_periods=2,
    alarm_description="Alert when model latency exceeds 5 seconds",
)


# Example 5: Cross-Stack References
# Deploy model in one stack, endpoint in another

# In ModelStack:
self.model = sagemaker.CfnModel(...)

# In EndpointStack:
class EndpointStack(Stack):
    def __init__(self, scope, id, model_stack, **kwargs):
        super().__init__(scope, id, **kwargs)

        cfn_endpoint = sagemaker.CfnEndpoint(
            self,
            "Endpoint",
            endpoint_config_name=...,
        )

# In app:
model_stack = ModelStack(app, "ModelStack")
endpoint_stack = EndpointStack(app, "EndpointStack", model_stack=model_stack)
"""
