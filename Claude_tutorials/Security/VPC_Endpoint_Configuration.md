# Private VPC Endpoints for SageMaker

## Architecture

```
Internet → ❌ No public access
    │
VPC Endpoints → SageMaker Runtime (private)
    │
Application (in VPC) → Invoke endpoints securely
```

## Setup

### 1. Create VPC Endpoints

```python
import boto3

ec2 = boto3.client('ec2')

# SageMaker Runtime endpoint
response = ec2.create_vpc_endpoint(
    VpcId='vpc-12345678',
    ServiceName='com.amazonaws.us-west-2.sagemaker.runtime',
    VpcEndpointType='Interface',
    SubnetIds=['subnet-abc', 'subnet-def'],
    SecurityGroupIds=['sg-12345'],
    PrivateDnsEnabled=True,
)
```

### 2. Security Group Rules

```
Inbound Rules:
- Type: HTTPS
- Port: 443
- Source: Application security group

Outbound Rules:
- Type: HTTPS
- Port: 443
- Destination: 0.0.0.0/0 (for model downloads)
```

### 3. Deploy in VPC Mode

```python
from sagemaker import Model

model = Model(
    image_uri="...",
    role=role,
    vpc_config={
        'Subnets': ['subnet-abc', 'subnet-def'],
        'SecurityGroupIds': ['sg-endpoint-12345'],
    }
)
```
