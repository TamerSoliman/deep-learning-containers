# Model Encryption at Rest

## S3 Bucket Encryption

```python
import boto3

s3 = boto3.client('s3')

# Enable default encryption
s3.put_bucket_encryption(
    Bucket='ml-models-bucket',
    ServerSideEncryptionConfiguration={
        'Rules': [{
            'ApplyServerSideEncryptionByDefault': {
                'SSEAlgorithm': 'aws:kms',
                'KMSMasterKeyID': 'arn:aws:kms:us-west-2:123456789012:key/abcd-1234',
            }
        }]
    }
)
```

## SageMaker Volume Encryption

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    # ... other params
    volume_kms_key='arn:aws:kms:us-west-2:123456789012:key/abcd-1234',
    output_kms_key='arn:aws:kms:us-west-2:123456789012:key/abcd-1234',
)
```

## EBS Volume Encryption (Endpoint Storage)

```python
model.deploy(
    instance_type='ml.g5.xlarge',
    volume_size=100,  # GB
    volume_kms_key='arn:aws:kms:us-west-2:123456789012:key/abcd-1234',
)
```
