# API Authentication Patterns

## AWS SigV4 Authentication

```python
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import requests

# Sign request with AWS credentials
session = boto3.Session()
credentials = session.get_credentials()

url = 'https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-endpoint/invocations'
request = AWSRequest(method='POST', url=url, data=payload)

SigV4Auth(credentials, 'sagemaker', 'us-west-2').add_auth(request)

# Send signed request
response = requests.post(url, headers=dict(request.headers), data=payload)
```

## IAM Role-based Access

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "AWS": "arn:aws:iam::123456789012:role/ApplicationRole"
    },
    "Action": "sagemaker:InvokeEndpoint",
    "Resource": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/my-endpoint"
  }]
}
```

## API Gateway + Lambda Authorizer

```python
import boto3

def lambda_handler(event, context):
    # Custom authorization logic
    token = event['authorizationToken']

    if validate_token(token):
        return generate_policy('user', 'Allow', event['methodArn'])
    else:
        return generate_policy('user', 'Deny', event['methodArn'])
```
