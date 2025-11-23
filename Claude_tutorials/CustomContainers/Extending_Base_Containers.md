# Extending AWS DLC Base Containers

## Why Extend Containers?

- Add custom Python packages
- Include proprietary libraries
- Add custom model handlers
- Integrate monitoring agents (DataDog, New Relic)
- Add preprocessing/postprocessing logic

## Method 1: Extend vLLM Container

```dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.11.2-gpu-sagemaker

# Add custom Python packages
RUN pip install --no-cache-dir \
    langchain==0.1.0 \
    pinecone-client==2.2.4 \
    redis==5.0.1

# Add custom preprocessing script
COPY custom_preprocessor.py /opt/ml/code/

# Add monitoring agent
RUN curl -L https://github.com/DataDog/dd-trace-py/archive/v1.20.0.tar.gz | tar xz
RUN cd dd-trace-py-1.20.0 && pip install .

# Set environment variables
ENV DD_SERVICE="vllm-inference" \
    DD_ENV="production"

# Custom entrypoint wrapper
COPY custom_entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/custom_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/custom_entrypoint.sh"]
```

### Custom Entrypoint (custom_entrypoint.sh)

```bash
#!/bin/bash

# Initialize DataDog
ddtrace-run python /usr/local/bin/start_monitoring.py &

# Start original vLLM entrypoint
exec /usr/local/bin/dockerd_entrypoint.sh
```

## Method 2: Extend HuggingFace Training Container

```dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.8-transformers4.56-gpu-py312-cu129-ubuntu22.04

# Add custom training libraries
RUN pip install --no-cache-dir \
    wandb==0.16.0 \
    deepspeed==0.13.0 \
    flash-attn==2.5.0

# Add custom dataset loaders
COPY custom_datasets/ /opt/ml/code/datasets/

# Add W&B configuration
ENV WANDB_API_KEY="" \
    WANDB_PROJECT="llm-finetuning"
```

## Method 3: Add Custom Model Handler

```python
# custom_handler.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CustomModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def initialize(self, context):
        """Load model on container startup"""
        model_dir = context.system_properties.get("model_dir")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def preprocess(self, data):
        """Custom preprocessing"""
        text = data[0].get("text")

        # Add custom prompt template
        prompt = f"### Instruction:\n{text}\n\n### Response:\n"

        return self.tokenizer(prompt, return_tensors="pt")

    def inference(self, inputs):
        """Run inference"""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
            )
        return outputs

    def postprocess(self, outputs):
        """Custom postprocessing"""
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part
        response = text.split("### Response:\n")[-1]

        return [{"generated_text": response}]

_service = CustomModelHandler()

def handle(data, context):
    if not _service.model:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
```

### Dockerfile with Custom Handler

```dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:2.8-gpu

COPY custom_handler.py /opt/ml/model/code/inference.py

ENV SAGEMAKER_PROGRAM=inference.py
```

## Build and Push

```bash
# Build
docker build -t custom-vllm:latest -f Dockerfile.custom .

# Tag
docker tag custom-vllm:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/custom-vllm:latest

# Push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/custom-vllm:latest
```

## Deploy Custom Container

```python
from sagemaker import Model

model = Model(
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/custom-vllm:latest",
    model_data="s3://bucket/model.tar.gz",
    role=role,
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
)
```
