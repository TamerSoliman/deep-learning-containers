# Adding Custom Dependencies to DLC Containers

## Python Packages

### requirements.txt Approach

```dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.8-gpu

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
```

```
# requirements.txt
langchain==0.1.20
pinecone-client==2.2.4
redis==5.0.1
faiss-gpu==1.7.4
sentence-transformers==2.3.1
```

### Conditional Dependencies

```dockerfile
# Install different packages based on GPU architecture
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      pip install faiss-gpu; \
    else \
      pip install faiss-cpu; \
    fi
```

## System Libraries

```dockerfile
# Add system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
```

## Custom Binaries

```dockerfile
# Add custom compiled binary
COPY ./custom_binary /usr/local/bin/
RUN chmod +x /usr/local/bin/custom_binary
```

## Model-Specific Dependencies

```dockerfile
# For Whisper (audio transcription)
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.8-gpu

RUN pip install --no-cache-dir \
    openai-whisper==20231117 \
    ffmpeg-python==0.2.0

# For Stable Diffusion
RUN pip install --no-cache-dir \
    diffusers==0.25.0 \
    accelerate==0.25.0 \
    safetensors==0.4.1
```
