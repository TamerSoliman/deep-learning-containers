# Annotated Dockerfile: PyTorch 2.8.0 GPU Training Container

**Source**: `pytorch/training/docker/2.8/py3/cu129/Dockerfile.gpu`

**Purpose**: This container is optimized for distributed training of Foundation Models (LLMs, Vision, Multimodal) on multi-GPU and multi-node AWS infrastructure. It provides state-of-the-art optimizations for large-scale training workloads.

---

## SECTION 1: Build Arguments - Version Pinning

```dockerfile
ARG PYTHON=python3
ARG PYTHON_VERSION=3.12.10
ARG PYTHON_SHORT_VERSION=3.12
ARG PYTORCH_VERSION=2.8.0
ARG TORCHTNT_VERSION=0.2.4
ARG TORCHAUDIO_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHDATA_VERSION=0.11.0

ARG GDRCOPY_VERSION=2.5.1
ARG TE_VERSION=2.5
ARG FLASH_ATTN_VERSION=2.8.3
```

### WHAT:
- Defines build-time arguments for all critical library versions
- Pins specific versions to ensure reproducibility and compatibility

### WHY:
- **Version Pinning**: Ensures that builds are reproducible across time - critical for production ML workflows
- **Compatibility Matrix**: PyTorch 2.8.0 is specifically matched with CUDA 12.9, Flash Attention 2.8.3, and Transformer Engine 2.5
- **Build Flexibility**: ARG allows these to be overridden at build time if needed

### HOW IT'S USED:
- These ARGs are referenced throughout the Dockerfile using `${VARIABLE_NAME}` syntax
- For example: `pip install torch==${PYTORCH_VERSION}` uses the pinned PyTorch version

### KEY OPTIMIZATIONS:
- **GDRCOPY (2.5.1)**: GPU Direct RDMA Copy library - enables direct memory access between GPUs without CPU involvement, critical for multi-GPU training
- **TE (Transformer Engine 2.5)**: NVIDIA's library for mixed-precision training of transformers, provides FP8 training for A100/H100 GPUs
- **FLASH_ATTN (2.8.3)**: Memory-efficient attention implementation, reduces memory usage by 2-3x for transformer models

---

## SECTION 2: Base Image - Foundation Layer

```dockerfile
FROM public.ecr.aws/deep-learning-containers/base:12.9.1-gpu-py312-ubuntu22.04-ec2 AS common
# base has EFA, PYTHON and CUDA 12.9
```

### WHAT:
- Multi-stage build starting from a pre-built base image
- Base image already contains: CUDA 12.9, Python 3.12, EFA drivers

### WHY:
- **Layer Reuse**: The base image is shared across multiple DLC types, reducing build time and storage
- **Pre-installed Infrastructure**: EFA (Elastic Fabric Adapter) drivers are complex to install - having them pre-installed ensures consistency
- **CUDA Version Lock**: CUDA 12.9 is specifically chosen for PyTorch 2.8.0 compatibility

### HOW IT OPTIMIZES:
- **EFA (Elastic Fabric Adapter)**: AWS's custom network interface for HPC workloads
  - Enables 100 Gbps networking with ultra-low latency (~20 microseconds)
  - Critical for multi-node distributed training (e.g., training Llama 70B across 8+ nodes)
  - Works with NCCL to provide optimal inter-node GPU communication
- **CUDA 12.9**: Latest CUDA version with:
  - Improved kernel launch performance
  - Better multi-GPU synchronization primitives
  - Support for H100's FP8 tensor cores

### STAGE NAME:
- `AS common` - This stage is shared by both EC2 and SageMaker variants built later

---

## SECTION 3: Environment Configuration - Runtime Paths

```dockerfile
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV EFA_PATH="/opt/amazon/efa"
ENV OPEN_MPI_PATH="/opt/amazon/openmpi"
```

### WHAT:
- Sets up critical environment variables for CUDA, EFA, and OpenMPI paths

### WHY:
- **CUDA_HOME**: Required by many GPU libraries (NCCL, Flash Attention) to locate CUDA toolkit during runtime and compilation
- **PATH Addition**: Makes CUDA binaries (nvcc, nvidia-smi) available without full paths
- **EFA_PATH**: Points to EFA drivers and libraries for network acceleration
- **OPEN_MPI_PATH**: OpenMPI is the MPI implementation used for multi-node training coordination

### HOW IT'S USED IN TRAINING:
- When you run `torchrun` or `accelerate launch` for distributed training:
  - OpenMPI coordinates process launch across nodes
  - EFA provides the network fabric
  - CUDA libraries are automatically found via CUDA_HOME

---

## SECTION 4: Python Environment Configuration

```dockerfile
# Python won't try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
```

### WHAT:
- Configures Python runtime behavior and locale settings

### WHY:
- **PYTHONDONTWRITEBYTECODE=1**:
  - Prevents Python from creating .pyc bytecode cache files
  - Reduces container image size
  - Avoids permission issues in read-only filesystems (common in containerized environments)
- **PYTHONUNBUFFERED=1**:
  - Critical for logging in containerized training jobs
  - Without this, logs are buffered and you won't see them in real-time
  - Essential for debugging long-running training jobs
- **UTF-8 Encoding**:
  - Ensures consistent text handling across different systems
  - Important for processing multilingual datasets

### IMPACT ON TRAINING:
- Real-time log visibility allows monitoring training progress, detecting errors early
- No bytecode caching means the container is truly stateless

---

## SECTION 5: CUDA Compilation Flags

```dockerfile
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
```

### WHAT:
- Sets compilation flags for NVCC (NVIDIA CUDA Compiler) when building CUDA extensions

### WHY:
- **-Xfatbin -compress-all**: Compresses CUDA binary (fatbin) files
- Reduces the size of compiled CUDA kernels
- Many PyTorch extensions compile custom CUDA kernels at install time (e.g., Flash Attention, apex)

### HOW:
- When you `pip install` a package with CUDA code, PyTorch uses `torch.utils.cpp_extension` which respects this environment variable
- Smaller binaries = smaller container image (can save hundreds of MBs)

---

## SECTION 6: Library Paths for Distributed Training

```dockerfile
ENV PATH="${OPEN_MPI_PATH}/bin:${EFA_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
```

### WHAT:
- Adds multiple library paths to PATH and LD_LIBRARY_PATH

### WHY - Critical for Multi-GPU/Multi-Node Training:
- **LD_LIBRARY_PATH** tells the dynamic linker where to find shared libraries (.so files)
- Let's break down each path:

#### `/opt/amazon/ofi-nccl/lib/`:
- **OFI (OpenFabrics Interface)**: Industry-standard fabric abstraction
- **OFI-NCCL Plugin**: Allows NCCL to use EFA for inter-node communication
- **Impact**: Without this, NCCL would use TCP instead of EFA, reducing bandwidth from 100 Gbps to ~10 Gbps

#### `/opt/amazon/openmpi/lib/`:
- **OpenMPI Libraries**: Required for MPI-based distributed training
- **Used by**: torchrun, Horovod, DeepSpeed
- **Impact**: Enables process coordination across multiple nodes

#### `/opt/amazon/efa/lib/`:
- **EFA Libraries**: Low-level EFA driver libraries
- **Provides**: Direct network access primitives used by OFI-NCCL

#### `/usr/local/cuda/lib64`:
- **CUDA Runtime Libraries**: cuDNN, cuBLAS, NCCL
- **Impact**: Core libraries for GPU operations

### HOW IT WORKS IN MULTI-NODE TRAINING:
When you run distributed training across 4 nodes with 8 GPUs each:
1. OpenMPI launches processes on each node
2. PyTorch's distributed module initializes NCCL
3. NCCL discovers the OFI-NCCL plugin via LD_LIBRARY_PATH
4. OFI-NCCL uses EFA libraries for inter-node GPU-to-GPU communication
5. Result: AllReduce operations (gradient synchronization) happen at near-PCIe speeds even across nodes

---

## SECTION 7: System Dependencies

```dockerfile
RUN apt-get update \
 && apt-get -y upgrade --only-upgrade systemd \
 && apt-get install -y --allow-change-held-packages --no-install-recommends \
    libgl1-mesa-glx \
    build-essential \
    ca-certificates \
    zlib1g-dev \
    openssl \
    python3-dev \
    pkg-config \
    check \
    llvm \
    xz-utils \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean
```

### WHAT:
- Installs system-level build tools and libraries

### WHY - Each Package:
- **build-essential**: gcc, g++, make - required to compile C/C++ extensions
- **python3-dev**: Python header files - required to build Python C extensions
- **libgl1-mesa-glx**: OpenGL libraries - required by OpenCV for computer vision tasks
- **llvm**: LLVM compiler infrastructure - used by Triton (PyTorch's JIT compiler)
- **ca-certificates**: SSL certificates - required for HTTPS downloads (model weights from S3/HuggingFace)

### HOW IT'S USED:
- When installing Flash Attention: requires gcc/g++ to compile CUDA kernels
- When loading datasets: OpenCV (which needs libgl1) for image preprocessing
- When downloading models: SSL certificates for secure connections

---

## SECTION 8: Python Scientific Stack

```dockerfile
RUN pip install --no-cache-dir \
    cython \
    cryptography \
    pyOpenSSL \
    pybind11 \
    mkl \
    mkl-include \
    parso \
    typing \
    charset-normalizer \
    packaging \
    PyYAML \
    numpy \
    scipy \
    click \
    psutil \
    ipython \
    ipykernel \
    pillow \
    h5py \
    fsspec \
    "idna>=3.7" \
    "tqdm>=4.66.3" \
    "requests>=2.32.0" \
    "setuptools>=70.0.0" \
    "urllib3>=2.5.0" \
    ninja \
    opencv-python==4.11.0.86 \
    mpi4py \
    jinja2>=3.1.6 \
    tornado>=6.5.1
```

### WHAT:
- Installs the Python scientific computing ecosystem

### WHY - Key Libraries for ML Training:
- **numpy/scipy**: Numerical computing - used by every ML library
- **mkl/mkl-include**: Intel Math Kernel Library - optimized BLAS/LAPACK operations
  - Provides 2-10x speedup for CPU operations (data preprocessing, metrics computation)
- **cython/pybind11**: Python-C++ binding tools - used by many libraries for performance
- **ninja**: Fast build system - speeds up compilation of CUDA extensions
- **mpi4py**: Python MPI bindings - enables Python scripts to use MPI directly
- **opencv-python**: Computer vision - image/video preprocessing for vision models
- **h5py**: HDF5 file format - efficient storage for large datasets
- **fsspec**: Filesystem abstraction - allows reading from S3, GCS, etc. with the same API

### SECURITY UPDATES:
- **"urllib3>=2.5.0"**, **"requests>=2.32.0"**: Patches known CVEs
- **"jinja2>=3.1.6"**: Patches template injection vulnerabilities

---

## SECTION 9: PyTorch Installation

```dockerfile
RUN pip install --no-cache-dir -U torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/cu129 \
    && pip install --no-cache-dir -U torchtnt==${TORCHTNT_VERSION} \
    torchdata==${TORCHDATA_VERSION} \
    triton \
    s3torchconnector \
    fastai \
    accelerate \
    spacy==3.8.7 \
    thinc==8.3.4 \
    blis \
    numpy \
 && pip uninstall -y dataclasses
```

### WHAT:
- Installs PyTorch and its ecosystem

### WHY - Breaking Down the Components:

#### Core PyTorch:
- **torch 2.8.0**: Core deep learning framework
- **--index-url https://download.pytorch.org/whl/cu129**:
  - Gets the CUDA 12.9 specific build (not CPU-only or other CUDA versions)
  - This is pre-compiled with cuDNN 9.x and NCCL 2.x
- **torchvision**: Computer vision models and utilities (ViT, ResNet, data augmentation)
- **torchaudio**: Audio processing (for speech models like Whisper)

#### Advanced Training Tools:
- **torchtnt (TorchTNT 0.2.4)**: PyTorch Training Toolkit
  - Provides high-level training loops
  - Handles checkpointing, logging, distributed training boilerplate
- **accelerate**: HuggingFace's distributed training library
  - Simplifies multi-GPU/multi-node training
  - Auto-detects hardware and configures optimal settings
  - Used by many LLM training scripts

#### Data Loading & Storage:
- **torchdata**: Next-gen DataLoader with better performance
- **s3torchconnector**: Direct S3 dataset loading
  - Streams data from S3 without downloading entire dataset
  - Critical for training on huge datasets (terabytes)
  - Example: Load ImageNet directly from S3 during training

#### Compiler & Optimization:
- **triton**: OpenAI's GPU programming language
  - PyTorch 2.x uses Triton for `torch.compile()`
  - Compiles Python code to optimized GPU kernels
  - Can provide 1.5-2x speedup for transformers

#### Cleanup:
- **`pip uninstall -y dataclasses`**:
  - `dataclasses` is built into Python 3.7+
  - Removing the package avoids conflicts

---

## SECTION 10: Flash Attention Installation

```dockerfile
ENV NVTE_FRAMEWORK=pytorch

RUN curl -LO https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
    && pip install flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-build-isolation \
    && rm flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

### WHAT:
- Installs pre-compiled Flash Attention 2.8.3 wheel

### WHY - Critical for LLM Training:
- **Flash Attention**: Memory-efficient attention implementation by Tri Dao (Stanford/Princeton)
- **Impact on Training**:
  - Standard attention: O(N²) memory for sequence length N
  - Flash Attention: O(N) memory
  - Example: For 4K sequence length, uses ~4x less memory
  - Enables training longer sequences or larger batch sizes

### HOW:
- **Pre-compiled Wheel**: Building Flash Attention from source takes 10-20 minutes
- **Wheel Naming**: `cu12torch2.8cxx11abiTRUE` indicates:
  - CUDA 12.x compatible
  - PyTorch 2.8 compatible
  - C++11 ABI enabled (matches PyTorch's build)
- **--no-build-isolation**: Trusts the pre-built binary, speeds up installation

### USAGE IN TRAINING:
```python
from flash_attn import flash_attn_func
# Automatically used by transformers library when available
# 2-3x faster training for LLMs like Llama, Mistral
```

---

## SECTION 11: Transformer Engine Installation

```dockerfile
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/TransformerEngine.git@release_v${TE_VERSION} --no-build-isolation
```

### WHAT:
- Installs NVIDIA Transformer Engine 2.5 from source (specific release tag)

### WHY - FP8 Training for H100:
- **Transformer Engine (TE)**: NVIDIA's library for mixed-precision transformer training
- **Key Feature - FP8 Support**:
  - H100 GPUs have FP8 tensor cores (not available on A100)
  - FP8 (8-bit floating point): 2x faster than FP16, with minimal accuracy loss
  - Training speedup: 1.5-2x for large transformers (e.g., 70B parameter models)

### HOW IT WORKS:
- Provides drop-in replacements for PyTorch layers:
  ```python
  import transformer_engine.pytorch as te
  # Replace nn.Linear with te.Linear for automatic FP8 training
  layer = te.Linear(4096, 4096)
  ```
- Automatically handles:
  - Scaling factors for FP8 precision
  - Gradient accumulation in higher precision
  - Mixed FP8/FP16/FP32 computation

### NVTE_FRAMEWORK=pytorch:
- TE supports multiple frameworks (JAX, PyTorch)
- Setting this env var avoids installing unnecessary dependencies

---

## SECTION 12: GDRCopy Installation

```dockerfile
RUN cd /tmp \
 && git clone https://github.com/NVIDIA/gdrcopy.git -b v${GDRCOPY_VERSION} \
 && cd gdrcopy \
 && sed -ie '13s@$@ -L $(CUDA)/lib64/stubs@' tests/Makefile \
 && CUDA=${CUDA_HOME} make install \
 && rm -rf /tmp/gdrcopy
```

### WHAT:
- Compiles and installs GDRCopy (GPU Direct RDMA Copy) from source

### WHY - Critical for Multi-GPU Communication:
- **GPU Direct RDMA**: Allows GPUs to directly access each other's memory without CPU involvement
- **Use Case**: Multi-GPU training on a single node (e.g., 8x A100 on p4d.24xlarge)

### HOW IT OPTIMIZES:
**Without GDRCopy (Traditional Path)**:
1. GPU 0 copies data to CPU memory
2. CPU copies data to GPU 1's memory
3. Throughput: ~50 GB/s (limited by PCIe to CPU)

**With GDRCopy + NVLink**:
1. GPU 0 directly writes to GPU 1's memory via NVLink
2. Throughput: ~600 GB/s (NVLink v4 on H100)
3. 12x faster!

### BUILD MODIFICATION:
- `sed -ie '13s@$@ -L $(CUDA)/lib64/stubs@'`:
  - Fixes build issue where test binaries need CUDA driver library
  - Points to stub library in CUDA toolkit
  - Only needed for building, not runtime

### USED BY:
- NCCL (NVIDIA Collective Communications Library)
- SageMaker Distributed Data Parallel
- Any multi-GPU gradient synchronization

---

## SECTION 13: Multi-Stage Build - EC2 Variant

```dockerfile
FROM common AS ec2

ARG PYTHON

WORKDIR /

COPY dockerd_entrypoint.sh /usr/local/bin/dockerd_entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd_entrypoint.sh

COPY setup_oss_compliance.sh setup_oss_compliance.sh
RUN bash setup_oss_compliance.sh ${PYTHON} && rm setup_oss_compliance.sh

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["bash", "-m", "dockerd_entrypoint.sh"]
CMD ["/bin/bash"]
```

### WHAT:
- Creates EC2-specific variant from the common base

### WHY EC2 vs SageMaker Variants:
- **EC2 Container**: Used when you manually manage instances (EC2, EKS)
- **SageMaker Container**: Used in SageMaker Training Jobs (managed service)
- They share 95% of the image (the `common` stage), only differ in:
  - Entry point scripts
  - SageMaker-specific libraries (sagemaker-training SDK)

### ENTRYPOINT:
- **dockerd_entrypoint.sh**: Initialization script for EC2 deployments
  - Likely handles: CUDA compatibility checks, EFA setup, telemetry

### WHY `CMD ["/bin/bash"]`:
- Default command if user doesn't specify one
- Allows interactive use: `docker run -it <image> /bin/bash`

---

## SECTION 14: Multi-Stage Build - SageMaker Variant

```dockerfile
FROM common AS sagemaker

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main

ARG PYTHON

WORKDIR /

# Install SM packages
RUN pip install --no-cache-dir -U \
    "awscli<1.42.50" \
    "boto3<1.40.50" \
    smclarify \
    "sagemaker>=2" \
    sagemaker-experiments \
    sagemaker-pytorch-training \
    sagemaker-training
```

### WHAT:
- Creates SageMaker-specific variant with SageMaker SDK integration

### WHY - SageMaker Integration:
- **sagemaker-training**: Core library that handles SageMaker Training Job lifecycle
  - Reads hyperparameters from `/opt/ml/input/config/hyperparameters.json`
  - Loads training data from `/opt/ml/input/data/`
  - Saves model to `/opt/ml/model/`
  - Streams logs to CloudWatch

- **sagemaker-pytorch-training**: PyTorch-specific integration
  - Automatically configures distributed training based on instance count
  - Sets up NCCL environment variables
  - Handles multi-node training coordination

### SAGEMAKER_TRAINING_MODULE:
- When SageMaker starts a training job, it runs:
  ```bash
  python -m sagemaker_pytorch_container.training:main
  ```
- This module:
  1. Reads job configuration
  2. Sets up distributed training environment
  3. Executes user's training script
  4. Handles checkpointing and logging

### VERSION CONSTRAINTS:
- `"awscli<1.42.50"`, `"boto3<1.40.50"`:
  - Workaround for dependency conflicts with smclarify
  - These versions are tested and known to work together

---

## SECTION 15: Additional SageMaker Packages

```dockerfile
# Install extra packages
RUN pip install --no-cache-dir -U \
    bokeh \
    imageio \
    numba \
    pandas \
    plotly \
    shap \
    scikit-learn \
    seaborn \
    cloudpickle
```

### WHAT:
- Installs additional data science and visualization packages

### WHY - Common in ML Workflows:
- **pandas**: Data manipulation - used in data preprocessing, metrics computation
- **scikit-learn**: Traditional ML algorithms - often used for baseline models, feature engineering
- **shap**: Model interpretability - explains model predictions
- **plotly/seaborn/bokeh**: Visualization - for training metrics, data exploration
- **numba**: JIT compiler for Python - can accelerate custom data preprocessing
- **cloudpickle**: Enhanced pickling - better serialization for complex Python objects

### USAGE IN SAGEMAKER:
- SageMaker notebooks often use these for experimentation
- Training scripts may use pandas for data preprocessing
- Model evaluation scripts use visualization libraries

---

## SECTION 16: SageMaker Entry Point

```dockerfile
# Copy workaround script for incorrect hostname
COPY changehostname.c /
COPY start_with_right_hostname.sh /usr/local/bin/start_with_right_hostname.sh
RUN chmod +x /usr/local/bin/start_with_right_hostname.sh

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]
```

### WHAT:
- Sets up SageMaker-specific entry point with hostname workaround

### WHY - Hostname Fix:
- **Problem**: In multi-node training, nodes need to discover each other via hostnames
- **Issue**: Sometimes the container's internal hostname doesn't match the network hostname
- **Solution**: `start_with_right_hostname.sh` fixes this before training starts
  - Uses `changehostname.c` (compiled C program) to update hostname
  - Ensures NCCL and MPI can correctly resolve node addresses

### HOW IT AFFECTS DISTRIBUTED TRAINING:
When you launch a 4-node training job:
1. SageMaker assigns each node a hostname (e.g., `algo-1`, `algo-2`, etc.)
2. `start_with_right_hostname.sh` ensures container sees correct hostname
3. NCCL uses hostnames to establish inter-node connections
4. Training proceeds with proper multi-node communication

### ENTRYPOINT vs CMD:
- **ENTRYPOINT**: Always runs (the startup script)
- **CMD**: Default argument to ENTRYPOINT (can be overridden by user)
- When SageMaker starts training, it overrides CMD with your training script

---

## Summary: Container Lifecycle

### For EC2/EKS Deployment:
```bash
docker run <image>
  ↓
dockerd_entrypoint.sh (CUDA setup, telemetry)
  ↓
User's training script
```

### For SageMaker Deployment:
```bash
SageMaker Training Job starts
  ↓
start_with_right_hostname.sh (hostname fix, EFA setup)
  ↓
sagemaker_pytorch_container.training:main
  ↓
  - Reads /opt/ml/input/config/hyperparameters.json
  - Sets NCCL env vars for multi-node
  - Discovers training instances via SageMaker API
  ↓
User's training script (with distributed setup already configured)
  ↓
Model saved to /opt/ml/model/
```

---

## Key Optimization Summary

| Optimization | Purpose | Impact on LLM Training |
|--------------|---------|------------------------|
| **Flash Attention 2.8.3** | Memory-efficient attention | 2-3x memory reduction, enables longer sequences |
| **Transformer Engine 2.5** | FP8 mixed precision | 1.5-2x speedup on H100 GPUs |
| **EFA + NCCL + OFI** | Inter-node GPU communication | 100 Gbps bandwidth, <20μs latency |
| **GDRCopy** | Intra-node GPU communication | 600 GB/s with NVLink (12x faster than CPU path) |
| **s3torchconnector** | Direct S3 streaming | Stream TB-scale datasets without local storage |
| **Triton (torch.compile)** | JIT compilation | 1.5-2x speedup for transformer models |

---

## Real-World Training Example

Training Llama 3 70B on 32x H100 (4 nodes × 8 GPUs):

1. **Data Loading**: `s3torchconnector` streams training data from S3
2. **Forward Pass**:
   - Transformer Engine uses FP8 on H100 (2x faster)
   - Flash Attention reduces memory by 3x
3. **Backward Pass**: Gradients computed in mixed precision
4. **Gradient Synchronization**:
   - Within each node: NCCL uses GDRCopy + NVLink (600 GB/s)
   - Across nodes: NCCL uses OFI-NCCL + EFA (100 Gbps)
   - AllReduce synchronizes 140 billion parameters in <1 second
5. **Checkpointing**: Model saved to `/opt/ml/model/` → uploaded to S3

**Result**: This container's optimizations enable training 70B models at near-linear scaling across multiple nodes, reducing training time from weeks to days.
