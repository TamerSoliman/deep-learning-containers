# PyTorch 2.8/2.7 NeuronX Training Containers - External Reference

**Source**: External GitHub Repository - https://github.com/aws-neuron/deep-learning-containers

**Note**: Starting from Neuron SDK 2.17.0, PyTorch Neuron Dockerfiles are maintained in a separate AWS Neuron repository.

---

## Key Differences from HuggingFace NeuronX Containers

| Aspect | PyTorch NeuronX (Native) | HuggingFace NeuronX |
|--------|--------------------------|---------------------|
| **Base** | PyTorch + torch-neuronx | HF PyTorch NeuronX + transformers |
| **Libraries** | Minimal (PyTorch essentials) | Full HF ecosystem (transformers, PEFT, etc.) |
| **Use Case** | Custom model architectures | Fine-tuning pre-trained models |
| **Flexibility** | Maximum (full control) | Simplified (HF abstractions) |

---

## PyTorch 2.8.0 NeuronX Training

**GitHub**: https://github.com/aws-neuron/deep-learning-containers/blob/2.26.1/pytorch/training/2.8.0/Dockerfile.neuronx

**Components**:
- PyTorch 2.8.0
- torch-neuronx (NeuronSDK 2.26.1)
- neuronx_distributed (tensor/pipeline parallelism)
- Python 3.11

**When to Use**:
- Custom model architectures (not standard transformers)
- Research requiring low-level control
- Don't need HuggingFace abstractions

---

## PyTorch 2.7.0 NeuronX Training

**GitHub**: https://github.com/aws-neuron/deep-learning-containers/blob/2.25.0/docker/pytorch/training/2.7.0/Dockerfile.neuronx

**Components**:
- PyTorch 2.7.0
- torch-neuronx (NeuronSDK 2.25.0)  
- neuronx_distributed
- Python 3.10

**When to Use**:
- More stable/tested than 2.8
- Production deployments
- Same use cases as 2.8 but prioritizing stability

---

## Architecture (Same as HF NeuronX)

Both build:
1. Ubuntu base
2. Python from source
3. OpenMPI for distributed training
4. EFA drivers
5. Neuron SDK components
6. PyTorch with torch-neuronx integration

**Difference**: HuggingFace variant adds transformers, PEFT, TRL, optimum-neuron on top

---

## Training Example (Native PyTorch NeuronX)

```python
import torch
import torch_neuronx
from torch_neuronx import xla

# Custom model (not using HuggingFace)
class MyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([...])
    
    def forward(self, x):
        return self.layers(x)

# Move to Neuron device
model = MyTransformer().to("xla")
optimizer = torch.optim.AdamW(model.parameters())

# Training loop
for batch in dataloader:
    outputs = model(batch)
    loss = compute_loss(outputs)
    loss.backward()
    xla.mark_step()  # Sync NeuronCores
    optimizer.step()
```

**HuggingFace Equivalent**: Would use `NeuronTrainer` abstraction instead of manual XLA management

---

## Decision Matrix

Use **PyTorch NeuronX (Native)** when:
- ✅ Building custom architectures
- ✅ Need low-level XLA control
- ✅ Research requiring flexibility
- ✅ Want minimal dependencies

Use **HuggingFace NeuronX** when:
- ✅ Fine-tuning Llama, GPT, T5, etc.
- ✅ Want HF ecosystem (datasets, PEFT, TRL)
- ✅ Prefer higher-level APIs
- ✅ Standard transformer models

---

## Reference Links

- **PyTorch 2.8 NeuronX**: https://github.com/aws-neuron/deep-learning-containers/blob/2.26.1/pytorch/training/2.8.0/Dockerfile.neuronx
- **PyTorch 2.7 NeuronX**: https://github.com/aws-neuron/deep-learning-containers/blob/2.25.0/docker/pytorch/training/2.7.0/Dockerfile.neuronx
- **AWS Neuron Documentation**: https://awsdocs-neuron.readthedocs-hosted.com/
- **torch-neuronx API**: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/index.html

---

**Summary**: These containers are for advanced users who need full control over Neuron training. Most users should use HuggingFace NeuronX containers which provide the same performance with simpler APIs.
