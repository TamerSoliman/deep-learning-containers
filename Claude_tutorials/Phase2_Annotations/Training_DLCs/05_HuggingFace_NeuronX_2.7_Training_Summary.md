# HuggingFace PyTorch 2.7 NeuronX Training Container - Key Differences

**Source**: `huggingface/pytorch/training/docker/2.7/py3/sdk2.24.1/Dockerfile.neuronx`

**Relationship to 2.8 NeuronX**: Previous generation, more stable, similar architecture

---

## Key Differences from 2.8 NeuronX

| Component | 2.7 NeuronX | 2.8 NeuronX |
|-----------|-------------|-------------|
| **PyTorch** | 2.7.0 | 2.8.0 |
| **Neuron SDK** | 2.24.1 | 2.26.0 |
| **transformers** | 4.51.0 | 4.55.4 |
| **optimum-neuron** | 0.3.x | 0.4.1 |
| **Python** | 3.10.12 | 3.10.12 (same) |
| **Stability** | Production-proven (6+ months) | Newer (2-3 months) |

---

## Version Alignment

**Neuron SDK 2.24.1** (2.7 container):
- Release: July 2025
- Stability: Proven in production
- Model Support: Llama 2/3, GPT-J, T5, BERT
- Known Issues: All documented and worked around

**Neuron SDK 2.26.0** (2.8 container):
- Release: September 2025
- Stability: Early production
- Model Support: Same + improved Mistral support
- Performance: 5-10% faster compilation

---

## When to Use 2.7 vs 2.8 NeuronX

### Use 2.7 NeuronX When:
- **Stability is critical** (production workloads)
- **Training Llama 2/3** (extensively tested)
- **Conservative infrastructure** (slow change approval)
- **Want 6+ months of bug fixes**

### Use 2.8 NeuronX When:
- **Need latest features** (improved Mistral support)
- **Willing to encounter edge cases**
- **Want slightly better performance** (5-10%)
- **Research/experimentation**

---

## Identical Architecture

Both containers share:
- ✅ Same build process (from Ubuntu base)
- ✅ Same EFA setup
- ✅ Same OpenMPI configuration
- ✅ Same distributed training approach (neuronx_distributed)
- ✅ Same cost savings (30-50% vs GPU)

**Key Point**: Architecture identical, only library versions differ
