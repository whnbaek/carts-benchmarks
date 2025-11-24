# Layer Normalization - Transformer Normalization Layer

## Description

Layer normalization normalizes activations across the feature dimension for each sample independently, essential for training stable transformer models. Unlike batch normalization which normalizes across the batch, layer norm computes statistics per token/row, making it ideal for variable-length sequences and small batch sizes.

## Algorithm

For each sample (token/row) independently:

```
1. Compute mean: μ = mean(x[h] for all h in hidden_dim)
2. Compute variance: σ² = var(x[h] for all h in hidden_dim)
3. Normalize: x̂[h] = (x[h] - μ) / sqrt(σ² + ε)
4. Scale and shift: y[h] = γ[h] * x̂[h] + β[h]
```

Where γ (gamma) and β (beta) are learned per-channel affine parameters.

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **MINI** | BATCH=4, HIDDEN=256 | Minimal size for quick testing |
| **SMALL** | BATCH=8, HIDDEN=512 | Small problem size |
| **MEDIUM** | BATCH=16, HIDDEN=1024 | Medium problem size - default |
| **STANDARD** | BATCH=16, HIDDEN=1024 | Standard problem size (same as medium) |
| **LARGE** | BATCH=64, HIDDEN=4096 | Large problem size (GPT-scale) |

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size
make small

# Build medium size - default
make medium

# Build large size
make large

# Build all pipeline stages
make all
```

### Build individual stages

```bash
# Generate sequential MLIR
make seq

# Collect runtime metadata
make metadata

# Generate parallel MLIR
make parallel

# Run concurrency analysis
make concurrency

# Run optimized concurrency analysis
make concurrency-opt
```

### Clean build artifacts

```bash
make clean
```

## Use in Machine Learning

Layer normalization is the standard normalization technique in transformer architectures:

- **Transformers**: BERT, GPT-2, GPT-3, GPT-4 use layer norm after each attention and feedforward sublayer
- **Pre-LN vs Post-LN**: Modern transformers (GPT) use Pre-LN (before attention/FFN) for training stability
- **Vision Transformers**: ViT, CLIP, DINO apply layer norm to patch embeddings
- **Multimodal Models**: CLIP, DALL-E use layer norm in both vision and language towers
- **Advantages over Batch Norm**:
  - Independent of batch size; works with batch size = 1
  - No train/test discrepancy (no running statistics)
  - Better for variable-length sequences (NLP tasks)
  - Parallelizes better across the batch dimension

Layer norm is applied millions of times during transformer inference, making it a critical performance kernel.

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP parallelization
