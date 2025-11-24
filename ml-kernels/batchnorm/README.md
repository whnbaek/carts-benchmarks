# Batch Normalization - CNN Normalization Layer

## Description

Batch normalization normalizes activations across the batch dimension in convolutional neural networks. Extracted from the Darknet framework, this kernel computes mean and variance for each channel across all batch samples and spatial positions, then normalizes, scales, and shifts the activations to stabilize training and improve convergence.

## Algorithm

For each channel c independently:

```
1. Compute mean: μ_c = mean(x[b,c,h,w] for all b,h,w)
2. Compute variance: σ²_c = var(x[b,c,h,w] for all b,h,w)
3. Normalize: x̂[b,c,h,w] = (x[b,c,h,w] - μ_c) / sqrt(σ²_c + ε)
4. Scale and shift: y[b,c,h,w] = γ_c * x̂[b,c,h,w] + β_c
```

Memory layout: NCHW (batch, channels, height, width)

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **MINI** | BATCH_SIZE=2, CHANNELS=8, HEIGHT=8, WIDTH=8 | Minimal size for quick testing |
| **SMALL** | BATCH_SIZE=4, CHANNELS=16, HEIGHT=16, WIDTH=16 | Small problem size |
| **MEDIUM** | BATCH_SIZE=4, CHANNELS=64, HEIGHT=32, WIDTH=32 | Medium problem size - default |
| **STANDARD** | BATCH_SIZE=4, CHANNELS=64, HEIGHT=32, WIDTH=32 | Standard problem size (same as medium) |
| **LARGE** | BATCH_SIZE=8, CHANNELS=128, HEIGHT=64, WIDTH=64 | Large problem size |

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

Batch normalization is a standard layer in convolutional neural networks:

- **CNNs**: Placed after convolutional layers (before or after activation); stabilizes training
- **ResNet**: Critical for training very deep networks (50-200+ layers)
- **VGG, AlexNet**: Improves convergence speed and allows higher learning rates
- **Object Detection**: Used in YOLO, SSD, Faster R-CNN backbones
- **Image Segmentation**: U-Net, DeepLab architectures
- **Benefits**: Reduces internal covariate shift; acts as regularizer; enables faster training

Note: Modern transformers typically use Layer Normalization instead, as batch norm's statistics across the batch dimension are less stable for variable-length sequences.

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP parallelization
