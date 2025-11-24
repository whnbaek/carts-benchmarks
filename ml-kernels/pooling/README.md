# Pooling Operations - CNN Downsampling Layers

## Description

Pooling operations downsample spatial dimensions in convolutional neural networks by aggregating values within fixed-size windows. Extracted from the Darknet framework, this benchmark implements three pooling variants: max pooling (takes maximum value), average pooling (computes mean), and global average pooling (averages entire spatial dimensions for classification).

## Algorithm

**Max Pooling**: For each output position (b, c, i, j):
```
output[b,c,i,j] = max(input[b,c,i*stride+m,j*stride+n] for m,n in [0, pool_size))
```

**Average Pooling**: For each output position (b, c, i, j):
```
output[b,c,i,j] = mean(input[b,c,i*stride+m,j*stride+n] for m,n in [0, pool_size))
```

**Global Average Pooling**: For each channel in each batch:
```
output[b,c] = mean(input[b,c,h,w] for all h,w)
```

Memory layout: NCHW (batch, channels, height, width)

## Problem Sizes

| Size | Configuration | Description |
|------|--------------|-------------|
| **MINI** | BATCH_SIZE=1, CHANNELS=8, HEIGHT=16, WIDTH=16, POOL_SIZE=2 | Minimal size for quick testing |
| **SMALL** | BATCH_SIZE=2, CHANNELS=16, HEIGHT=32, WIDTH=32, POOL_SIZE=2 | Small problem size |
| **MEDIUM** | BATCH_SIZE=4, CHANNELS=64, HEIGHT=64, WIDTH=64, POOL_SIZE=2 | Medium problem size - default |
| **STANDARD** | BATCH_SIZE=4, CHANNELS=64, HEIGHT=64, WIDTH=64, POOL_SIZE=2 | Standard problem size (same as medium) |
| **LARGE** | BATCH_SIZE=8, CHANNELS=128, HEIGHT=128, WIDTH=128, POOL_SIZE=3 | Large problem size |

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

Pooling layers are fundamental components in convolutional neural networks:

- **Max Pooling**: Most common pooling operation; used in AlexNet, VGG, ResNet; 2x2 with stride 2 halves spatial dimensions; preserves strongest activations; provides translation invariance
- **Average Pooling**: Used in some architectures (LeNet); smoother downsampling than max pooling; less common in modern networks
- **Global Average Pooling**: Replaces fully connected layers for classification; used in ResNet, Inception, MobileNet; reduces parameters and overfitting; connects feature maps directly to class predictions
- **Object Detection**: YOLO, SSD, Faster R-CNN use pooling in feature extraction
- **Image Segmentation**: Downsampling path in U-Net, encoder-decoder architectures

Typical configurations:
- 2x2 pooling with stride 2: Standard for halving dimensions
- 3x3 pooling with stride 2: Used in some older architectures (AlexNet)

## CARTS Compatibility

- No global variables
- Clean parameter passing
- OpenMP parallelization
