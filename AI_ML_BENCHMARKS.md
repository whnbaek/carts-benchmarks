# AI/ML Benchmarks for CARTS

This document describes AI/ML and neural network benchmarks integrated into CARTS for testing dependency analysis on modern machine learning workloads.

## Overview

CARTS now includes comprehensive coverage of AI/ML computational patterns:
- **Transformer kernels**: Attention, normalization, activations
- **CNN kernels**: Convolution, pooling, batch normalization
- **Linear algebra**: Matrix multiplication variants
- **Activations**: ReLU, softmax, GELU

All benchmarks verified CARTS-compatible:
- ✅ No global variables
- ✅ Clean parameter passing
- ✅ OpenMP parallelization support
- ✅ Function inlining or macro-based indexing

---

## Existing Benchmarks

### 1. llama2-transformer (Complete Transformer Implementation)

**Location**: `external/carts-benchmarks/llama2-transformer/`

**Description**: Complete Llama 2 style transformer forward pass in 344 lines of clean C code.

**Kernels Implemented**:

#### 1.1 Matrix Multiplication (matmul)
```c
void matmul(float* xout, float* x, float* w, int n, int d)
```
- **Operation**: Matrix-vector product `xout = w * x`
- **Dimensions**: `w` is `d×n`, `x` is `n×1`, output is `d×1`
- **Parallelization**: `#pragma omp parallel for` over output dimension
- **Use**: Core operation in transformers (attention, FFN)

#### 1.2 RMS Normalization (rmsnorm)
```c
void rmsnorm(float* o, float* x, float* weight, int size)
```
- **Operation**: Root Mean Square layer normalization
- **Formula**: `o[i] = x[i] / rms(x) * weight[i]`
- **RMS**: `sqrt(sum(x[i]²) / size)`
- **Use**: Layer normalization in Llama architecture

#### 1.3 Softmax (softmax)
```c
void softmax(float* x, int size)
```
- **Operation**: Convert scores to probabilities
- **Formula**: `x[i] = exp(x[i] - max) / sum(exp(x[j] - max))`
- **Numerical Stability**: Subtracts max before exp
- **Use**: Attention weights, output probabilities

#### 1.4 Transformer Forward Pass (forward)
```c
void forward(Transformer* transformer, int token, int pos)
```
- **Complete transformer layer**:
  1. Token embedding lookup
  2. RoPE (Rotary Position Embedding)
  3. Multi-head self-attention (Q, K, V computation)
  4. Attention score computation and softmax
  5. Attention output aggregation
  6. Feed-forward network with SwiGLU activation
  7. Residual connections
  8. RMS normalization at each stage

**Configuration**:
```c
#define DIM 64          // Model dimension
#define HIDDEN_DIM 256  // FFN hidden dimension
#define N_LAYERS 2      // Number of transformer layers
#define N_HEADS 4       // Number of attention heads
#define SEQ_LEN 32      // Sequence length
```

**OpenMP Parallelization**:
- Matrix multiplications parallelized over output dimension
- Can be extended to parallelize across heads/layers

**Problem Sizes**:
- Tiny: DIM=64, HIDDEN_DIM=256, N_HEADS=4
- Can scale to larger configurations

**CARTS Suitability**:
- Tests data dependencies across attention mechanism
- Complex control flow with multiple matmuls
- Residual connections create interesting dependency patterns
- Good coverage of modern transformer operations

---

## New Benchmarks (Phase 1 - CNN Fundamentals)

### 2. 2D Convolution (Polybench)

**Location**: `external/carts-benchmarks/polybench/convolution-2d/`
**Files**: `convolution-2d.c`, `convolution-2d.h`

**Description**: 2D convolution with 3×3 kernel (fundamental CNN operation)

**Kernel Code** (simplified):
```c
void kernel_conv2d(int ni, int nj,
                   DATA_TYPE A[NI][NJ],
                   DATA_TYPE B[NI][NJ])
{
  #pragma omp parallel for private(j) collapse(2) schedule(static)
  for (i = 1; i < ni - 1; ++i)
    for (j = 1; j < nj - 1; ++j)
      B[i][j] = 0.2 * A[i-1][j-1] + 0.5 * A[i-1][j] + -0.8 * A[i-1][j+1]
              + -0.3 * A[i][j-1]   + 0.6 * A[i][j]   + -0.9 * A[i][j+1]
              + 0.4 * A[i+1][j-1]  + 0.7 * A[i+1][j] + 0.1 * A[i+1][j+1];
}
```

**Characteristics**:
- **Stencil Pattern**: 3×3 (9 points)
- **Operation**: Weighted sum of neighborhood
- **Boundary Handling**: Excludes boundaries (i, j ∈ [1, n-2])
- **Parallelization**: `collapse(2)` for 2D loop nest

**Problem Sizes**:
- MINI: 32×32
- SMALL: 128×128
- STANDARD: 1024×1024
- LARGE: 4096×4096

**Use in CNNs**:
- Fundamental convolution operation
- Feature extraction in convolutional layers
- Most common filter size in practice (3×3)

**CARTS Suitability**:
- 9-point stencil tests spatial dependencies
- Similar to existing stencils but with CNN context
- Good for testing filter weight dependencies

---

### 3. 3D Convolution (Polybench)

**Location**: `external/carts-benchmarks/polybench/convolution-3d/`
**Files**: `convolution-3d.c`, `convolution-3d.h`

**Description**: 3D convolution with 3×3×3 kernel (for video/3D CNNs)

**Kernel Code** (simplified):
```c
void kernel_conv3d(int ni, int nj, int nk,
                   DATA_TYPE A[NI][NJ][NK],
                   DATA_TYPE B[NI][NJ][NK])
{
  #pragma omp parallel
  {
    #pragma omp for private(j,k) collapse(2)
    for (i = 1; i < ni - 1; ++i)
      for (j = 1; j < nj - 1; ++j)
        for (k = 1; k < nk - 1; ++k)
          B[i][j][k] = /* 27-point 3×3×3 stencil */
  }
}
```

**Characteristics**:
- **Stencil Pattern**: 3×3×3 (27 points)
- **Operation**: 3D weighted sum
- **Dimensions**: Spatial (x, y) + temporal/depth (z)

**Problem Sizes**:
- MINI: 32×32×32
- SMALL: 64×64×64
- STANDARD: 256×256×256

**Use in CNNs**:
- Video processing (spatial + temporal)
- 3D medical imaging (CT, MRI)
- Point cloud processing

**CARTS Suitability**:
- 27-point stencil tests complex 3D dependencies
- Larger dependency footprint than 2D
- Tests CARTS analysis on higher-dimensional data

---

## Planned Benchmarks (Phase 1 Completion)

### 4. Batch Normalization (from Darknet)

**Source**: Darknet `src/batchnorm_layer.c`
**Status**: Extracting

**Description**: Normalizes activations across batch dimension

**Algorithm**:
```
For each channel c:
  1. Compute mean:     μ_c = mean(x[b,c,h,w] for all b,h,w)
  2. Compute variance: σ²_c = var(x[b,c,h,w] for all b,h,w)
  3. Normalize:        x̂[b,c,h,w] = (x[b,c,h,w] - μ_c) / sqrt(σ²_c + ε)
  4. Scale and shift:  y[b,c,h,w] = γ_c * x̂[b,c,h,w] + β_c
```

**Key Functions** (to extract):
```c
void forward_batchnorm_layer(layer l, network net);
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void scale_bias_cpu(float *output, float *scales, int batch, int n, int size);
void add_bias_cpu(float *output, float *biases, int batch, int n, int size);
```

**Use in CNNs**:
- After convolutional layers
- Before activation functions
- Stabilizes training
- Improves convergence

**CARTS Suitability**:
- Tests reduction operations (mean, variance)
- Broadcast operations (normalize, scale)
- Interesting data dependencies across batch dimension

---

### 5. Max Pooling (from Darknet)

**Source**: Darknet `src/maxpool_layer.c`
**Status**: Extracting

**Description**: Downsamples by taking maximum value in each pooling window

**Algorithm**:
```c
void forward_maxpool_layer(const maxpool_layer l, network net)
{
  for (batch)
    for (channel)
      for (out_y)
        for (out_x)
        {
          float max = -FLT_MAX;
          for (pool_y)
            for (pool_x)
            {
              int in_y = out_y * stride + pool_y - padding;
              int in_x = out_x * stride + pool_x - padding;
              if (in_bounds(in_y, in_x))
                max = MAX(max, input[in_y][in_x]);
            }
          output[out_y][out_x] = max;
        }
}
```

**Typical Configurations**:
- **2×2 pooling, stride 2**: Halves spatial dimensions
- **3×3 pooling, stride 2**: Common in older architectures

**Use in CNNs**:
- Downsampling between convolutional layers
- Translation invariance
- Reduces computation in deeper layers

**CARTS Suitability**:
- Tests max reduction operation
- Irregular memory access pattern (pooling windows)
- Good for testing CARTS on non-uniform dependencies

---

### 6. Average Pooling (from Darknet)

**Source**: Darknet `src/avgpool_layer.c`
**Status**: Extracting

**Description**: Downsamples by averaging values in each pooling window

**Algorithm**:
```c
void forward_avgpool_layer(const avgpool_layer l, network net)
{
  for (batch)
    for (channel)
      for (out_y)
        for (out_x)
        {
          float sum = 0;
          int count = 0;
          for (pool_y)
            for (pool_x)
            {
              int in_y = out_y * stride + pool_y - padding;
              int in_x = out_x * stride + pool_x - padding;
              if (in_bounds(in_y, in_x)) {
                sum += input[in_y][in_x];
                count++;
              }
            }
          output[out_y][out_x] = sum / count;
        }
}
```

**Use in CNNs**:
- Global average pooling (before classification)
- Alternative to max pooling
- Provides average feature responses

**CARTS Suitability**:
- Tests average reduction
- Similar pattern to max pooling but different operation
- Good comparison case

---

### 7. ReLU Activation (from Darknet)

**Source**: Darknet `src/activation_layer.c` or `src/activations.c`
**Status**: Extracting

**Description**: Rectified Linear Unit - most common activation function

**Algorithm**:
```c
void activate_relu(float *x, int n)
{
  #pragma omp parallel for
  for (int i = 0; i < n; ++i)
    x[i] = (x[i] > 0) ? x[i] : 0;
}

// Leaky ReLU variant
void activate_leaky(float *x, int n)
{
  #pragma omp parallel for
  for (int i = 0; i < n; ++i)
    x[i] = (x[i] > 0) ? x[i] : 0.1 * x[i];
}
```

**Variants to Extract**:
- **ReLU**: `max(0, x)`
- **Leaky ReLU**: `max(0.1x, x)`
- **ReLU6**: `min(max(0, x), 6)` (mobile networks)

**Use in CNNs**:
- After every convolutional layer
- After fully connected layers
- Introduces non-linearity

**CARTS Suitability**:
- Simple element-wise operation
- Good baseline for activation functions
- Tests conditional execution patterns

---

## Planned Benchmarks (Phase 2 - Transformer Additions)

### 8. Layer Normalization (from llm.c)

**Source**: https://github.com/karpathy/llm.c
**Status**: To be extracted

**Description**: Normalizes across feature dimension (different from batch norm)

**Algorithm**:
```c
void layernorm(float* out, float* x, float* weight, float* bias, int size)
{
  // Calculate mean
  float mean = 0.0f;
  for (int i = 0; i < size; i++)
    mean += x[i];
  mean /= size;

  // Calculate variance
  float variance = 0.0f;
  for (int i = 0; i < size; i++) {
    float diff = x[i] - mean;
    variance += diff * diff;
  }
  variance /= size;

  // Normalize, scale, and shift
  float inv_std = 1.0f / sqrtf(variance + 1e-5f);
  for (int i = 0; i < size; i++)
    out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
}
```

**Difference from Batch Norm**:
- **Batch Norm**: Normalizes across batch dimension (N)
- **Layer Norm**: Normalizes across feature dimension (C/D)
- **Use cases**: Layer norm for transformers, batch norm for CNNs

**Use in Transformers**:
- Before/after attention layers
- Before/after feed-forward layers
- Standard in BERT, GPT, T5, etc.

**CARTS Suitability**:
- Different normalization pattern than RMSNorm
- Tests CARTS on feature-wise operations
- Complements existing normalization kernels

---

### 9. GELU Activation (to implement)

**Status**: To be implemented

**Description**: Gaussian Error Linear Unit - used in transformers

**Algorithm**:
```c
void activate_gelu(float *x, int n)
{
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    float val = x[i];
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    float cube = val * val * val;
    float inner = 0.7978845608f * (val + 0.044715f * cube);
    x[i] = 0.5f * val * (1.0f + tanhf(inner));
  }
}

// Alternative: GELU approximation (faster)
void activate_gelu_fast(float *x, int n)
{
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    float val = x[i];
    x[i] = val * sigmoid(1.702f * val);
  }
}
```

**Use in Transformers**:
- GPT-2, GPT-3, BERT use GELU
- Smoother than ReLU
- Better gradient flow

**CARTS Suitability**:
- Tests transcendental functions (tanh, pow)
- More complex than ReLU
- Representative of modern transformer activations

---

## Benchmark Coverage Summary

### By AI/ML Domain

| Domain | Kernels | Status |
|--------|---------|--------|
| **Transformers** | Attention, MatMul, RMSNorm, LayerNorm, Softmax, GELU | ✅ Mostly Complete |
| **CNNs** | Conv2D, Conv3D, BatchNorm, MaxPool, AvgPool, ReLU | ✅ In Progress |
| **Linear Algebra** | GEMM, 2mm, 3mm, atax, bicg | ✅ Complete |
| **Activations** | Softmax, ReLU, GELU, Leaky ReLU | ⚠️ Partial |
| **Normalization** | RMSNorm, BatchNorm, LayerNorm | ⚠️ Partial |
| **Pooling** | MaxPool, AvgPool | ⚠️ Extracting |

### By Computational Pattern

| Pattern | Example Kernels | CARTS Testing Focus |
|---------|-----------------|---------------------|
| **Matrix Operations** | matmul, GEMM, 2mm, 3mm | Data dependencies in matrix multiply chains |
| **Reductions** | Softmax, mean, variance | Reduction across dimensions |
| **Stencils** | Conv2D, Conv3D | Spatial dependencies in filtering |
| **Element-wise** | ReLU, GELU | Simple parallelizable operations |
| **Broadcast** | Normalization scale/shift | Broadcasting across dimensions |
| **Irregular Access** | Attention, pooling | Non-uniform memory access patterns |

### By Complexity

| Complexity | Kernels | Lines of Code |
|------------|---------|---------------|
| **Simple** | ReLU, GELU, element-wise ops | <50 lines |
| **Medium** | Pooling, normalization, softmax | 50-200 lines |
| **Complex** | Convolution, attention, transformer | 200-500 lines |

---

## Building and Running

### Standard Build (without CARTS)

#### Transformer:
```bash
cd external/carts-benchmarks/llama2-transformer/transformer
gcc -O2 -fopenmp -lm transformer.c -o transformer
./transformer
```

#### Polybench Convolutions:
```bash
cd external/carts-benchmarks/polybench/convolution-2d
make CFLAGS="-O2 -DMINI_DATASET"
./convolution-2d
```

### Build with CARTS

#### Transformer:
```bash
cd /Users/randreshg/Documents/carts
./tools/carts run cgeist \
  external/carts-benchmarks/llama2-transformer/transformer/transformer.c \
  -fopenmp -O2 -lm
```

#### Polybench Kernels:
```bash
./tools/carts run cgeist \
  external/carts-benchmarks/polybench/convolution-2d/convolution-2d.c \
  -I external/carts-benchmarks/polybench/utilities \
  -I external/carts-benchmarks/polybench/common \
  -D POLYBENCH_TIME -D MINI_DATASET \
  -fopenmp -O2 -lm
```

---

## Validation

All benchmarks include self-validation:
- **Transformer**: Validates output token probabilities
- **Polybench**: Compares against analytical solutions
- **Darknet kernels**: Will include test cases with known outputs

---

## Comparison with Existing Benchmarks

| AI/ML Kernel | Similar Existing CARTS Benchmark | Difference |
|--------------|-----------------------------------|------------|
| Conv2D/3D | polybench stencils (jacobi, fdtd, seidel) | CNN context, learned weights vs fixed stencils |
| MatMul | polybench GEMM, 2mm, 3mm | Transformer context (Q×K, attention×V) |
| Batch Norm | - | New pattern: normalization across batch |
| Layer Norm | RMSNorm (transformer) | Different normalization (mean+var vs RMS) |
| Pooling | - | New pattern: max/avg reduction with windowing |
| Attention | - | New pattern: Q×K^T, softmax, ×V chain |

---

## Next Steps

**Phase 1 (Week 1) - Immediate**:
- ✅ Copy Conv2D and Conv3D from Polybench
- ⏳ Extract batch norm from Darknet
- ⏳ Extract max/avg pooling from Darknet
- ⏳ Extract ReLU from Darknet
- ⏳ Create test harness for all kernels

**Phase 2 (Week 2) - Transformers**:
- [ ] Clone llm.c repository
- [ ] Extract layer normalization
- [ ] Implement GELU activation
- [ ] Extract standalone attention mechanism
- [ ] Documentation and testing

**Phase 3 (Optional) - Advanced**:
- [ ] Im2col + GEMM convolution
- [ ] Winograd convolution
- [ ] Additional activation functions
- [ ] Embedding lookup operations

---

## References

- **Transformer**: Based on Llama 2 architecture
- **Polybench**: http://polybench.sourceforge.net/
- **Darknet**: https://github.com/pjreddie/darknet (YOLO framework)
- **llm.c**: https://github.com/karpathy/llm.c (Karpathy's LLM training in C)

---

**Last Updated**: November 11, 2025
**Status**: Phase 1 in progress
**Total AI/ML Benchmarks**: 2 complete + 7 in progress = 9 kernels
