# correlation - Correlation Coefficient Matrix

## Description

Computes the correlation coefficients between rows of a matrix after centering and standardizing them.

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `datamining/correlation/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." IMPACT, 2012.

## Algorithm

Three-stage algorithm:
1. Mean and standard deviation calculation
2. Data normalization (centering and standardizing)
3. Correlation matrix construction

```
Input: data[M×N]
Output: corr[M×M]

For each row i:
  mean[i] = sum(data[i]) / N
  stddev[i] = sqrt(sum((data[i][j] - mean[i])^2) / N)

For each element data[i][j]:
  data[i][j] = (data[i][j] - mean[i]) / stddev[i]

For each pair of rows i, j:
  corr[i][j] = sum(data[i][k] * data[j][k]) / N
```

## Problem Sizes

| Size | Dimensions | Description |
|------|------------|-------------|
| **MINI** | 32×32 | Minimal size for quick testing |
| **SMALL** | 128×128 | Small problem size |
| **MEDIUM** | 256×256 | Standard problem size (default) |
| **LARGE** | 512×512 | Large problem size |
| **EXTRALARGE** | 1024×1024 | Extra large problem size |

## Building and Running

### Build with CARTS pipeline

```bash
# Build small size (128×128)
make small

# Build medium size (1024×1024) - default
make medium

# Build large size (2000×2000)
make large

# Build all pipeline stages (seq, metadata, parallel, concurrency)
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

## References

- **Polybench Suite**: http://polybench.sourceforge.net/
- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
