# CARTS Benchmarks

Comprehensive benchmark suite for testing the CARTS (Compiler for Automatic Runtime Task Scheduling) system. This collection includes benchmarks from classical HPC suites, AI/ML kernels, stencil computations, and task-parallel applications.

## Overview

All benchmarks in this repository have been verified for CARTS compatibility:
- ✅ **No global variables** - Clean parameter passing
- ✅ **OpenMP parallelization** - Using parallel-for or task-based constructs
- ✅ **Self-contained implementations** - Minimal external dependencies
- ✅ **Well-documented** - Each benchmark includes README with attribution

## Benchmark Categories

### 1. Machine Learning & AI Kernels

#### [ml-kernels/](ml-kernels/)
Modern neural network computational kernels extracted from production frameworks.

**Kernels**:
- **[batchnorm/](ml-kernels/batchnorm/)** - Batch normalization (from Darknet)
- **[pooling/](ml-kernels/pooling/)** - Max/Average/Global pooling (from Darknet)
- **[activations/](ml-kernels/activations/)** - ReLU, GELU, Softmax, etc.

**Source**: [Darknet](https://github.com/pjreddie/darknet), llama2.c
**Focus**: Reduction operations, irregular memory access, element-wise operations

#### [llama2-transformer/](llama2-transformer/)
Decoder-only transformer architecture for language modeling.

**Source**: [llama2.c by Andrej Karpathy](https://github.com/karpathy/llama2.c)
**Features**: Multi-head attention, RMSNorm, SwiGLU activation
**Focus**: Matrix operations, attention mechanism, layer-by-layer dependencies

---

### 2. Dense Linear Algebra (Polybench)

#### [polybench/](polybench/)
Classical linear algebra kernels from the Polybench benchmark suite.

**Matrix Operations**:
- **[2mm/](polybench/2mm/)** - Two matrix multiplications: D = A × B × C
- **[3mm/](polybench/3mm/)** - Three matrix multiplications: G = (A × B) × (C × D) × (E × F)
- **[atax/](polybench/atax/)** - Matrix transpose and vector multiply
- **[bicg/](polybench/bicg/)** - BiConjugate Gradient sub-kernel
- **[gemm/](polybench/gemm/)** - General Matrix Multiply (existing)

**Iterative Solvers**:
- **[jacobi2d/](polybench/jacobi2d/)** - 2D Jacobi stencil (existing)
- **[seidel-2d/](polybench/seidel-2d/)** - Gauss-Seidel 9-point stencil

**Stencils**:
- **[fdtd-2d/](polybench/fdtd-2d/)** - 2D Finite Difference Time Domain
- **[convolution-2d/](polybench/convolution-2d/)** - 2D convolution (3×3 kernel)
- **[convolution-3d/](polybench/convolution-3d/)** - 3D convolution (3×3×3 kernel)

**Source**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**License**: BSD-like
**Focus**: Dense matrix operations, stencil patterns, iterative methods

---

### 3. Stencil Computations

#### [prk/stencil/](prk/stencil/)
Configurable 2D stencil with runtime-adjustable radius and patterns.

**Source**: [Parallel Research Kernels (PRK)](https://github.com/ParRes/Kernels)
**Features**: Star/compact patterns, configurable radius
**Use**: PDEs, image processing, CFD
**Focus**: Neighbor dependencies, spatial locality

#### [sw4lite_rhs4sg_revNW/](sw4lite_rhs4sg_revNW/)
Seismic wave propagation kernel from SW4lite.

**Source**: SW4lite (LLNL)
**Application**: Earthquake simulation
**Focus**: 3D stencil, complex arithmetic dependencies

---

### 4. Task-Parallel Benchmarks (KaStORS)

#### [kastors/](kastors/)
OpenMP 4.0 task dependency benchmarks from the KaStORS suite.

**Benchmarks**:
- **[jacobi/](kastors/jacobi/)** - Jacobi iterative solver
  - `jacobi-for`: Parallel-for version
  - `jacobi-task-dep`: Task-dependency version

- **[sparselu/](kastors/sparselu/)** - Sparse LU factorization
  - `sparselu-task`: Task-based
  - `sparselu-task-dep`: With explicit dependencies

- **[strassen/](kastors/strassen/)** - Strassen matrix multiplication
  - `strassen-task`: Task-based
  - `strassen-task-dep`: With explicit dependencies

**Source**: [KaStORS](https://github.com/viroulep/kastors)
**Original**: Barcelona OpenMP Tasks Suite (BOTS)
**License**: GPL v2.0
**Focus**: Task parallelism, dependency patterns, data-flow execution

---

### 5. Embarrassingly Parallel

#### [npb/ep/](npb/ep/)
Monte Carlo random number generation benchmark.

**Source**: NAS Parallel Benchmarks (NASA)
**Pattern**: Embarrassingly parallel (no communication)
**Focus**: Raw computational throughput, perfect parallelism

---

### 6. Additional Benchmarks

#### [simple-kernels/](simple-kernels/)
Basic kernels for testing fundamental CARTS features.

#### [bots/](bots/)
Additional benchmarks from Barcelona OpenMP Tasks Suite.

#### [task-parallelism-omp/](task-parallelism-omp/)
Task-parallel patterns and examples.

#### [miniapps/](miniapps/)
Small representative applications.

---

## Quick Start

### Building Individual Benchmarks

Most benchmarks include Makefiles with multiple size configurations:

```bash
# ML kernels
cd ml-kernels/batchnorm/
make                    # Standard size
make mini              # Small (fast testing)
make large             # Large (stress testing)

# Polybench
cd polybench/2mm/
make                    # Uses polybench.mk

# KaStORS
cd kastors/sparselu/sparselu-task-dep/
make

# NPB
cd npb/ep/
gcc -O2 -fopenmp -lm ep.c -o ep -DM=24
```

### Running with CARTS

```bash
cd /Users/randreshg/Documents/carts

# ML kernel example
./tools/carts run cgeist \
  external/carts-benchmarks/ml-kernels/batchnorm/batchnorm.c \
  -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=32 -DWIDTH=32 \
  -fopenmp -O2 -lm

# Polybench example
./tools/carts run cgeist \
  external/carts-benchmarks/polybench/2mm/2mm.c \
  -I external/carts-benchmarks/polybench/utilities \
  -I external/carts-benchmarks/polybench/common \
  external/carts-benchmarks/polybench/utilities/polybench.c \
  -DPOLYBENCH_TIME -DMINI_DATASET \
  -fopenmp -O2 -lm
```

---

## Benchmark Statistics

### Total Benchmarks

| Category | Benchmarks | Variants | Total Files |
|----------|-----------|----------|-------------|
| **ML Kernels** | 3 | 15+ functions | 3 |
| **Polybench** | 10 | - | 10 |
| **Stencils** | 2 | - | 2 |
| **KaStORS** | 3 | 8 variants | 8 |
| **NPB** | 1 | - | 1 |
| **Others** | ~10 | - | ~10 |
| **TOTAL** | **~29** | - | **~34** |

### Lines of Code

- **ML Kernels**: ~1,300 lines
- **Polybench**: ~5,000 lines (benchmarks only)
- **KaStORS**: ~3,000 lines (3 benchmarks)
- **Total**: ~15,000+ lines of benchmark code

---

## Testing Focus Areas

### Memory Access Patterns
- **Sequential**: Token embeddings, array scans
- **Strided**: Multi-dimensional arrays, blocked algorithms
- **Irregular**: Sparse matrices, dynamic indexing
- **Stencil**: Neighbor access patterns
- **Reduction**: Sum, max, statistics computation

### Dependency Patterns
- **Independent**: Embarrassingly parallel (EP)
- **Reduction**: Batch normalization statistics
- **Producer-consumer**: Task dependencies
- **Wavefront**: Stencil computations
- **Recursive**: Strassen's algorithm

### Parallelization Strategies
- **Data parallel**: `#pragma omp parallel for`
- **Task parallel**: `#pragma omp task`
- **Dependencies**: `#pragma omp task depend(...)`
- **Reductions**: `#pragma omp ... reduction(...)`

---

## Documentation

Each benchmark directory contains a `README.md` with:
- Algorithm description
- Original source and repository links
- Build and usage instructions
- Mathematical background
- CARTS compatibility notes
- Performance characteristics
- Academic citations

### Additional Documentation

- **[README_NEW_BENCHMARKS.md](README_NEW_BENCHMARKS.md)** - Integration guide for newly added benchmarks
- **[AI_ML_BENCHMARKS.md](AI_ML_BENCHMARKS.md)** - Detailed AI/ML benchmark research and analysis

---

## Source Attribution

All benchmarks are properly attributed to their original sources:

### Primary Sources
- **Polybench**: [PolyBench-ACC](https://github.com/cavazos-lab/PolyBench-ACC) - BSD-like license
- **KaStORS**: [KaStORS](https://github.com/viroulep/kastors) - GPL v2.0
- **PRK**: [Parallel Research Kernels](https://github.com/ParRes/Kernels) - BSD 3-Clause
- **NPB**: [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html) - NASA Open Source Agreement
- **Darknet**: [Darknet](https://github.com/pjreddie/darknet) - Public Domain
- **llama2.c**: [llama2.c](https://github.com/karpathy/llama2.c) - MIT License

### Key Papers
- **Polybench**: Pouchet et al., "Polybench: The polyhedral benchmark suite." IMPACT, 2012.
- **KaStORS**: Virouleau et al., "Evaluation of OpenMP dependent tasks with the KASTORS benchmark suite." IWOMP, 2014.
- **BOTS**: Duran et al., "Barcelona OpenMP Tasks Suite." ICPP, 2009.
- **NPB**: Bailey et al., "The NAS Parallel Benchmarks." IJSA, 1991.
- **PRK**: Van der Wijngaart & Mattson, "The Parallel Research Kernels." HPCC, 2014.

---

## CARTS Compatibility Notes

### Supported Patterns
- ✅ OpenMP `parallel for` (maps to multi-node in CARTS)
- ✅ OpenMP `task` (maps to single-node in CARTS)
- ✅ Reduction operations
- ✅ Static arrays and VLAs
- ✅ Clean function interfaces

### Known Limitations
- ❌ Global variables (all benchmarks verified to avoid)
- ❌ Complex pointer aliasing (minimized in selection)
- ❌ Inline assembly (not present)
- ❌ External library dependencies (minimized)

### Benchmark Modifications
Some benchmarks include `-carts` variants with:
- Removed global variables
- Simplified control flow
- Added explicit parameter passing
- Adjusted for function inlining

---

## Problem Sizes

Most benchmarks support multiple problem sizes:

### Size Classes
- **MINI**: Fast compilation/execution (seconds) - for development
- **SMALL**: Quick testing (tens of seconds)
- **STANDARD**: Realistic workloads (minutes)
- **LARGE**: Stress testing (tens of minutes)

### Configuration Methods
- **Compile-time**: `-DMINI_DATASET`, `-DSTANDARD_DATASET`, etc.
- **Command-line**: Arguments for matrix size, iterations
- **Makefile targets**: `make mini`, `make small`, `make standard`, `make large`

---

## Contributing

When adding new benchmarks, ensure:

1. **No global variables** - Pass all data as parameters
2. **OpenMP support** - Use standard OpenMP constructs
3. **Self-contained** - Minimize external dependencies
4. **Documentation** - Include README.md with:
   - Algorithm description
   - Original source and links
   - Build instructions
   - Academic citations
5. **Validation** - Include correctness checks
6. **Multiple sizes** - Support at least mini/small/standard sizes

---

## License

See individual benchmark directories for specific license information. Common licenses:
- **BSD/BSD-like**: Polybench, PRK
- **GPL v2.0**: KaStORS, BOTS
- **MIT**: llama2.c
- **Public Domain**: Darknet
- **NASA Open Source**: NPB

---

## Contact & References

- **CARTS Documentation**: See main CARTS repository
- **Benchmark Issues**: Report in individual source repositories
- **Integration Questions**: See CARTS team

---

**Last Updated**: November 11, 2025
**Total Benchmarks**: ~29 unique kernels
**Total Variants**: ~34 implementations
**Lines of Code**: ~15,000+
