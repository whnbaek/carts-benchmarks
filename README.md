# carts benchmarkss

Comprehensive benchmark suite for the CARTS (Compiler for Automatic Runtime Task Scheduling) system. Includes classical HPC benchmarks, AI/ML kernels, stencil computations, and task-parallel applications.

## Quick Start

### Building Benchmarks

Use the `carts benchmarks` command from anywhere in the CARTS project:

```bash
# Show available benchmarks and options
carts benchmarks --help

# Build with standard problem sizes
carts benchmarks polybench/2mm small          # ~1,000 elements
carts benchmarks sw4lite/rhs4sg-base medium   # ~10,000 elements
carts benchmarks ml-kernels/batchnorm large   # ~100,000 elements

# Build with default size
carts benchmarks simple-kernels/stream

# Build with custom ARTS configuration
carts benchmarks polybench/gemm small /path/to/arts.cfg
```

### Cleaning Benchmarks

Remove build artifacts, MLIR files, and binaries:

```bash
# Clean specific benchmark
carts benchmarks clean polybench/2mm

# Clean entire suite
carts benchmarks clean sw4lite
carts benchmarks clean polybench

# Clean all benchmarks
carts benchmarks clean --all

# Show what will be removed
carts benchmarks clean --help
```

Removes: build/, logs/, *.mlir, *.ll, *.o, metadata files

### Direct Makefile Usage

```bash
# Build individual benchmark
make -C polybench/2mm all

# Build with specific size
make -C sw4lite/rhs4sg-base small
make -C sw4lite/rhs4sg-base medium
make -C sw4lite/rhs4sg-base large

# Build entire suite
make -C sw4lite all
make -C specfem3d all
```

## Repository Structure

```
carts-benchmarks/
├── polybench/           # Dense linear algebra (11 benchmarks)
├── ml-kernels/          # Neural network kernels (4 benchmarks)
├── sw4lite/             # Seismic wave propagation (3 benchmarks)
├── specfem3d/           # Spectral element seismic (2 benchmarks)
├── seissol/             # High-order DG seismic (1 benchmark)
├── kastors/             # Task-parallel OpenMP (9 benchmarks)
├── task-parallelism-omp/ # OpenMP task patterns (4 benchmarks)
├── simple-kernels/      # Microbenchmarks (2 benchmarks)
├── miniapps/            # Small applications (1 benchmark)
├── npb/                 # NAS Parallel Benchmarks (1 benchmark)
├── prk/                 # Parallel Research Kernels (1 benchmark)
└── llama2-transformer/  # LLM transformer (1 benchmark)
```

## Benchmark Suites

### Dense Linear Algebra (Polybench)

**Source**: [PolyBench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)

Matrix operations and iterative solvers:

- `2mm`, `3mm` - Matrix multiplication chains
- `atax`, `bicg` - Matrix-vector operations
- `gemm` - General matrix multiply
- `jacobi2d`, `seidel-2d` - Iterative stencil solvers
- `fdtd-2d` - Finite difference time domain
- `convolution-2d`, `convolution-3d` - Convolution kernels
- `correlation` - Statistical correlation

**Size configuration**: Uses dataset macros (SMALL_DATASET, STANDARD_DATASET, LARGE_DATASET)

### Machine Learning Kernels

**Source**: [Darknet](https://github.com/pjreddie/darknet), [llama2.c](https://github.com/karpathy/llama2.c)

Neural network computational kernels:

- `batchnorm` - Batch normalization
- `pooling` - Max/average pooling operations
- `activations` - ReLU, GELU, Softmax
- `layernorm` - Layer normalization
- `transformer` - Decoder-only transformer (llama2)

**Size configuration**: Uses preset targets (mini, small, standard, large)

### Seismic Wave Propagation

**SW4Lite** - [geodynamics/sw4lite](https://github.com/geodynamics/sw4lite)

- `rhs4sg-base` - RHS assembly baseline (107 lines)
- `rhs4sg-revnw` - RHS assembly optimized (1,189 lines)
- `vel4sg-base` - Velocity update kernel

**SPECFEM3D** - [SPECFEM/specfem3d](https://github.com/SPECFEM/specfem3d)

- `stress` - Stress tensor update
- `velocity` - Velocity update with divergence

**SeisSol** - [SeisSol/SeisSol](https://github.com/SeisSol/SeisSol)

- `volume-integral` - ADER-DG volume integral

**Size configuration**: Grid dimensions for ~1K, ~10K, ~100K elements

### Task Parallelism

**KaStORS** - [viroulep/kastors](https://github.com/viroulep/kastors)

OpenMP 4.0 task dependency benchmarks:

- `jacobi/` - 4 variants (for, task-dep, poisson-for, poisson-task)
- `sparselu/` - 2 variants (task, task-dep)
- `strassen/` - 3 variants (main, task, task-dep)

**Task Parallelism OMP** - [avcourt/task-parallelism-omp](https://github.com/avcourt/task-parallelism-omp)

- `merge-serial`, `merge-tasks` - Merge sort implementations
- `quick-tasks` - Quicksort with tasks
- `get-time` - Timing utilities

**Size configuration**: Matrix/grid dimensions (1024, 4096, 8192)

### Microbenchmarks and Applications

**Simple Kernels**

- `stream` - Memory bandwidth (STREAM triad)
- `axpy` - Basic BLAS operation

**Miniapps**

- `stencil2d` - 2D stencil miniapp

**NPB**

- `ep` - Embarrassingly parallel (Monte Carlo)

**PRK**

- `stencil` - Configurable stencil patterns

## Standard Build Targets

Every benchmark supports these targets:

```bash
make seq              # Sequential MLIR (no OpenMP)
make metadata         # Collect parallelism metadata
make parallel         # Parallel MLIR (with OpenMP)
make concurrency      # Run concurrency analysis
make concurrency-opt  # Run optimized concurrency
make all              # Build all stages
make clean            # Remove build artifacts
```

## Problem Sizes

All benchmarks support three standardized sizes:

| Size | Elements | Use Case |
|------|----------|----------|
| small | ~1,000 | Quick testing, development |
| medium | ~10,000 | Standard testing |
| large | ~100,000 | Performance evaluation |

Suite-specific sizes:

| Suite | Small | Medium | Large |
|-------|-------|--------|-------|
| simple-kernels | N=1000 | N=10000 | N=100000 |
| miniapps | N=1000 | N=10000 | N=100000 |
| sw4lite | 10×10×10 | 21×21×22 | 46×46×47 |
| specfem3d | 5×5×5×8 | 5×5×5×80 | 5×5×5×800 |
| seissol | 1000 | 10000 | 100000 |
| polybench | SMALL_DATASET | STANDARD_DATASET | LARGE_DATASET |
| ml-kernels | mini | standard | large |
| kastors | 1024 | 4096 | 8192 |
| npb | M=20 | M=24 | M=28 |

## Build Configuration

### Compile-time Options

```bash
# Custom compiler flags
make -C polybench/2mm CFLAGS="-DMINI_DATASET -I/custom/path" all

# Custom ARTS configuration
make -C sw4lite/rhs4sg-base ARTS_CFG=/path/to/arts.cfg all

# Custom build/log directories
make -C ml-kernels/batchnorm BUILD_DIR=./output LOG_DIR=./logs all
```

### Runtime Parameters

Some benchmarks use runtime parameters:

```bash
# PRK stencil: <threads> <iterations> <dimension>
./prk-stencil 4 100 1000

# Task parallelism: element counts passed at runtime
./merge-tasks 100000
```

## CARTS Compatibility

All benchmarks meet these requirements:

- No global variables (clean parameter passing)
- OpenMP parallelization (parallel-for or task-based)
- Self-contained implementations
- Minimal external dependencies
- Documented with upstream attribution

## Workflow

Standard CARTS compilation workflow:

```
C/C++ source
    ↓ cgeist (sequential)
Sequential MLIR
    ↓ carts run --collect-metadata
Metadata JSON
    ↓ cgeist (with OpenMP)
Parallel MLIR
    ↓ carts run --concurrency
ARTS MLIR
    ↓ carts run --emit-llvm
LLVM IR
    ↓ carts compile
Executable
```

Each benchmark's Makefile automates this pipeline with configurable stages.

## Advanced Usage

### Building Suites

```bash
# Build all benchmarks in a suite
make -C polybench all
make -C sw4lite all
make -C kastors all

# Build specific suite with size
cd kastors/jacobi
make small   # Builds all jacobi variants with SIZE=1024
```

### Custom Configurations

```bash
# Override specific parameters
make -C sw4lite/rhs4sg-base CFLAGS="-DNX=64 -DNY=64 -DNZ=64" all

# Build with debug symbols
make -C polybench/2mm CFLAGS="-g -DSMALL_DATASET" all

# Specify custom radius for PRK stencil
make -C prk/stencil RADIUS=4 DOUBLE=1 all
```

### Intermediate Outputs

```bash
# Stop at specific pipeline stage
make -C polybench/2mm seq          # Only sequential MLIR
make -C polybench/2mm metadata     # Stop after metadata collection
make -C polybench/2mm parallel     # Only parallel MLIR

# View generated artifacts
ls polybench/2mm/build/
# 2mm_seq.mlir
# 2mm.carts-metadata.json
# 2mm.mlir
# 2mm_complete.mlir
# 2mm.ll
```

## Directory Structure

Each benchmark follows this structure:

```
benchmark-name/
├── Makefile              # Includes suite-specific common.mk
├── README.md             # Benchmark documentation
├── source.c              # Source code
├── arts.cfg              # ARTS runtime config (optional)
├── build/                # Generated MLIR/LLVM artifacts
└── logs/                 # Build logs for each stage
```

Suites provide shared infrastructure:

```
suite-name/
├── common/
│   └── carts-example.mk  # Suite-specific build configuration
├── Makefile              # Suite-level build driver
├── README.md             # Suite documentation
└── benchmark-1/
    └── ...
```

## References

### Upstream Sources

- **PolyBench**: Pouchet et al., "Polybench: The polyhedral benchmark suite." IMPACT, 2012
- **KaStORS**: Virouleau et al., "Evaluation of OpenMP dependent tasks with the KASTORS benchmark suite." IWOMP, 2014
- **NPB**: Bailey et al., "The NAS Parallel Benchmarks." IJSA, 1991
- **PRK**: Van der Wijngaart & Mattson, "The Parallel Research Kernels." HPCC, 2014
- **SPECFEM3D**: Komatitsch & Tromp, "Introduction to the spectral element method." GJI, 1999
- **SeisSol**: Dumbser & Käser, "An arbitrary high-order discontinuous Galerkin method." GJI, 2006

### Licenses

- BSD/BSD-like: Polybench, PRK
- GPL v2.0: KaStORS, BOTS
- MIT: llama2.c
- Public Domain: Darknet
- NASA Open Source: NPB

## Statistics

- Total suites: 12
- Total benchmarks: 45+
- Lines of code: ~15,000+
- File formats: C, C++, MLIR
- Parallelization: OpenMP (parallel-for, tasks, task dependencies)

## Contributing

When adding benchmarks:

1. Use no global variables
2. Include OpenMP parallelization
3. Minimize external dependencies
4. Provide README with algorithm description and attribution
5. Include validation/correctness checks
6. Support multiple problem sizes (small, medium, large)
7. Follow suite-specific Makefile patterns

---

**Last Updated**: November 23, 2024
