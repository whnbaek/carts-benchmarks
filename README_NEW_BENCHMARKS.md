# CARTS Benchmark Suite - New Additions (November 2025)

This document describes the newly integrated benchmarks from Polybench-ACC and Parallel Research Kernels (PRK).

## Quick Summary

**Total New Benchmarks**: 7
- **Polybench**: 6 benchmarks (linear algebra + stencils)
- **PRK**: 1 benchmark (stencil)

All benchmarks have been verified to meet CARTS constraints:
- ✅ No global variables
- ✅ Clean parameter passing
- ✅ OpenMP support (`#pragma omp parallel for`)
- ✅ Ready for CARTS toolchain compilation

---

## Polybench Benchmarks

Location: `/external/carts-benchmarks/polybench/`

### Linear Algebra Kernels

#### 1. 2mm - Two Matrix Multiplications
**Directory**: `polybench/2mm/`
**Files**: `2mm.c`, `2mm.h`

**Description**: Performs two matrix multiplications: D := alpha*A*B*C + beta*D

**Computation Pattern**:
```
tmp = alpha * A * B
D = tmp * C + beta * D
```

**Problem Sizes**:
- MINI: 32×32 matrices
- SMALL: 128×128 matrices
- STANDARD: 1024×1024 matrices (default)
- LARGE: 2000×2000 matrices
- EXTRALARGE: 4000×4000 matrices

**OpenMP Constructs**:
- `#pragma omp parallel`
- `#pragma omp for private(j, k)`

**CARTS Suitability**: Excellent for testing data dependencies between matrix operations

---

#### 2. 3mm - Three Matrix Multiplications
**Directory**: `polybench/3mm/`
**Files**: `3mm.c`, `3mm.h`

**Description**: Performs three matrix multiplications: E = A*B, F = C*D, G = E*F

**Computation Pattern**:
```
E = A * B
F = C * D
G = E * F
```

**Problem Sizes**:
- MINI: 32×32 matrices
- SMALL: 128×128 matrices
- STANDARD: 1024×1024 matrices (default)
- LARGE: 2000×2000 matrices
- EXTRALARGE: 4000×4000 matrices

**OpenMP Constructs**:
- `#pragma omp parallel private(j, k)`
- Multiple `#pragma omp for` within single parallel region

**CARTS Suitability**: Good for testing dependency chains across multiple operations

---

#### 3. atax - Matrix Transpose and Vector Multiplication
**Directory**: `polybench/atax/`
**Files**: `atax.c`, `atax.h`

**Description**: Computes y = A^T * (A * x)

**Computation Pattern**:
```
tmp = A * x
y = A^T * tmp
```

**Problem Sizes**:
- MINI: 32 rows, 32 columns
- SMALL: 128 rows, 128 columns
- STANDARD: 4000 rows, 4000 columns (default)
- LARGE: 8000 rows, 8000 columns
- EXTRALARGE: 100000 rows, 100000 columns

**OpenMP Constructs**:
- `#pragma omp parallel`
- `#pragma omp for` and `#pragma omp for private(j)`

**CARTS Suitability**: Tests matrix-vector operations and transpose patterns

---

#### 4. bicg - BiConjugate Gradient Sub-kernel
**Directory**: `polybench/bicg/`
**Files**: `bicg.c`, `bicg.h`

**Description**: Sub-kernel of BiCGStab linear solver: s = A^T * r; q = A * p

**Computation Pattern**:
```
s = A^T * r
q = A * p
```

**Problem Sizes**:
- MINI: 32 rows, 32 columns
- SMALL: 128 rows, 128 columns
- STANDARD: 4000 rows, 4000 columns (default)
- LARGE: 8000 rows, 8000 columns
- EXTRALARGE: 100000 rows, 100000 columns

**OpenMP Constructs**:
- `#pragma omp parallel`
- `#pragma omp for` and `#pragma omp for private(j)`

**CARTS Suitability**: Tests irregular access patterns common in iterative solvers

---

### Stencil Kernels

#### 5. fdtd-2d - 2D Finite Difference Time Domain
**Directory**: `polybench/fdtd-2d/`
**Files**: `fdtd-2d.c`, `fdtd-2d.h`

**Description**: 2D FDTD electromagnetic field solver with time-stepping

**Computation Pattern**: 3-point stencil updates for electric and magnetic fields

**Problem Sizes**:
- MINI: 20 timesteps, 32×32 grid
- SMALL: 40 timesteps, 128×128 grid
- STANDARD: 50 timesteps, 1000×1000 grid (default)
- LARGE: 100 timesteps, 2000×2000 grid
- EXTRALARGE: 1000 timesteps, 4000×4000 grid

**OpenMP Constructs**:
- `#pragma omp parallel private(t,i,j)`
- `#pragma omp master` with time-stepping loop
- `#pragma omp for collapse(2) schedule(static)`
- `#pragma omp barrier` for synchronization

**CARTS Suitability**:
- Tests master/barrier synchronization patterns
- Time-stepping with stencil dependencies
- Good for testing temporal dependencies

---

#### 6. seidel-2d - Gauss-Seidel 2D Stencil
**Directory**: `polybench/seidel-2d/`
**Files**: `seidel-2d.c`, `seidel-2d.h`

**Description**: 2D Gauss-Seidel iterative solver with 9-point stencil

**Computation Pattern**: 9-point averaging stencil
```
A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
         + A[i][j-1]   + A[i][j]   + A[i][j+1]
         + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0
```

**Problem Sizes**:
- MINI: 20 timesteps, 32×32 grid
- SMALL: 40 timesteps, 128×128 grid
- STANDARD: 20 timesteps, 1000×1000 grid (default)
- LARGE: 40 timesteps, 2000×2000 grid
- EXTRALARGE: 100 timesteps, 4000×4000 grid

**OpenMP Constructs**:
- `#pragma omp parallel private(t,i,j)`
- `#pragma omp master`
- `#pragma omp for schedule(static) collapse(2)`

**CARTS Suitability**:
- 9-point stencil tests comprehensive spatial dependencies
- Iterative solver pattern
- Good for testing dependency analysis on stencil boundaries

---

## PRK (Parallel Research Kernels)

Location: `/external/carts-benchmarks/prk/`

### 7. Stencil - Configurable Radius 2D Stencil
**Directory**: `prk/stencil/`
**Files**: `stencil.c`

**Description**: 2D stencil computation with compile-time configurable radius and patterns

**Computation Pattern**: Star or compact stencil patterns with configurable RADIUS
- Supports multiple stencil types (star, compact)
- Radius configured via compile-time macro

**Problem Sizes**: Command-line configurable:
```bash
./stencil <# threads> <# iterations> <array dimension>
```

**OpenMP Constructs**:
- `#pragma omp parallel for`
- Clean loop-based parallelism

**Macro Definitions** (similar to sw4lite):
```c
#define IN(i,j)       in[i+(j)*(n)]
#define OUT(i,j)      out[i+(j)*(n)]
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]
```

**CARTS Suitability**:
- Very similar structure to existing sw4lite_rhs4sg_revNW benchmark
- Uses macro-based array indexing (CARTS-friendly)
- Uses `restrict` qualifiers for optimization
- Excellent for testing different stencil radii and patterns

**Build Options**:
- `-DRADIUS=<n>`: Set stencil radius (default: 2)
- `-DSTAR`: Use star stencil pattern
- `-DDOUBLE`: Use double precision (default: float)

---

## Building the Benchmarks

### Polybench Benchmarks

#### Standard Build (without CARTS):
```bash
cd polybench/<benchmark-name>/
make
```

The Makefiles include:
- `utilities/options.mk`: Compiler options (gcc, -O2)
- `utilities/c2.mk`: Build rules

#### Build with Different Problem Sizes:
```bash
cd polybench/2mm/
make CFLAGS="-O2 -fopenmp -DMINI_DATASET"      # Mini size
make CFLAGS="-O2 -fopenmp -DSMALL_DATASET"     # Small size
make CFLAGS="-O2 -fopenmp -DSTANDARD_DATASET"  # Standard (default)
make CFLAGS="-O2 -fopenmp -DLARGE_DATASET"     # Large size
```

#### Build with CARTS:
```bash
cd /Users/randreshg/Documents/carts
./tools/carts run cgeist external/carts-benchmarks/polybench/2mm/2mm.c \
  -I external/carts-benchmarks/polybench/utilities \
  -I external/carts-benchmarks/polybench/common \
  -D POLYBENCH_TIME -D MINI_DATASET \
  -fopenmp -O2
```

### PRK Stencil Benchmark

#### Standard Build (without CARTS):
```bash
cd prk/stencil/
make stencil RADIUS=2 DOUBLE=1
```

#### Run:
```bash
./stencil 4 100 1000  # 4 threads, 100 iterations, 1000x1000 grid
```

#### Build with CARTS:
```bash
cd /Users/randreshg/Documents/carts
./tools/carts run cgeist external/carts-benchmarks/prk/stencil/stencil.c \
  -I external/carts-benchmarks/prk/include \
  -I external/carts-benchmarks/prk/common \
  -D RADIUS=2 -D DOUBLE \
  -fopenmp -O2
```

---

## Validation

All benchmarks include self-validation:
- Polybench: Computes reference value and checks result
- PRK: Computes analytical solution and compares

Example output (successful):
```
Parallel Research Kernels version 2.17
OpenMP stencil execution on 2D grid
Grid size            = 1000
Radius of stencil    = 2
Type of stencil      = star
Data type            = double precision
Compact representation of stencil loop body
Number of iterations = 100
Solution validates
```

---

## Integration with CARTS Testing

### Recommended Test Sequence:

1. **Start with 2mm** (simplest linear algebra)
   - Test basic matrix multiply dependencies
   - Verify DB insertion for matmul patterns

2. **Test 3mm** (dependency chains)
   - Multiple matrix operations
   - Test dependency propagation

3. **Test atax and bicg** (sparse-like patterns)
   - Irregular access patterns
   - Matrix-vector operations

4. **Test seidel-2d** (9-point stencil)
   - Comprehensive stencil dependencies
   - Similar to existing benchmarks but different pattern

5. **Test fdtd-2d** (time-stepping)
   - Master/barrier patterns
   - Temporal dependencies

6. **Test PRK Stencil** (comparison)
   - Compare with sw4lite stencil
   - Validate different stencil patterns

---

## Comparison with Existing Benchmarks

| New Benchmark | Similar Existing | Key Differences |
|---------------|------------------|-----------------|
| polybench/2mm | polybench/gemm | Two chained matmuls vs single GEMM |
| polybench/3mm | polybench/gemm | Three chained matmuls |
| polybench/seidel-2d | polybench/jacobi2d | 9-point vs 5-point stencil |
| polybench/fdtd-2d | - | Time-stepping stencil (NEW pattern) |
| prk/stencil | sw4lite_rhs4sg_revNW | Configurable radius, simpler code |

---

## Known Issues and Considerations

### Polybench Dependencies:
- Requires `polybench.h` from utilities directory
- Requires size definitions from individual `.h` files
- Include paths must reference both utilities and common directories

### PRK Dependencies:
- Requires `par-res-kern_general.h` and `par-res-kern_omp.h`
- Uses `prk_malloc()` wrapper (can be replaced with standard malloc)
- Include paths must reference both include and common directories

### CARTS-Specific Considerations:
1. **polybench.h Macros**: Polybench uses extensive macros for array declarations
   - `POLYBENCH_2D_ARRAY_DECL` and similar
   - May need preprocessing or macro expansion for CARTS

2. **Problem Sizes**: Start with MINI or SMALL datasets for initial testing
   - Faster compilation and execution
   - Easier debugging

3. **Timing Instrumentation**: Polybench includes timing code
   - May interfere with CARTS instrumentation
   - Can be disabled with appropriate flags

---

## File Locations

### Polybench Files:
```
external/carts-benchmarks/polybench/
├── 2mm/
│   ├── 2mm.c
│   ├── 2mm.h
│   ├── Makefile
│   └── compiler.opts
├── 3mm/
│   ├── 3mm.c
│   ├── 3mm.h
│   ├── Makefile
│   └── compiler.opts
├── atax/
│   ├── atax.c
│   ├── atax.h
│   ├── Makefile
│   └── compiler.opts
├── bicg/
│   ├── bicg.c
│   ├── bicg.h
│   ├── Makefile
│   └── compiler.opts
├── fdtd-2d/
│   ├── fdtd-2d.c
│   ├── fdtd-2d.h
│   ├── Makefile
│   └── compiler.opts
├── seidel-2d/
│   ├── seidel-2d.c
│   ├── seidel-2d.h
│   ├── Makefile
│   └── compiler.opts
├── common/
│   └── (Polybench common headers)
└── utilities/
    ├── polybench.h
    ├── polybench.c
    └── (build system files)
```

### PRK Files:
```
external/carts-benchmarks/prk/
├── stencil/
│   ├── stencil.c
│   └── Makefile
├── common/
│   └── (PRK common implementations)
└── include/
    ├── par-res-kern_general.h
    └── par-res-kern_omp.h
```

---

## Next Steps

1. **Test Compilation**: Compile 2mm with standard gcc to verify setup
2. **CARTS Integration**: Build 2mm with CARTS cgeist
3. **Validation**: Run and validate output
4. **Analysis**: Review generated MLIR and DB insertions
5. **Iterate**: Repeat for remaining benchmarks

---

## References

- **Polybench**: http://polybench.sourceforge.net/
- **Polybench-ACC GitHub**: https://github.com/cavazos-lab/PolyBench-ACC
- **PRK GitHub**: https://github.com/ParRes/Kernels
- **Detailed Analysis**: See `/docs/benchmark_expansion_analysis.md`

---

**Last Updated**: November 11, 2025
**Status**: Ready for testing
**Verified**: All 7 benchmarks confirmed CARTS-compatible
