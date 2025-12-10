# KaStORS Jacobi - Poisson Equation Solver

## Description

Iterative Jacobi solver for the 2D Poisson equation using finite differences. Implements both OpenMP parallel-for and task-based dependency versions.

## Original Source

**Repository**: [KaStORS (Karlsruhe OpenMP Tasks Suite)](https://github.com/viroulep/kastors)
**Original Name**: Barcelona OpenMP Tasks Suite (BOTS)
**License**: GNU General Public License v2.0
**Copyright**:
- Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
- Universitat Politecnica de Catalunya

**KaStORS Project**:
- **Repository**: https://github.com/viroulep/kastors
- **Purpose**: OpenMP 4.0 task dependency benchmarks
- **Features**: Task-based parallelism with explicit dependencies

## Algorithm

Solves the Poisson equation ∇²u = f on a 2D grid using Jacobi iteration:

```
For each iteration:
  1. Save current estimate: u_old = u

  2. Update interior points using 5-point stencil:
     u[i][j] = 0.25 * (u_old[i-1][j] + u_old[i+1][j] +
                        u_old[i][j-1] + u_old[i][j+1] +
                        f[i][j] * dx * dy)

  3. Keep boundary conditions: u[boundary] = f[boundary]

  4. Repeat until convergence
```

## Variants

### 1. jacobi-for
- **Parallelization**: `#pragma omp parallel for`
- **Pattern**: Data-parallel across grid rows
- **Best for**: Multi-node execution (in CARTS context)

### 2. jacobi-task-dep
- **Parallelization**: `#pragma omp task depend(in:...) depend(out:...)`
- **Pattern**: Block-based with explicit dependencies
- **Best for**: Single-node execution with fine-grained control

### 3. poisson-for
- **Description**: Variant with different initialization
- **Parallelization**: Similar to jacobi-for

### 4. poisson-task
- **Description**: Task-based variant with different problem setup
- **Parallelization**: Similar to jacobi-task-dep

## Build

```bash
cd kastors/jacobi/

# For parallel-for version
cd jacobi-for/
make

# For task-dependency version
cd jacobi-task-dep/
make
```

## Usage

```bash
# Jacobi-for version
./jacobi-for <nx> <ny> <iterations>

# Example: 512×512 grid, 100 iterations
./jacobi-for 512 512 100
```

## Mathematical Background

**Poisson Equation**: ∇²u = f

In 2D with finite differences (5-point stencil):
```
(u[i-1][j] - 2*u[i][j] + u[i+1][j])/dx² +
(u[i][j-1] - 2*u[i][j] + u[i][j+1])/dy² = f[i][j]
```

Rearranging for u[i][j] (assuming dx = dy):
```
u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - f[i][j]*h²)
```

## Use in Computing

Poisson equation solvers are fundamental in:
- **Computational Fluid Dynamics**: Pressure field computation
- **Electromagnetics**: Electric/magnetic potential
- **Heat Transfer**: Steady-state temperature distribution
- **Image Processing**: Seamless cloning, gradient domain editing
- **Graphics**: Mesh smoothing, surface reconstruction

## Key Features

- **Iterative solver**: Jacobi method (simpler than Gauss-Seidel)
- **5-point stencil**: Classic finite difference pattern
- **Block-based**: Task version uses spatial blocking
- **Convergence**: Iterates until solution stabilizes

## References

- **KaStORS Repository**: https://gitlab.inria.fr/openmp/kastors
- **IWOMP 2014 Paper**: "The KASTORS Benchmark Suite: Evaluating Iterative Kahn Process Networks on Multicore"
- **Jacobi Method**: Named after Carl Gustav Jacob Jacobi (1804-1851)
- **Finite Differences**: Standard numerical method for PDEs

## Citation

### KaStORS
```
Virouleau, Philippe, et al.
"Evaluation of OpenMP dependent tasks with the KASTORS benchmark suite."
International Workshop on OpenMP (IWOMP), 2014.
```

### Original BOTS
```
Duran, Alejandro, et al.
"Barcelona OpenMP Tasks Suite: A set of benchmarks targeting the exploitation of task parallelism in OpenMP."
International Conference on Parallel Processing (ICPP), 2009.
```

### Jacobi Method
```
Jacobi, C.G.J.
"Über ein leichtes Verfahren die in der Theorie der Säcularstörungen vorkommenden Gleichungen numerisch aufzulösen."
Crelle's Journal, 1846.
```
