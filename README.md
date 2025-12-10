# CARTS Benchmarks

Benchmark suite for the CARTS system (HPC, ML kernels, stencils, task-parallel apps).

## Quick Start

- Discover commands, options, sizes, and available benchmarks:
  - `carts benchmarks --help`
  - `carts benchmarks list`

### Build examples

- CARTS pipeline (default): `carts benchmarks polybench/2mm small`
- Standard OpenMP: `carts benchmarks polybench/2mm --openmp`

### Clean

- `carts benchmarks clean polybench/2mm`
- `carts benchmarks clean --all`

## Suites (top-level)

- `polybench/`, `ml-kernels/`, `sw4lite/`, `specfem3d/`, `seissol/`
- `kastors-jacobi/`, `task-parallelism-omp/`, `simple-kernels/`
- `miniapps/`, `npb/`, `prk/`, `llama2-transformer/`
