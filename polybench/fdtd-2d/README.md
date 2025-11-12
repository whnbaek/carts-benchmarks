# fdtd-2d - 2D Finite Difference Time Domain

## Description

2D FDTD solver for electromagnetic field simulation. Time-stepping stencil computation.

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/stencils/fdtd-2d/`

**Polybench Project**: http://polybench.sourceforge.net/

## Algorithm

3-point stencil applied iteratively over time steps for electric (ex, ey) and magnetic (hz) fields.

## References

- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **FDTD Method**: Yee, K. (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations"
