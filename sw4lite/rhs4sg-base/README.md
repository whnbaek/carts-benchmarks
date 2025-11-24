# SW4Lite RHS4SG (Base)

Simplified interior right-hand-side assembly kernel from SW4Lite, focusing on fourth-order accurate Laplacian computations with Lamé coefficients.

## Origin

- **Source**: [geodynamics/sw4lite](https://github.com/geodynamics/sw4lite) (LLNL)
- **Pattern**: 3D fourth-order accurate Laplacian with Lamé coefficients for seismic wave equations
- **Purpose**: Baseline interior stencil without one-sided boundary handling

## Build

### Standard sizes
```bash
make -C sw4lite/rhs4sg-base small   # 10×10×10 = 1,000 elements
make -C sw4lite/rhs4sg-base medium  # 21×21×22 ≈ 10,000 elements
make -C sw4lite/rhs4sg-base large   # 46×46×47 ≈ 100,000 elements
```

### Custom size
```bash
make -C sw4lite/rhs4sg-base CFLAGS="-DNX=64 -DNY=64 -DNZ=64" all
```

### All CARTS stages
```bash
make -C sw4lite/rhs4sg-base clean all
```

## Characteristics

- **Stencil**: 3D 27-point (4th-order accurate)
- **Grid**: Uniform Cartesian mesh
- **Operations**: Laplacian + Lamé material coefficients
- **Parallelization**: OpenMP parallel-for over grid points

## See Also

- **rhs4sg-revnw**: Full-featured optimized version with restrict pointers
- **vel4sg-base**: Complementary velocity update kernel
