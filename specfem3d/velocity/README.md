# SPECFEM3D Velocity Update

- **Origin**: Based on the velocity-update routines in [SPECFEM3D](https://github.com/SPECFEM/specfem3d), where particle velocities are advanced by integrating the stress divergence.
- **Pattern**: Compute divergence of stress components and divide by density to update `vx`, `vy`, `vz`.
- **Goal**: Provide a companion kernel to `specfem3d_stress` so both halves of the elastic update loop are represented in CARTS.

## Build

```bash
make -C external/carts-benchmarks/specfem3d_velocity clean all
make -C external/carts-benchmarks/specfem3d_velocity CFLAGS="-DNX=72 -DNY=48 -DNZ=48 -DDT=5e-4" all
```
