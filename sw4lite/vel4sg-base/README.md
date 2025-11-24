# SW4Lite Velocity Update (Base)

- **Origin**: Derived from the `vel4sg` velocity update kernel in [geodynamics/sw4lite](https://github.com/geodynamics/sw4lite).
- **Pattern**: Update the three velocity components from stress divergence divided by density (staggered-grid scheme).
- **Purpose**: Completes the SW4Lite coverage so both RHS assembly and velocity updates can be profiled independently.

## Build

```bash
make -C external/carts-benchmarks/sw4lite_vel4sg_base clean all
make -C external/carts-benchmarks/sw4lite_vel4sg_base CFLAGS="-DNX=64 -DNY=64 -DNZ=64 -DDT=0.00025f" all
```

Artifacts follow the standard CARTS pipeline (MLIR in `build/`, logs in `logs/`).
