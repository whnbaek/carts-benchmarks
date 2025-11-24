# TASKS

Lets carefully analyze what is wrong with the

```bash
   carts benchmarks kastors/jacobi/jacobi-for small example.
```

Check that we are getting an error .

```txt
    [ERROR] [carts_run] Error when creating Dbs
```

Lets carefully analyze what is going on.

To manually replicate the error run:

```bash
  cd /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/jacobi/jacobi-for
  carts cgeist jacobi-for.c -O0 --print-debug-info -S --raise-scf-to-affine &> jacobi-for.mlir
  carts run jacobi-for.mlir --collect-metadata
  carts cgeist jacobi-for.c -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine &> jacobi-for.mlir
  carts run jacobi-for.mlir --pre-lowering --debug-only=canonicalize_memrefs&> dotproduct_edt_lowering.mlir
```

I think the problem is how we access the information. If you check the canonicalize_dbs pass we do not provide support for strided accesses, and we expect the code to be with the correct dimensionality. We should be able to indetify that, and fail the canonicalizememref pass if we have memrefs in data layouts we can not handle.

After that lets fix the carts benckmark code to make sure whether the code actually failed or not. Right now we returned that it was a success, even if it failed. Ultrathink

``` bash
  carts benchmarks kastors/jacobi/jacobi-for small
```

```txt
[ERROR] [carts_run] Error when creating Dbs
"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "main"}> ({
    ...
    "func.return"(%5) : (i32) -> ()
  }) {...} : () -> ()
[SUCCESS] Build completed successfully

[INFO] Generated artifacts in /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/jacobi/jacobi-for/build:
  - Parallel MLIR: /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/jacobi/jacobi-for/build/jacobi-for.mlir
  - Sequential MLIR: /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/jacobi/jacobi-for/build/jacobi-for_seq.mlir
  - Metadata: /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/jacobi/jacobi-for/build/jacobi-for.carts-metadata.json

[SUCCESS] Benchmark ready: jacobi/jacobi-for
[SUCCESS] Primary artifact: /Users/randreshg/Documents/carts/external/carts-benchmarks/kastors/jacobi/jacobi-for/build/jacobi-for.mlir
```

Lets make sure we mention there was an error building the artifact.

Finally, lets modify the example, as an array of array of ptrs that we already canonicalize so it works with CARTS.

```bash
  carts run jacobi-for.mlir --pre-lowering --debug-only=create_dbs,parallel_edt_lowering &> dotproduct_edt_lowering.mlir
```
