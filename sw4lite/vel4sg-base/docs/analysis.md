# vel4sg_base example analysis

Walk through these steps to debug CARTS pipeline issues.

## 1. Navigate to the vel4sg-base example directory

```bash
cd ~/Documents/carts/external/carts-benchmarks/sw4lite/vel4sg-base
```

## 2. Build CARTS if any changes were made

```bash
carts build
```

## 3. Generate MLIR from C source

Generate the sequential MLIR (for metadata):
```bash
carts cgeist vel4sg_base.c -O0 --print-debug-info -S --raise-scf-to-affine -I../common 2>/dev/null > /tmp/vel4sg_seq.mlir
```

Generate the parallel MLIR (with OpenMP):
```bash
carts cgeist vel4sg_base.c -O0 --print-debug-info -S -fopenmp --raise-scf-to-affine -I../common 2>/dev/null > /tmp/vel4sg_par.mlir
```

## 4. Run pipeline stages incrementally

### Stage 1: collect-metadata (on sequential code)
```bash
carts run /tmp/vel4sg_seq.mlir --collect-metadata 2>&1 | head -20
```

### Stage 2: openmp-to-arts
```bash
carts run /tmp/vel4sg_par.mlir --openmp-to-arts > /tmp/vel4sg_arts.mlir 2>&1
```

### Stage 3: create-dbs
```bash
carts run /tmp/vel4sg_par.mlir --create-dbs > /tmp/vel4sg_dbs.mlir 2>&1
```

### Stage 4: concurrency
```bash
carts run /tmp/vel4sg_par.mlir --concurrency > /tmp/vel4sg_conc.mlir 2>&1
```

## 5. Build and run the example

```bash
make clean && make all
./vel4sg_base_arts
```

---

## Issues Fixed

### Issue 1: Switch statement not supported

**Original code:**
```c
static inline float derivative(const float ***arr, int i, int j, int k, int dir) {
  switch (dir) {
  case 0:
    return 0.5f * (arr[i + 1][j][k] - arr[i - 1][j][k]);
  case 1:
    return 0.5f * (arr[i][j + 1][k] - arr[i][j - 1][k]);
  default:
    return 0.5f * (arr[i][j][k + 1] - arr[i][j][k - 1]);
  }
}
```

**Fix:** Replace with separate inline functions:
```c
static inline float derivative_x(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i + 1][j][k] - arr[i - 1][j][k]);
}

static inline float derivative_y(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i][j + 1][k] - arr[i][j - 1][k]);
}

static inline float derivative_z(const float ***arr, int i, int j, int k) {
  return 0.5f * (arr[i][j][k + 1] - arr[i][j][k - 1]);
}
```

### Issue 2: Collapse clause not supported

**Original:**
```c
#pragma omp parallel for collapse(2) schedule(static)
```

**Fix:**
```c
#pragma omp parallel for schedule(static)
```

### Issue 3: Early return not supported

**Original:**
```c
if (!vx || !vy || !vz || !rho || !sxx || !syy || !szz || !sxy || !sxz || !syz) {
  fprintf(stderr, "allocation failure\n");
  return 1;
}
```

**Fix:** Remove the early return check.

### Issue 4: Makefile naming mismatch

**Original:**
```makefile
EXAMPLE_NAME := vel4sg-base
```

**Fix:**
```makefile
EXAMPLE_NAME := vel4sg_base
```

---

## Verification

```bash
cd ~/Documents/carts/external/carts-benchmarks/sw4lite/vel4sg-base
make clean && make all
./vel4sg_base_arts

# Expected output:
# sw4lite_vel4sg_base checksum=0.000000
```
