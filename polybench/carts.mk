################################################################################
# PolyBench specific defaults layered on top of the shared CARTS pipeline.
################################################################################

POLYBENCH_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(POLYBENCH_ROOT)/..)
POLYBENCH_INCLUDES ?= -I. -I../common -I../utilities
BASE_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(POLYBENCH_INCLUDES)
SEQ_FLAGS ?= $(BASE_FLAGS) $(CFLAGS)
PAR_FLAGS ?= $(BASE_FLAGS) -fopenmp $(CFLAGS)
OMP_FLAGS ?= -fopenmp -O3 $(CFLAGS) $(POLYBENCH_INCLUDES) -lm -lcartsbenchmarks

# Size configurations for PolyBench benchmarks
# SMALL: 128x128 matrices, MEDIUM: 1024x1024 matrices, LARGE: 2000x2000 matrices
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL_DATASET (128x128)"
	$(MAKE) all openmp CFLAGS="-DSMALL_DATASET $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with STANDARD_DATASET (1024x1024)"
	$(MAKE) all openmp CFLAGS="-DSTANDARD_DATASET $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE_DATASET (2000x2000)"
	$(MAKE) all openmp CFLAGS="-DLARGE_DATASET $(EXTRA_CFLAGS)"

include $(BENCHMARKS_ROOT)/common/carts.mk
