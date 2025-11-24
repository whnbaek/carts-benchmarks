################################################################################
# PolyBench specific defaults layered on top of the shared CARTS pipeline.
################################################################################

POLYBENCH_TEMPLATE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
POLYBENCH_ROOT := $(abspath $(POLYBENCH_TEMPLATE_DIR)/../..)
POLYBENCH_INCLUDES ?= -I. -I../common -I../utilities
BASE_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(POLYBENCH_INCLUDES)
SEQ_FLAGS ?= $(BASE_FLAGS) $(CFLAGS)
PAR_FLAGS ?= $(BASE_FLAGS) -fopenmp $(CFLAGS)

# Size configurations for PolyBench benchmarks
# SMALL: 128x128 matrices, MEDIUM: 1024x1024 matrices, LARGE: 2000x2000 matrices
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL_DATASET (128x128)"
	$(MAKE) all CFLAGS="-DSMALL_DATASET $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with STANDARD_DATASET (1024x1024)"
	$(MAKE) all CFLAGS="-DSTANDARD_DATASET $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE_DATASET (2000x2000)"
	$(MAKE) all CFLAGS="-DLARGE_DATASET $(EXTRA_CFLAGS)"

include $(POLYBENCH_ROOT)/common/carts-pipeline.mk
