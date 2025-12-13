################################################################################
# KaStORS specific defaults layered on top of the shared CARTS pipeline.
################################################################################

KASTORS_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(KASTORS_ROOT)/..)
KASTORS_INCLUDES ?= -I. -I../include
BASE_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(KASTORS_INCLUDES)
SEQ_FLAGS ?= $(BASE_FLAGS) $(CFLAGS)
PAR_FLAGS ?= $(BASE_FLAGS) -fopenmp $(CFLAGS)

# Size configurations for KaStORS benchmarks
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size"
	$(MAKE) all openmp CFLAGS="-DSMALL $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size"
	$(MAKE) all openmp CFLAGS="-DMEDIUM $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size"
	$(MAKE) all openmp CFLAGS="-DLARGE $(EXTRA_CFLAGS)"

include $(BENCHMARKS_ROOT)/common/carts.mk
