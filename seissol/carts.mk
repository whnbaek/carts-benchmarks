################################################################################
# SeisSol defaults layered on top of the shared CARTS pipeline.
################################################################################

SEISSOL_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(SEISSOL_ROOT)/..)
SEISSOL_INCLUDES ?= -I$(SEISSOL_ROOT)/common

SEQ_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(SEISSOL_INCLUDES) $(CFLAGS)
PAR_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S -fopenmp $(SEISSOL_INCLUDES) $(CFLAGS)

ifneq ($(wildcard $(SEISSOL_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(SEISSOL_ROOT)/arts.cfg
endif

# Size configurations for SeisSol benchmarks
# NELEM: number of elements to process
# SMALL: 1000 elements, MEDIUM: 10000 elements, LARGE: 100000 elements
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size (1000 elements)"
	$(MAKE) all openmp CFLAGS="-DNELEM=1000 $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size (10000 elements)"
	$(MAKE) all openmp CFLAGS="-DNELEM=10000 $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size (100000 elements)"
	$(MAKE) all openmp CFLAGS="-DNELEM=100000 $(EXTRA_CFLAGS)"

include $(BENCHMARKS_ROOT)/common/carts.mk
