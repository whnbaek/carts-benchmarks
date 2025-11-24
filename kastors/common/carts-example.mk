################################################################################
# KaStORS defaults layered on top of the shared CARTS pipeline.
################################################################################

KASTORS_TEMPLATE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
KASTORS_ROOT := $(abspath $(KASTORS_TEMPLATE_DIR)/..)
CARTS_BENCH_ROOT := $(abspath $(KASTORS_ROOT)/..)
KASTORS_INCLUDES ?= -I$(KASTORS_ROOT)/common

SEQ_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(KASTORS_INCLUDES) $(CFLAGS)
PAR_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S -fopenmp $(KASTORS_INCLUDES) $(CFLAGS)

ifneq ($(wildcard $(KASTORS_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(KASTORS_ROOT)/arts.cfg
endif

# Size configurations for KaStORS benchmarks
# SMALL: 1024, MEDIUM: 4096, LARGE: 8192 (matrix/grid dimensions)
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size (1024)"
	$(MAKE) all CFLAGS="$(CFLAGS) -DSIZE=1024 $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size (4096)"
	$(MAKE) all CFLAGS="$(CFLAGS) -DSIZE=4096 $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size (8192)"
	$(MAKE) all CFLAGS="$(CFLAGS) -DSIZE=8192 $(EXTRA_CFLAGS)"

include $(CARTS_BENCH_ROOT)/common/carts-pipeline.mk
