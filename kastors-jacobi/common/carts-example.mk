################################################################################
# KaStORS specific defaults layered on top of the shared CARTS pipeline.
################################################################################

KASTORS_TEMPLATE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
KASTORS_ROOT := $(abspath $(KASTORS_TEMPLATE_DIR)/../..)
KASTORS_INCLUDES ?= -I. -I../include
BASE_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(KASTORS_INCLUDES)
SEQ_FLAGS ?= $(BASE_FLAGS) $(CFLAGS)
PAR_FLAGS ?= $(BASE_FLAGS) -fopenmp $(CFLAGS)

# Size configurations for KaStORS benchmarks
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size"
	$(MAKE) all CFLAGS="-DSMALL $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size"
	$(MAKE) all CFLAGS="-DMEDIUM $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size"
	$(MAKE) all CFLAGS="-DLARGE $(EXTRA_CFLAGS)"

include $(KASTORS_ROOT)/common/carts-pipeline.mk
