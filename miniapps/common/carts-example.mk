################################################################################
# Miniapps defaults layered on top of the shared CARTS pipeline.
################################################################################

TEMPLATE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MINIAPPS_ROOT := $(abspath $(TEMPLATE_DIR)/..)
CARTS_BENCH_ROOT := $(abspath $(MINIAPPS_ROOT)/..)

# Add any category-specific includes or flags here
SEQ_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(CFLAGS)
PAR_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S -fopenmp $(CFLAGS)

# Size configurations for miniapp benchmarks
# SMALL: N=1000, MEDIUM: N=10000, LARGE: N=100000
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size (N=1000)"
	$(MAKE) all CFLAGS="-DN=1000 $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size (N=10000)"
	$(MAKE) all CFLAGS="-DN=10000 $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size (N=100000)"
	$(MAKE) all CFLAGS="-DN=100000 $(EXTRA_CFLAGS)"

include $(CARTS_BENCH_ROOT)/common/carts-pipeline.mk
