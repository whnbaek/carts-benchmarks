################################################################################
# SPECFEM3D defaults layered on top of the shared CARTS pipeline.
################################################################################

SPECFEM3D_TEMPLATE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SPECFEM3D_ROOT := $(abspath $(SPECFEM3D_TEMPLATE_DIR)/..)
CARTS_BENCH_ROOT := $(abspath $(SPECFEM3D_ROOT)/..)
SPECFEM3D_INCLUDES ?= -I$(SPECFEM3D_ROOT)/common

SEQ_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(SPECFEM3D_INCLUDES) $(CFLAGS)
PAR_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S -fopenmp $(SPECFEM3D_INCLUDES) $(CFLAGS)

ifneq ($(wildcard $(SPECFEM3D_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(SPECFEM3D_ROOT)/arts.cfg
endif

# Size configurations for SPECFEM3D benchmarks
# Grid dimensions: NGLLX, NGLLY, NGLLZ, NSPEC (number of spectral elements)
# SMALL: 1000 elements, MEDIUM: 10000 elements, LARGE: 100000 elements
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size (~1000 elements)"
	$(MAKE) all CFLAGS="-DNGLLX=5 -DNGLLY=5 -DNGLLZ=5 -DNSPEC=8 $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size (~10000 elements)"
	$(MAKE) all CFLAGS="-DNGLLX=5 -DNGLLY=5 -DNGLLZ=5 -DNSPEC=80 $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size (~100000 elements)"
	$(MAKE) all CFLAGS="-DNGLLX=5 -DNGLLY=5 -DNGLLZ=5 -DNSPEC=800 $(EXTRA_CFLAGS)"

include $(CARTS_BENCH_ROOT)/common/carts-pipeline.mk
