################################################################################
# SW4Lite defaults layered on top of the shared CARTS pipeline.
################################################################################

SW4LITE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCHMARKS_ROOT := $(abspath $(SW4LITE_ROOT)/..)
SW4LITE_INCLUDES ?= -I$(SW4LITE_ROOT)/common

SEQ_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(SW4LITE_INCLUDES) $(CFLAGS)
PAR_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S -fopenmp $(SW4LITE_INCLUDES) $(CFLAGS)

ifneq ($(wildcard $(SW4LITE_ROOT)/arts.cfg),)
  ARTS_CFG ?= $(SW4LITE_ROOT)/arts.cfg
endif

# Size configurations for SW4Lite benchmarks
# Grid dimensions: NX, NY, NZ
# SMALL: 1000 elements, MEDIUM: 10000 elements, LARGE: 100000 elements
.PHONY: small medium large

small:
	@echo "[$(EXAMPLE_NAME)] Building with SMALL size (10x10x10 = 1000 elements)"
	$(MAKE) all openmp CFLAGS="-DNX=10 -DNY=10 -DNZ=10 $(EXTRA_CFLAGS)"

medium:
	@echo "[$(EXAMPLE_NAME)] Building with MEDIUM size (21x21x22 ≈ 10000 elements)"
	$(MAKE) all openmp CFLAGS="-DNX=21 -DNY=21 -DNZ=22 $(EXTRA_CFLAGS)"

large:
	@echo "[$(EXAMPLE_NAME)] Building with LARGE size (46x46x47 ≈ 100000 elements)"
	$(MAKE) all openmp CFLAGS="-DNX=46 -DNY=46 -DNZ=47 $(EXTRA_CFLAGS)"

include $(BENCHMARKS_ROOT)/common/carts.mk
