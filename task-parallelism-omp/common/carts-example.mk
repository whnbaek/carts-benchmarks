################################################################################
# Task Parallelism OMP defaults layered on top of the shared CARTS pipeline.
################################################################################

TASK_OMP_TEMPLATE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TASK_OMP_ROOT := $(abspath $(TASK_OMP_TEMPLATE_DIR)/..)
CARTS_BENCH_ROOT := $(abspath $(TASK_OMP_ROOT)/..)

SEQ_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(CFLAGS)
PAR_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S -fopenmp $(CFLAGS)

# Note: These benchmarks use runtime parameters rather than compile-time sizes
# Size configurations documented in individual READMEs
# SMALL: 10000 elements, MEDIUM: 100000 elements, LARGE: 1000000 elements

include $(CARTS_BENCH_ROOT)/common/carts-pipeline.mk
