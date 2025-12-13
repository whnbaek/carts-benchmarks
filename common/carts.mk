################################################################################
# Shared CARTS pipeline fragment used by all benchmark suites.
# Expected variables (set before include):
#   EXAMPLE_NAME - logical benchmark name (required)
#   SRC          - space separated list of C/C++ translation units (default:
#                  $(EXAMPLE_NAME).c)
# Optional knobs:
#   CFLAGS, SEQ_FLAGS, PAR_FLAGS, PIPELINE_DEPS, ARTS_CFG, BUILD_DIR, LOG_DIR,
#   METADATA, CONCURRENCY_FLAGS, CONCURRENCY_OPT_FLAGS, METADATA_FLAGS
################################################################################

SHELL ?= /bin/bash
.SHELLFLAGS ?= -eo pipefail -c

ifndef EXAMPLE_NAME
$(error EXAMPLE_NAME must be defined before including carts-pipeline.mk)
endif

SRC ?= $(EXAMPLE_NAME).c
CARTS ?= carts
BUILD_DIR ?= build
LOG_DIR ?= logs
SEQ_MLIR ?= $(BUILD_DIR)/$(EXAMPLE_NAME)_seq.mlir
PAR_MLIR ?= $(BUILD_DIR)/$(EXAMPLE_NAME).mlir
METADATA ?= $(BUILD_DIR)/$(EXAMPLE_NAME).carts-metadata.json
PIPELINE_DEPS ?=

SEQ_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S $(CFLAGS)
PAR_FLAGS ?= --print-debug-info --raise-scf-to-affine -O0 -S -fopenmp $(CFLAGS)
METADATA_FLAGS ?=
CONCURRENCY_FLAGS ?=
CONCURRENCY_OPT_FLAGS ?=

ARTS_CFG ?= $(firstword $(wildcard arts.cfg))
ifeq ($(strip $(ARTS_CFG)),)
  ARTS_CFG_ARG :=
  ARTS_CFG_NOTICE = @echo "[$(EXAMPLE_NAME)] warning: arts.cfg not found; continuing with CARTS runtime defaults."
else
  ARTS_CFG_ARG := --arts-config $(ARTS_CFG)
  ARTS_CFG_NOTICE = @:
endif

.PHONY: all seq metadata parallel concurrency concurrency-opt clean

all: seq metadata parallel concurrency concurrency-opt

seq: $(SEQ_MLIR)

metadata: $(METADATA)

parallel: $(PAR_MLIR)

concurrency: $(PAR_MLIR) $(METADATA) | $(LOG_DIR)
	$(ARTS_CFG_NOTICE)
	@echo "[$(EXAMPLE_NAME)] Running carts --concurrency"
	@$(CARTS) run $(PAR_MLIR) --metadata-file $(METADATA) $(ARTS_CFG_ARG) \
	  --concurrency $(CONCURRENCY_FLAGS) 2>&1 | tee $(LOG_DIR)/concurrency.log; \
	exit $${PIPESTATUS[0]}

concurrency-opt: $(PAR_MLIR) $(METADATA) | $(LOG_DIR)
	$(ARTS_CFG_NOTICE)
	@echo "[$(EXAMPLE_NAME)] Running carts --concurrency-opt"
	@$(CARTS) run $(PAR_MLIR) --metadata-file $(METADATA) $(ARTS_CFG_ARG) \
	  --concurrency-opt $(CONCURRENCY_OPT_FLAGS) 2>&1 | tee $(LOG_DIR)/concurrency-opt.log; \
	exit $${PIPESTATUS[0]}

# Standard OpenMP compilation (no ARTS transformation)
# Usage: make openmp or carts benchmarks <name> --openmp
OMP_BINARY ?= $(BUILD_DIR)/$(EXAMPLE_NAME)_omp
OMP_FLAGS ?= -fopenmp -O3 $(CFLAGS) -lm -lcartsbenchmarks

# Make openmp a proper file target to avoid rebuilding when binary exists
$(OMP_BINARY): $(SRC) $(PIPELINE_DEPS) | $(BUILD_DIR) $(LOG_DIR)
	@echo "[$(EXAMPLE_NAME)] Compiling with standard OpenMP -> $(OMP_BINARY)"
	@$(CARTS) clang $(SRC) $(OMP_FLAGS) -o $(OMP_BINARY) 2>&1 | tee $(LOG_DIR)/openmp.log; \
	exit $${PIPESTATUS[0]}

# Alias target for convenience
openmp: $(OMP_BINARY)

$(SEQ_MLIR): $(SRC) $(PIPELINE_DEPS) | $(BUILD_DIR) $(LOG_DIR)
	@echo "[$(EXAMPLE_NAME)] Generating sequential MLIR -> $@"
	$(ARTS_CFG_NOTICE)
	@$(CARTS) execute $(SRC) -O3 $(ARTS_CFG_ARG) $(SEQ_FLAGS) > $(LOG_DIR)/seq-stdout.log 2> $(LOG_DIR)/seq.log || (cat $(LOG_DIR)/seq.log >&2; exit 1)
	@if [ -f "$(EXAMPLE_NAME)_seq.mlir" ]; then mv "$(EXAMPLE_NAME)_seq.mlir" "$@"; fi
	@if [ -f ".carts-metadata.json" ]; then mv ".carts-metadata.json" "$(BUILD_DIR)/$(EXAMPLE_NAME).carts-metadata.json"; fi
	@cat $(LOG_DIR)/seq.log >&2

$(PAR_MLIR): $(SRC) $(PIPELINE_DEPS) | $(BUILD_DIR) $(LOG_DIR)
	@echo "[$(EXAMPLE_NAME)] Generating OpenMP MLIR -> $@"
	$(ARTS_CFG_NOTICE)
	@$(CARTS) execute $(SRC) -O3 $(ARTS_CFG_ARG) $(PAR_FLAGS) > $(LOG_DIR)/par-stdout.log 2> $(LOG_DIR)/par.log || (cat $(LOG_DIR)/par.log >&2; exit 1)
	@if [ -f "$(EXAMPLE_NAME).mlir" ]; then mv "$(EXAMPLE_NAME).mlir" "$@"; fi
	@if [ -f "$(EXAMPLE_NAME)-arts.ll" ]; then mv "$(EXAMPLE_NAME)-arts.ll" "$(BUILD_DIR)/"; fi
	@if [ -f "$(EXAMPLE_NAME)" ] && [ -x "$(EXAMPLE_NAME)" ]; then mv "$(EXAMPLE_NAME)" "$(BUILD_DIR)/"; fi
	@cat $(LOG_DIR)/par.log >&2

$(METADATA): $(SEQ_MLIR) | $(BUILD_DIR) $(LOG_DIR)
	@echo "[$(EXAMPLE_NAME)] Metadata already collected during compilation"
	@if [ ! -f "$@" ]; then \
		echo "[$(EXAMPLE_NAME)] Running additional metadata collection"; \
		$(CARTS) run $(SEQ_MLIR) --collect-metadata --metadata-file $(METADATA) \
		  $(METADATA_FLAGS) 2>&1 | tee $(LOG_DIR)/metadata.log; \
	fi

$(BUILD_DIR):
	@mkdir -p $@

$(LOG_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LOG_DIR)
