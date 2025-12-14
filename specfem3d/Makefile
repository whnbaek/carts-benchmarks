################################################################################
# SPECFEM3D Suite Makefile - Builds all SPECFEM3D benchmark variants
################################################################################

EXAMPLES := stress velocity

.PHONY: all clean $(EXAMPLES)

all: $(EXAMPLES)

$(EXAMPLES):
	@echo "Building SPECFEM3D/$@..."
	@$(MAKE) -C $@ all

clean:
	@for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Cleaning SPECFEM3D/$$dir..."; \
			$(MAKE) -C $$dir clean; \
		fi; \
	done
