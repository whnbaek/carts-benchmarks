################################################################################
# Graph500 Suite Makefile - Builds all Graph500-inspired benchmark variants
################################################################################

EXAMPLES := graph-gen

.PHONY: all clean $(EXAMPLES)

all: $(EXAMPLES)

$(EXAMPLES):
	@echo "Building Graph500/$@..."
	@$(MAKE) -C $@ all

clean:
	@for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Cleaning Graph500/$$dir..."; \
			$(MAKE) -C $$dir clean; \
		fi; \
	done
