################################################################################
# SeisSol Suite Makefile - Builds all SeisSol benchmark variants
################################################################################

EXAMPLES := volume-integral

.PHONY: all clean $(EXAMPLES)

all: $(EXAMPLES)

$(EXAMPLES):
	@echo "Building SeisSol/$@..."
	@$(MAKE) -C $@ all

clean:
	@for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Cleaning SeisSol/$$dir..."; \
			$(MAKE) -C $$dir clean; \
		fi; \
	done
