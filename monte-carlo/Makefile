################################################################################
# Monte Carlo Suite Makefile - Builds all Monte Carlo benchmark variants
################################################################################

EXAMPLES := ensemble

.PHONY: all clean $(EXAMPLES)

all: $(EXAMPLES)

$(EXAMPLES):
	@echo "Building Monte-Carlo/$@..."
	@$(MAKE) -C $@ all

clean:
	@for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Cleaning Monte-Carlo/$$dir..."; \
			$(MAKE) -C $$dir clean; \
		fi; \
	done
