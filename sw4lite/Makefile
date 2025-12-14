################################################################################
# SW4Lite Suite Makefile - Builds all SW4Lite benchmark variants
################################################################################

EXAMPLES := rhs4sg-base rhs4sg-revnw vel4sg-base

.PHONY: all clean $(EXAMPLES)

all: $(EXAMPLES)

$(EXAMPLES):
	@echo "Building SW4Lite/$@..."
	@$(MAKE) -C $@ all

clean:
	@for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			echo "Cleaning SW4Lite/$$dir..."; \
			$(MAKE) -C $$dir clean; \
		fi; \
	done
