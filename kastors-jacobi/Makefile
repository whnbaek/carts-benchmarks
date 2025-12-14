EXAMPLES := jacobi-for jacobi-task-dep poisson-for poisson-task

.PHONY: all clean $(EXAMPLES)

all: $(EXAMPLES)

$(EXAMPLES):
	@echo "==> Building $@"
	@$(MAKE) -C $@ all

clean:
	@for dir in $(EXAMPLES); do \
		if [ -d "$$dir" ]; then \
			$(MAKE) -C $$dir clean; \
		fi; \
	done
