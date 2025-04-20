# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -arch=sm_86  # For RTX A6000

# Source files (all .cu files in the directory)
SRCS = $(wildcard *.cu)
# Target executables (replace .cu with no extension)
TARGETS = $(SRCS:.cu=)

# Default target: build all programs
all: $(TARGETS)

# Pattern rule for building any .cu file
%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run specific program
run-%: %
	./$<

# Run all programs
run-all: $(TARGETS)
	@for target in $(TARGETS); do \
		echo "Running $$target..."; \
		./$$target; \
		echo "-------------------"; \
	done

# Clean up all executables
clean:
	rm -f $(TARGETS)

# Help message
help:
	@echo "Available targets:"
	@echo "  make all           - Build all programs"
	@echo "  make <program>     - Build specific program (e.g., make 01-idxing)"
	@echo "  make run-<program> - Run specific program (e.g., make run-01-idxing)"
	@echo "  make run-all       - Run all programs"
	@echo "  make clean         - Remove all executables"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Available programs:"
	@for target in $(TARGETS); do \
		echo "  - $$target"; \
	done

.PHONY: all run-all clean help 