# Makefile for Softmax Optimization Tutorial
# This builds the CUDA softmax benchmark program

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_89 -lineinfo
# -O3: Maximum optimization
# -arch=sm_89: Target Ada Lovelace architecture (RTX 4080)
# -lineinfo: Enable line-level profiling info

# For debugging, uncomment these flags:
# NVCC_FLAGS = -g -G -arch=sm_89
# -g: Host debug info
# -G: Device debug info

# Target executable
TARGET = softmax_benchmark

# Source files
SOURCES = main.cu

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET)

# Run the benchmark with default parameters
run: $(TARGET)
	./$(TARGET)

# Run with custom parameters (rows cols iterations)
# Example: make run-custom ROWS=2048 COLS=2000 ITERS=200
run-custom: $(TARGET)
	./$(TARGET) $(ROWS) $(COLS) $(ITERS)

# Run small test (quick verification)
test: $(TARGET)
	./$(TARGET) 128 100 10

# Run large test (stress test)
test-large: $(TARGET)
	./$(TARGET) 4096 4096 100

# Profile with nvprof (for older GPUs)
profile: $(TARGET)
	nvprof ./$(TARGET) 1024 1000 10

# Profile with nsys (for newer GPUs)
profile-nsys: $(TARGET)
	nsys profile --stats=true ./$(TARGET) 1024 1000 10

# Profile with ncu (compute profiler)
profile-ncu: $(TARGET)
	ncu --set full ./$(TARGET) 1024 1000 1

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Clean profiling reports
clean-profile:
	rm -f *.qdrep *.sqlite *.nsys-rep

# Clean everything
clean-all: clean clean-profile

# Show GPU info
gpu-info:
	nvidia-smi

# Help target
help:
	@echo "Available targets:"
	@echo "  make                - Build the benchmark"
	@echo "  make run            - Run with default parameters (1024x1000)"
	@echo "  make test           - Quick test with small size"
	@echo "  make test-large     - Stress test with large size"
	@echo "  make run-custom     - Run with custom ROWS, COLS, ITERS"
	@echo "  make profile        - Profile with nvprof"
	@echo "  make profile-nsys   - Profile with Nsight Systems"
	@echo "  make profile-ncu    - Profile with Nsight Compute"
	@echo "  make gpu-info       - Show GPU information"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make clean-all      - Remove all generated files"
	@echo ""
	@echo "Example custom run:"
	@echo "  make run-custom ROWS=2048 COLS=2000 ITERS=200"

.PHONY: all run run-custom test test-large profile profile-nsys profile-ncu clean clean-profile clean-all gpu-info help
