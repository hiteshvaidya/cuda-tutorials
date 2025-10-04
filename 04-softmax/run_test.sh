#!/bin/bash

# Softmax Optimization Benchmark Runner
# This script runs comprehensive benchmarks and generates a report

set -e  # Exit on error

echo "========================================="
echo "  CUDA Softmax Optimization Benchmark"
echo "========================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. No NVIDIA GPU detected."
    exit 1
fi

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# Build the project
echo "Building project..."
make clean
make
echo ""

# Create results directory
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.txt"

echo "Running benchmarks..."
echo "Results will be saved to: $RESULT_FILE"
echo ""

# Function to run benchmark and save results
run_benchmark() {
    local rows=$1
    local cols=$2
    local iters=$3
    local desc=$4
    
    echo "================================" | tee -a $RESULT_FILE
    echo "Test: $desc" | tee -a $RESULT_FILE
    echo "Size: ${rows}x${cols}, Iterations: ${iters}" | tee -a $RESULT_FILE
    echo "================================" | tee -a $RESULT_FILE
    ./softmax_benchmark $rows $cols $iters | tee -a $RESULT_FILE
    echo "" | tee -a $RESULT_FILE
}

# Header for results file
echo "CUDA Softmax Optimization Benchmark Results" > $RESULT_FILE
echo "Generated: $(date)" >> $RESULT_FILE
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $RESULT_FILE
echo "" >> $RESULT_FILE

# Benchmark Suite
# Small test (quick verification)
run_benchmark 128 100 10 "Small Test (Verification)"

# Medium tests (typical batch sizes)
run_benchmark 256 1000 100 "Medium - Batch 256, 1K classes"
run_benchmark 512 1000 100 "Medium - Batch 512, 1K classes"
run_benchmark 1024 1000 100 "Medium - Batch 1024, 1K classes"

# Large feature dimension tests
run_benchmark 256 4096 100 "Large Features - Batch 256, 4K features"
run_benchmark 512 4096 100 "Large Features - Batch 512, 4K features"

# Large batch tests
run_benchmark 2048 1000 100 "Large Batch - Batch 2K, 1K classes"
run_benchmark 4096 1000 50 "Large Batch - Batch 4K, 1K classes"

# Stress test (very large)
run_benchmark 4096 4096 10 "Stress Test - 4Kx4K"

echo "================================" | tee -a $RESULT_FILE
echo "Benchmark Complete!" | tee -a $RESULT_FILE
echo "Results saved to: $RESULT_FILE" | tee -a $RESULT_FILE
echo "================================" | tee -a $RESULT_FILE

# Generate summary
echo ""
echo "Generating summary..."
python3 - <<EOF
import re
import sys

# Read the results file
with open("$RESULT_FILE", "r") as f:
    content = f.read()

# Parse benchmark results
pattern = r"([\w\s]+?)\s*:\s*([\d.]+)\s*ms.*?(PASSED|FAILED)"
matches = re.findall(pattern, content)

if matches:
    print("\n=== Performance Summary ===\n")
    print(f"{'Kernel':<30} {'Avg Time (ms)':<15} {'Status':<10}")
    print("-" * 55)
    
    # Group by kernel type
    from collections import defaultdict
    kernel_times = defaultdict(list)
    
    for kernel, time, status in matches:
        kernel = kernel.strip()
        kernel_times[kernel].append(float(time))
    
    # Calculate average across all test sizes
    for kernel, times in kernel_times.items():
        avg_time = sum(times) / len(times)
        print(f"{kernel:<30} {avg_time:<15.4f} {'✓' if all('PASSED' in content for _ in times) else '✗':<10}")
    
    print("\n" + "=" * 55)
    print(f"\nTotal tests: {len(matches) // 4}")
    print(f"Results file: $RESULT_FILE\n")
else:
    print("No benchmark results found.")
    sys.exit(1)
EOF

echo ""
echo "Done! Check the results directory for detailed output."
