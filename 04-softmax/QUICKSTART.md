# 🚀 Quick Start Guide

## First Time Setup (5 minutes)

### 1. Check Your GPU
```bash
nvidia-smi
```
You should see your GPU listed. Note the compute capability.

### 2. Verify CUDA Installation
```bash
nvcc --version
```
Should show CUDA toolkit version (11.0+ recommended).

### 3. Update Architecture Flag (if needed)
Edit `Makefile` line 10 based on your GPU:

| GPU Model | Compute Capability | Flag |
|-----------|-------------------|------|
| RTX 4090/4080 | 8.9 | `-arch=sm_89` |
| RTX 3090/3080 | 8.6 | `-arch=sm_86` |
| RTX 2080 Ti | 7.5 | `-arch=sm_75` |
| GTX 1080 Ti | 6.1 | `-arch=sm_61` |

Find yours: `nvidia-smi --query-gpu=compute_cap --format=csv`

### 4. Build
```bash
cd 04-softmax
make
```

### 5. Run Quick Test
```bash
make test
```

Expected output:
```
=== Softmax Optimization Benchmark ===
Problem size: 128 x 100
Iterations: 10

Naive Softmax            : 0.0205 ms - PASSED ✓
Coalesced Memory         : 0.0118 ms - PASSED ✓
Optimized Block Reduce   : 0.0090 ms - PASSED ✓
Online Softmax           : 0.0106 ms - PASSED ✓
```

All should show **PASSED**! 🎉

---

## Understanding the Code (30 minutes)

### Step 1: Read the Naive Implementation
Open `main.cu`, find the `softmax_naive` kernel (line 63).

**Ask yourself:**
- How does it process data? (Sequential!)
- Why is it slow? (Single thread, multiple passes)
- What's the time complexity? (O(n) per row)

### Step 2: Compare with Coalesced Version
Find `softmax_coalesced` (line 103).

**Notice:**
- Multiple threads per row
- Parallel reduction in shared memory
- Coalesced memory access pattern

**Experiment:**
```bash
# Run with different problem sizes
./softmax_benchmark 256 1000 100
./softmax_benchmark 1024 4096 100
```

Watch how the speedup changes!

### Step 3: Study Warp Primitives
Find `softmax_optimized` (line 178).

**Key insight:**
```cuda
__shfl_down_sync(0xffffffff, val, offset)
```
This is **faster** than shared memory within a warp!

### Step 4: Master Online Softmax
Find `softmax_online` (line 258).

**The magic:**
```cuda
thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
```
Updates sum while computing max - single pass!

---

## Benchmarking (15 minutes)

### Run Full Suite
```bash
./run_test.sh
```

This will:
1. Test multiple problem sizes
2. Generate performance report
3. Save results to `results/` directory

### View Results
```bash
cat results/benchmark_*.txt
```

### Custom Benchmark
```bash
# Syntax: ./softmax_benchmark [rows] [cols] [iterations]

# Simulate ImageNet (batch=256, 1000 classes)
./softmax_benchmark 256 1000 100

# Simulate BERT attention (seq_len=512, d_model=768)
./softmax_benchmark 512 768 100

# Large batch training
./softmax_benchmark 4096 1000 100
```

---

## Profiling (Advanced, 20 minutes)

### Timeline Profile (Nsight Systems)
```bash
make profile-nsys
```

Opens interactive viewer showing:
- Kernel execution timeline
- Memory transfers
- GPU utilization

### Kernel Analysis (Nsight Compute)
```bash
make profile-ncu
```

Shows detailed metrics:
- Memory throughput
- Occupancy
- Warp efficiency
- Bottleneck analysis

**Tip:** Focus on one kernel:
```bash
ncu --kernel-name softmax_optimized ./softmax_benchmark 1024 1000 1
```

---

## Common Issues & Solutions

### Build Error: "nvcc: command not found"
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or activate conda environment with CUDA
conda activate cuda
```

### Runtime Error: "no CUDA-capable device"
- Check `nvidia-smi` works
- Ensure GPU isn't being used by another process
- Try `sudo nvidia-smi -pm 1` (enable persistence mode)

### Benchmark Shows "FAILED"
- First time? Might be numerical precision. Check the diff value.
- If diff > 1e-3: likely a bug
- If diff < 1e-5: adjust epsilon in code (line 327)

### Slow Performance
1. Check GPU isn't throttling: `nvidia-smi` (temp should be < 80°C)
2. Ensure correct architecture flag in Makefile
3. Close other GPU applications
4. Try smaller problem size first

---

## Learning Path

### Day 1: Foundations (You are here! 🎯)
- ✅ Build and run all kernels
- ✅ Understand naive vs. optimized
- ✅ Read code with comments
- ✅ Run benchmarks

### Day 2: Deep Dive
- Read `README.md` mathematical background
- Modify code: try different block sizes
- Profile with `nsys` and `ncu`
- Understand warp-level primitives

### Day 3: Extend
- Read `EXTENSION_GUIDE.md`
- Try kernel fusion (softmax + another operation)
- Experiment with different data types (FP16)
- Integrate into a simple CNN

### Week 2: Advanced
- Implement FlashAttention-style tiling
- Add support for transformers
- Optimize for your specific use case
- Share your results!

---

## Next Steps

1. **Understand the code**: Don't just run it, read it!
2. **Experiment**: Change block sizes, problem sizes, see what happens
3. **Profile**: Use Nsight tools to see where time is spent
4. **Extend**: Try the CNN integration in `EXTENSION_GUIDE.md`
5. **Share**: Post your results, help others learn

---

## Resources

- **Full documentation**: `README.md`
- **Extension guide**: `EXTENSION_GUIDE.md`
- **Code**: `main.cu` (heavily commented!)
- **Build options**: `Makefile` (see `make help`)

---

## Getting Help

If stuck:
1. Check `README.md` Troubleshooting section
2. Review code comments in `main.cu`
3. Run with debug flags: `NVCC_FLAGS = -g -G -arch=sm_89` in Makefile
4. Use `cuda-gdb` for debugging

---

**Happy Learning! You've got this! 💪**

The best way to learn CUDA is to:
1. Run the code ✓
2. Understand why it works
3. Modify and experiment
4. Build something new

You're already on step 1. Keep going! 🚀
