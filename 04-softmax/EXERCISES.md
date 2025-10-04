# Learning Exercise Guide

## 🎯 How to Use This Tutorial for Maximum Learning

This guide provides hands-on exercises to deepen your understanding of CUDA optimization.

---

## Exercise 1: Understanding Thread Organization (Beginner)

### Goal
Understand how threads are organized and how they access memory.

### Tasks

1. **Print Thread Information**
   
   Add this to the beginning of `softmax_coalesced`:
   ```cuda
   if (blockIdx.x == 0 && threadIdx.x < 4) {
       printf("Block %d, Thread %d processing row %d\n", 
              blockIdx.x, threadIdx.x, row);
   }
   ```
   
   Rebuild and run with small size:
   ```bash
   ./softmax_benchmark 4 10 1
   ```
   
   **Question**: How many threads work on each row?

2. **Experiment with Block Size**
   
   In `main.cu`, line 413, change:
   ```cuda
   int block_size = 256;  // Try: 64, 128, 256, 512
   ```
   
   Run benchmark:
   ```bash
   ./softmax_benchmark 1024 1000 100
   ```
   
   **Question**: Which block size is fastest? Why?

---

## Exercise 2: Memory Access Patterns (Intermediate)

### Goal
Visualize and optimize memory access patterns.

### Tasks

1. **Compare Memory Access**
   
   Modify `softmax_naive` to print first few memory accesses:
   ```cuda
   if (row == 0) {
       for (int i = 0; i < 5; i++) {
           printf("Thread %d reading index %d (addr: %p)\n", 
                  row, i, &row_data[i]);
       }
   }
   ```
   
   Do the same for `softmax_coalesced` with thread 0-3.
   
   **Question**: What's the difference in access patterns?

2. **Measure Memory Bandwidth**
   
   Use Nsight Compute:
   ```bash
   ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
       --kernel-name softmax_naive ./softmax_benchmark 1024 1000 1
   
   ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
       --kernel-name softmax_coalesced ./softmax_benchmark 1024 1000 1
   ```
   
   **Question**: Which kernel reads/writes more efficiently?

---

## Exercise 3: Parallel Reduction (Intermediate)

### Goal
Master the parallel reduction pattern.

### Tasks

1. **Visualize Reduction Steps**
   
   Add debug output to `softmax_coalesced` reduction:
   ```cuda
   if (row == 0) {
       for (int s = blockDim.x / 2; s > 0; s >>= 1) {
           if (tid < s) {
               if (tid < 8) {  // Print first 8 threads
                   printf("Step s=%d: tid=%d combines max[%d]=%.2f with max[%d]=%.2f\n",
                          s, tid, tid, max_vals[tid], tid+s, max_vals[tid+s]);
               }
               max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
           }
           __syncthreads();
       }
   }
   ```
   
   **Question**: How many steps does reduction take? (Hint: log₂(threads))

2. **Implement Your Own Reduction**
   
   Create a new kernel that computes the sum (not max) using reduction:
   ```cuda
   __global__ void sum_reduction(const float* input, float* output, int n) {
       // Your code here!
       // Use shared memory and parallel reduction
   }
   ```
   
   **Challenge**: Make it handle any array size, not just powers of 2.

---

## Exercise 4: Warp-Level Primitives (Advanced)

### Goal
Understand and use warp shuffle operations.

### Tasks

1. **Compare Warp Shuffle vs Shared Memory**
   
   Modify `softmax_optimized` to use shared memory instead of shuffle.
   Measure the difference:
   ```bash
   ./softmax_benchmark 1024 1000 100
   ```
   
   **Question**: How much slower is shared memory? Why?

2. **Implement Warp-Level Sum**
   
   Create your own warp reduction for sum (not just max):
   ```cuda
   __inline__ __device__ float warp_reduce_sum(float val) {
       // Your code here!
       // Use __shfl_down_sync
   }
   ```
   
   **Hint**: It's almost identical to `warp_reduce_max`!

3. **Understand Warp Divergence**
   
   Add a conditional to create divergence:
   ```cuda
   if (threadIdx.x % 2 == 0) {
       // Even threads do this
   } else {
       // Odd threads do that
   }
   ```
   
   Profile with:
   ```bash
   ncu --metrics smsp__average_warps_issue_stalled_branch_resolving.pct \
       ./softmax_benchmark 1024 1000 1
   ```
   
   **Question**: How does divergence affect performance?

---

## Exercise 5: Online Algorithms (Advanced)

### Goal
Master the online softmax algorithm.

### Tasks

1. **Verify Correctness Manually**
   
   For a small example [1.0, 2.0, 3.0]:
   
   Step 1 (x=1.0):
   - max = 1.0
   - sum = exp(1.0 - 1.0) = 1.0
   
   Step 2 (x=2.0):
   - old_max = 1.0, new_max = 2.0
   - sum = 1.0 * exp(1.0 - 2.0) + exp(2.0 - 2.0)
   -     = 1.0 * exp(-1.0) + 1.0
   -     = 0.368 + 1.0 = 1.368
   
   **Task**: Complete step 3, verify with CPU calculation.

2. **Compare with Two-Pass**
   
   Modify `softmax_online` to count memory accesses:
   ```cuda
   __shared__ long long memory_reads;
   if (tid == 0) memory_reads = 0;
   __syncthreads();
   
   // Increment for each read
   atomicAdd(&memory_reads, 1);
   ```
   
   **Question**: How many reads vs two-pass approach?

3. **Implement Online Variance**
   
   Apply the same technique to compute variance online:
   ```cuda
   __global__ void online_variance(const float* input, 
                                   float* mean_out, 
                                   float* var_out, 
                                   int n) {
       // Use Welford's online algorithm
       // mean_k = mean_{k-1} + (x_k - mean_{k-1}) / k
       // M2_k = M2_{k-1} + (x_k - mean_{k-1}) * (x_k - mean_k)
   }
   ```

---

## Exercise 6: Kernel Fusion (Advanced)

### Goal
Combine multiple operations into a single kernel.

### Tasks

1. **Fuse Softmax + Cross-Entropy**
   
   Implement a kernel that computes both:
   ```cuda
   __global__ void softmax_cross_entropy_fused(
       const float* logits,
       const int* labels,
       float* loss,
       int batch_size,
       int num_classes
   ) {
       // 1. Compute softmax
       // 2. Immediately compute -log(softmax[label])
       // 3. Store loss
   }
   ```
   
   **Benefit**: Save one kernel launch and one memory read/write!

2. **Fuse Matrix Multiply + Softmax**
   
   For the final layer of a neural network:
   ```cuda
   __global__ void matmul_softmax_fused(
       const float* input,    // [batch, input_dim]
       const float* weight,   // [input_dim, num_classes]
       float* output,         // [batch, num_classes]
       int batch, int input_dim, int num_classes
   ) {
       // Your code here!
       // Compute logits = input @ weight, then apply softmax
   }
   ```
   
   **Challenge**: Use shared memory for the weight matrix.

---

## Exercise 7: Real-World Integration (Advanced)

### Goal
Integrate into a simple neural network.

### Tasks

1. **Simple 2-Layer Network**
   
   Implement a tiny MLP:
   ```cuda
   // Forward pass
   matmul_kernel(input, W1, hidden);     // [B, 784] x [784, 128]
   relu_kernel(hidden);                   // ReLU activation
   matmul_kernel(hidden, W2, logits);    // [B, 128] x [128, 10]
   softmax_kernel(logits, output);       // Your optimized softmax!
   ```
   
   Benchmark on MNIST (or random data).

2. **Add Batch Normalization**
   
   Before softmax, add batch norm:
   ```cuda
   __global__ void batch_norm_softmax_fused(
       const float* input,
       float* output,
       const float* gamma,
       const float* beta,
       int batch, int features
   ) {
       // Compute batch statistics (mean, var)
       // Normalize: (x - mean) / sqrt(var + eps)
       // Scale/shift: gamma * x_norm + beta
       // Apply softmax
   }
   ```

---

## Exercise 8: Profiling & Optimization (Advanced)

### Goal
Become proficient with CUDA profiling tools.

### Tasks

1. **Nsight Systems Analysis**
   
   ```bash
   nsys profile -o softmax_profile ./softmax_benchmark 1024 1000 100
   nsys-ui softmax_profile.qdrep
   ```
   
   **Find**:
   - Kernel execution time
   - Memory transfer time
   - CPU overhead
   - GPU utilization gaps
   
   **Question**: What's the bottleneck?

2. **Nsight Compute Deep Dive**
   
   ```bash
   ncu --set full -o softmax_ncu ./softmax_benchmark 1024 1000 1
   ncu-ui softmax_ncu.ncu-rep
   ```
   
   **Analyze**:
   - Memory throughput (GB/s)
   - Achieved occupancy (%)
   - Warp execution efficiency
   - Register usage per thread
   
   **Task**: Find one bottleneck and fix it!

3. **Roofline Model**
   
   Calculate arithmetic intensity:
   ```
   AI = FLOPs / Bytes
   
   For softmax (per element):
   FLOPs = 3 exp + 2 sub + 1 div ≈ 6 ops
   Bytes = 2 reads + 1 write = 12 bytes (FP32)
   AI = 6/12 = 0.5 ops/byte
   ```
   
   **Question**: Is softmax compute-bound or memory-bound?

---

## Exercise 9: Different Data Types (Expert)

### Goal
Optimize for mixed precision training.

### Tasks

1. **Implement FP16 Version**
   
   Use half precision:
   ```cuda
   __global__ void softmax_fp16(
       const __half* input,
       __half* output,
       int rows, int cols
   ) {
       // Use __hmax, __hmul, etc.
       // Be careful with accumulation (use FP32 internally!)
   }
   ```
   
   **Benefit**: 2x memory bandwidth!

2. **Mixed Precision**
   
   Input/output FP16, compute FP32:
   ```cuda
   // Read as FP16, convert to FP32
   float val = __half2float(input[i]);
   // Compute in FP32
   // Convert back and write
   output[i] = __float2half(result);
   ```

---

## Exercise 10: Scaling to Multiple GPUs (Expert)

### Goal
Learn multi-GPU programming.

### Tasks

1. **Naive Multi-GPU**
   
   Split batch across GPUs:
   ```cuda
   for (int gpu = 0; gpu < num_gpus; gpu++) {
       cudaSetDevice(gpu);
       int rows_per_gpu = total_rows / num_gpus;
       softmax<<<...>>>(d_input[gpu], d_output[gpu], ...);
   }
   ```

2. **NCCL for Collectives**
   
   If you need to gather results:
   ```cpp
   ncclAllGather(send_buffer, recv_buffer, ...);
   ```

---

## 🎓 Learning Checklist

After completing these exercises, you should be able to:

- [ ] Explain thread hierarchy (grid, block, warp, thread)
- [ ] Write coalesced memory access patterns
- [ ] Implement parallel reduction
- [ ] Use warp shuffle primitives
- [ ] Understand online algorithms
- [ ] Fuse multiple kernels
- [ ] Profile with Nsight tools
- [ ] Optimize based on profiling data
- [ ] Use shared memory effectively
- [ ] Avoid warp divergence
- [ ] Calculate arithmetic intensity
- [ ] Work with different data types
- [ ] Integrate CUDA into real applications

---

## 📝 Challenge Projects

Ready for more? Try these:

1. **FlashAttention Lite**: Implement simplified version
2. **Layer Normalization**: Similar to softmax, great practice
3. **Custom Activation Function**: Create fused kernel
4. **Sparse Softmax**: Handle sparse attention masks
5. **Gradient Computation**: Implement backward pass

---

## 🎉 Completion

When you finish:
1. Document your learnings
2. Share your optimizations
3. Help others learn!

**Remember**: The goal isn't just to make it fast, but to **understand why** it's fast.

Happy hacking! 🚀
