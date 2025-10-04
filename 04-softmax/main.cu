/**
 * Softmax Optimization Tutorial - Main Benchmark Harness
 *
 * This file demonstrates progressive optimization of the softmax operation in CUDA.
 * We implement and benchmark multiple versions:
 * 1. Naive version (sequential operations)
 * 2. Coalesced memory access version
 * 3. Block-wise reduction version
 * 4. Online softmax (single-pass algorithm)
 *
 * Author: CUDA Tutorials
 * Date: October 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

// Error checking macro
#define CUDA_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess)                                          \
        {                                                                  \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Timing macro
#define TIME_KERNEL(kernel_call, time_ms)                        \
    do                                                           \
    {                                                            \
        cudaEvent_t start, stop;                                 \
        CUDA_CHECK(cudaEventCreate(&start));                     \
        CUDA_CHECK(cudaEventCreate(&stop));                      \
        CUDA_CHECK(cudaEventRecord(start));                      \
        kernel_call;                                             \
        CUDA_CHECK(cudaEventRecord(stop));                       \
        CUDA_CHECK(cudaEventSynchronize(stop));                  \
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop)); \
        CUDA_CHECK(cudaEventDestroy(start));                     \
        CUDA_CHECK(cudaEventDestroy(stop));                      \
    } while (0)

/**
 * KERNEL 1: Naive Softmax
 *
 * Each thread processes one row independently.
 * Steps:
 * 1. Find max value in row (for numerical stability)
 * 2. Compute exp(x - max) and sum
 * 3. Normalize by dividing by sum
 *
 * Issues:
 * - Inefficient: 3 sequential passes over data
 * - No parallelism within a row
 * - Poor memory access patterns
 */
__global__ void softmax_naive(const float *input, float *output, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        const float *row_data = input + row * cols;
        float *row_out = output + row * cols;

        // Step 1: Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < cols; i++)
        {
            max_val = fmaxf(max_val, row_data[i]);
        }

        // Step 2: Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < cols; i++)
        {
            sum += expf(row_data[i] - max_val);
        }

        // Step 3: Normalize
        for (int i = 0; i < cols; i++)
        {
            row_out[i] = expf(row_data[i] - max_val) / sum;
        }
    }
}

/**
 * KERNEL 2: Softmax with Coalesced Memory Access
 *
 * Improvements:
 * - Threads within a block process the same row
 * - Coalesced writes to global memory
 * - Still uses sequential reduction
 *
 * Each thread computes a portion of the row, then we use
 * shared memory for reductions.
 */
__global__ void softmax_coalesced(const float *input, float *output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float *row_data = input + row * cols;
    float *row_out = output + row * cols;

    // Shared memory for reduction
    extern __shared__ float shared[];
    float *max_vals = shared;
    float *sum_vals = shared + blockDim.x;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Step 1: Find max (parallel reduction)
    float thread_max = -FLT_MAX;
    for (int i = tid; i < cols; i += stride)
    {
        thread_max = fmaxf(thread_max, row_data[i]);
    }
    max_vals[tid] = thread_max;
    __syncthreads();

    // Reduce max values
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }
    float max_val = max_vals[0];
    __syncthreads();

    // Step 2: Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        thread_sum += expf(row_data[i] - max_val);
    }
    sum_vals[tid] = thread_sum;
    __syncthreads();

    // Reduce sum values
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sum_vals[tid] += sum_vals[tid + s];
        }
        __syncthreads();
    }
    float sum = sum_vals[0];
    __syncthreads();

    // Step 3: Normalize (coalesced writes!)
    for (int i = tid; i < cols; i += stride)
    {
        row_out[i] = expf(row_data[i] - max_val) / sum;
    }
}

/**
 * KERNEL 3: Optimized Block Reduction
 *
 * Improvements:
 * - Uses warp-level primitives for faster reduction
 * - Better shared memory utilization
 * - Optimized for modern GPU architectures
 */
__inline__ __device__ float warp_reduce_max(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_optimized(const float *input, float *output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float *row_data = input + row * cols;
    float *row_out = output + row * cols;

    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = (blockDim.x + 31) / 32;

    // Step 1: Find max with warp reduction
    float thread_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
    {
        thread_max = fmaxf(thread_max, row_data[i]);
    }

    float warp_max = warp_reduce_max(thread_max);

    // First thread in each warp writes to shared memory
    if (lane == 0)
    {
        shared[warp_id] = warp_max;
    }
    __syncthreads();

    // Final reduction across warps
    float max_val = -FLT_MAX;
    if (tid < num_warps)
    {
        max_val = shared[tid];
    }
    max_val = warp_reduce_max(max_val);
    if (tid == 0)
    {
        shared[0] = max_val;
    }
    __syncthreads();
    max_val = shared[0];

    // Step 2: Compute exp and sum with warp reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
    {
        thread_sum += expf(row_data[i] - max_val);
    }

    float warp_sum = warp_reduce_sum(thread_sum);

    if (lane == 0)
    {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction across warps
    float sum = 0.0f;
    if (tid < num_warps)
    {
        sum = shared[tid];
    }
    sum = warp_reduce_sum(sum);
    if (tid == 0)
    {
        shared[0] = sum;
    }
    __syncthreads();
    sum = shared[0];

    // Step 3: Normalize
    for (int i = tid; i < cols; i += blockDim.x)
    {
        row_out[i] = expf(row_data[i] - max_val) / sum;
    }
}

/**
 * KERNEL 4: Online Softmax (Single-Pass Algorithm)
 *
 * Based on the paper "Online normalizer calculation for softmax"
 * Reference: https://arxiv.org/pdf/1805.02867
 *
 * Key Innovation:
 * - Computes max and sum in a single pass
 * - Uses numerical stability trick: dynamically updates max and rescales sum
 * - More cache-friendly and reduces memory bandwidth
 *
 * Algorithm:
 * For each element x_i:
 *   old_max = max
 *   max = max(max, x_i)
 *   sum = sum * exp(old_max - max) + exp(x_i - max)
 */
__global__ void softmax_online(const float *input, float *output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float *row_data = input + row * cols;
    float *row_out = output + row * cols;

    extern __shared__ float shared[];
    float *max_vals = shared;
    float *sum_vals = shared + blockDim.x;

    int tid = threadIdx.x;

    // Online algorithm: compute max and sum in single pass
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x)
    {
        float val = row_data[i];
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, val);
        // Rescale sum when max changes
        thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
    }

    max_vals[tid] = thread_max;
    sum_vals[tid] = thread_sum;
    __syncthreads();

    // Reduce max and sum together (handling rescaling)
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            float old_max = max_vals[tid];
            float other_max = max_vals[tid + s];
            float new_max = fmaxf(old_max, other_max);

            // Rescale both sums to new max
            float rescaled_sum1 = sum_vals[tid] * expf(old_max - new_max);
            float rescaled_sum2 = sum_vals[tid + s] * expf(other_max - new_max);

            max_vals[tid] = new_max;
            sum_vals[tid] = rescaled_sum1 + rescaled_sum2;
        }
        __syncthreads();
    }

    float max_val = max_vals[0];
    float sum = sum_vals[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < cols; i += blockDim.x)
    {
        row_out[i] = expf(row_data[i] - max_val) / sum;
    }
}

// CPU reference implementation for verification
void softmax_cpu(const float *input, float *output, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
    {
        const float *row_in = input + r * cols;
        float *row_out = output + r * cols;

        // Find max
        float max_val = row_in[0];
        for (int c = 1; c < cols; c++)
        {
            max_val = fmaxf(max_val, row_in[c]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int c = 0; c < cols; c++)
        {
            sum += expf(row_in[c] - max_val);
        }

        // Normalize
        for (int c = 0; c < cols; c++)
        {
            row_out[c] = expf(row_in[c] - max_val) / sum;
        }
    }
}

// Verification function
bool verify_results(const float *cpu_output, const float *gpu_output, int rows, int cols, float epsilon = 1e-5)
{
    for (int i = 0; i < rows * cols; i++)
    {
        float diff = fabsf(cpu_output[i] - gpu_output[i]);
        if (diff > epsilon)
        {
            printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f, diff=%.6f\n",
                   i, cpu_output[i], gpu_output[i], diff);
            return false;
        }
    }
    return true;
}

// Benchmark runner
void run_benchmark(const char *name,
                   void (*kernel)(const float *, float *, int, int),
                   const float *d_input, float *d_output,
                   int rows, int cols,
                   int block_size, int num_blocks,
                   size_t shared_mem_size,
                   const float *cpu_output,
                   int num_iterations = 100)
{

    float total_time = 0.0f;

    // Warmup
    kernel<<<num_blocks, block_size, shared_mem_size>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    for (int i = 0; i < num_iterations; i++)
    {
        float iter_time;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<num_blocks, block_size, shared_mem_size>>>(d_input, d_output, rows, cols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&iter_time, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        total_time += iter_time;
    }

    float avg_time = total_time / num_iterations;

    // Verify correctness
    float *h_output = (float *)malloc(rows * cols * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    bool correct = verify_results(cpu_output, h_output, rows, cols);

    printf("%-25s: %.4f ms (avg over %d runs) - %s\n",
           name, avg_time, num_iterations, correct ? "PASSED" : "FAILED");

    free(h_output);
}

int main(int argc, char **argv)
{
    // Default problem size
    int rows = 1024; // Number of samples
    int cols = 1000; // Feature dimension (e.g., number of classes)
    int num_iterations = 100;

    // Parse command line arguments
    if (argc > 1)
        rows = atoi(argv[1]);
    if (argc > 2)
        cols = atoi(argv[2]);
    if (argc > 3)
        num_iterations = atoi(argv[3]);

    printf("=== Softmax Optimization Benchmark ===\n");
    printf("Problem size: %d x %d\n", rows, cols);
    printf("Iterations: %d\n\n", num_iterations);

    // Allocate host memory
    size_t size = rows * cols * sizeof(float);
    float *h_input = (float *)malloc(size);
    float *h_output_cpu = (float *)malloc(size);

    // Initialize input with random data
    srand(42);
    for (int i = 0; i < rows * cols; i++)
    {
        h_input[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f; // Range [-5, 5]
    }

    // Compute CPU reference
    printf("Computing CPU reference...\n");
    softmax_cpu(h_input, h_output_cpu, rows, cols);
    printf("CPU reference computed.\n\n");

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Benchmark parameters
    int block_size = 256;
    int num_blocks;
    size_t shared_mem_size;

    printf("Running benchmarks...\n\n");

    // 1. Naive version
    num_blocks = (rows + block_size - 1) / block_size;
    run_benchmark("Naive Softmax", softmax_naive, d_input, d_output,
                  rows, cols, block_size, num_blocks, 0, h_output_cpu, num_iterations);

    // 2. Coalesced memory access
    num_blocks = rows; // One block per row
    shared_mem_size = 2 * block_size * sizeof(float);
    run_benchmark("Coalesced Memory", softmax_coalesced, d_input, d_output,
                  rows, cols, block_size, num_blocks, shared_mem_size, h_output_cpu, num_iterations);

    // 3. Optimized with warp primitives
    num_blocks = rows;
    shared_mem_size = 32 * sizeof(float); // For warp results
    run_benchmark("Optimized Block Reduce", softmax_optimized, d_input, d_output,
                  rows, cols, block_size, num_blocks, shared_mem_size, h_output_cpu, num_iterations);

    // 4. Online softmax
    num_blocks = rows;
    shared_mem_size = 2 * block_size * sizeof(float);
    run_benchmark("Online Softmax", softmax_online, d_input, d_output,
                  rows, cols, block_size, num_blocks, shared_mem_size, h_output_cpu, num_iterations);

    printf("\n=== Benchmark Complete ===\n");

    // Cleanup
    free(h_input);
    free(h_output_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
