#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000000
#define BLOCK_SIZE 256

// Example:
// A = [1, 2, 3, 4, 5]
// B = [6, 7, 8, 9, 10]
// C = A + B = [7, 9, 11, 13, 15]

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)(rand() % 100);
    }
}

// Function measure the time taken by the CPU and GPU vector addition
double get_time(){
    // timespec is a C struct that holds time with nanosecond precision
    // tv_sec: seconds since epoch
    // tv_nsec: nanoseconds since last second (0-999999999)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts); // Get current time with monotonic clock
    return ts.tv_sec + ts.tv_nsec * 1e-9; // Convert to seconds with nanosecond precision
}

int main(int argc, char **argv){
    float *a, *b, *c_cpu, *c_gpu; // host vectors
    float *d_a, *d_b, *d_c; // device vectors
    size_t size = N * sizeof(float);

    // Allocate memory for host vectors
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c_cpu = (float *)malloc(size);
    c_gpu = (float *)malloc(size);

    // Initialize vectors with random values
    init_vector(a, N);
    init_vector(b, N);

    // Allocate memory for device vectors
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy vectors to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // warm up
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++){
        vector_add_cpu(a, b, c_cpu, N);
        vector_add_gpu<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    double cpu_time = 0.0;
    for (int i = 0; i < 10; i++){
        double start_time = get_time();
        vector_add_cpu(a, b, c_cpu, N);
        double end_time = get_time();
        cpu_time += end_time - start_time;
    }
    cpu_time /= 10;

    // Benchmark GPU implementation
    double gpu_time = 0.0;
    for (int i = 0; i < 10; i++){
        double start_time = get_time();
        vector_add_gpu<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_time += end_time - start_time;
    }
    gpu_time /= 10;

    // Copy result back to host
    cudaMemcpy(c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++){
        if (fabs(c_cpu[i] - c_gpu[i]) > 1e-5){
            correct = false;
            break;
        }
    }
    
    // Print results
    printf("CPU time: %f ms\n", cpu_time * 1000);
    printf("GPU time: %f ms\n", gpu_time * 1000);
    printf("Result is %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}