/**
 * @file 01-idxing.cu
 * @author Hitesh Vaidya
 * @brief This file demonstrates CUDA thread and block indexing concepts
 *        It contains two kernel functions:
 *        1. hello_from_gpu() - A simple kernel that prints "Hello World" from GPU
 *        2. whoami() - A kernel that shows how thread and block indices work in CUDA
 *                      by printing detailed information about each thread's position
 *                      in the block and grid hierarchy
 * 
 * The main function launches these kernels with a 3D grid configuration:
 * - Grid dimensions: 2x3x4 blocks
 * - Block dimensions: 4x4x4 threads
 * This results in 24 blocks with 64 threads each, totaling 1536 threads.
 */

// PS: since I am using A6000, I need to compile with: nvcc -arch=sm_86 -o 01-idxing 01-idxing.cu
// To put in the correct sm_<number> for your GPU, you can identify the compute capability of your GPU using:
// deviceQuery program in - ../cuda_samples/1_Utilities/deviceQuery

#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello, World! from GPU!\n");
}

__global__ void whoami(void) {
       
    // block_id calculates the block's ID by summing the block's X, Y, and Z indices
    // which represent the block's position in the grid
    int block_id = 
        blockIdx.x +  // apartment number of this floor (points across)
        blockIdx.y * gridDim.x + // floor number in this building (rows high)
        blockIdx.z * gridDim.x * gridDim.y; // building number in the city (panes deep)

    // block_offset calculates the total number of threads that come before this block
    // by multiplying the block's ID by the total number of threads per block
    int block_offset = 
        block_id * //times our apartment number
        blockDim.x * blockDim.y * blockDim.z;  // total threads per block (people per apartment )

    // thread_offset calculates the thread's ID by summing the thread's X, Y, and Z indices
    // which represent the thread's position in the block
    int thread_offset = 
        threadIdx.x + // apartment number of this floor (points across)
        threadIdx.y * blockDim.x + // apartment number of this floor (points down)
        threadIdx.z * blockDim.x * blockDim.y; // total threads per block (people per apartment ) 
    
    // id is the total number of threads that come before this thread
    int thread_id = block_offset + thread_offset;
    
    printf("global thread_id: %04d | Block(%d,%d,%d) = %3d (local) | Thread(%d,%d,%d) = %3d (local)\n", 
            thread_id, 
            blockIdx.x, blockIdx.y, blockIdx.z, block_id,
            threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv) {
    const int b_x = 2, b_y = 2, b_z = 2;
    const int t_x = 4, t_y = 4, t_z = 4;
    // max warp size is 32 threads, so we will get 2 warps of 32 threads each

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;
    
    printf("Blocks per grid: %d\n", blocks_per_grid);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Total threads: %d\n", blocks_per_grid * threads_per_block);
    
    dim3 blocksPerGrid(b_x, b_y, b_z);  // 3D cude of shape 2*3*4 = 24
    dim3 threadsPerBlock(t_x, t_y, t_z);  // 3D cude of shape 4*4*4 = 64

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}