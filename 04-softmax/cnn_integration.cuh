/**
 * CNN Integration Example - Softmax Layer
 *
 * This file demonstrates how to integrate optimized softmax kernels
 * into a convolutional neural network for image classification.
 *
 * Network Architecture (Simple CNN for Tiny-ImageNet):
 * Input (64x64x3) -> Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Pool
 * -> FC1 -> ReLU -> FC2 (logits) -> Softmax -> Output (200 classes)
 */

#ifndef SOFTMAX_CNN_H
#define SOFTMAX_CNN_H

#include <cuda_runtime.h>

// Forward declarations of optimized softmax kernels
// (defined in main.cu - you can extract these to a separate .cuh file)

__global__ void softmax_online(const float *input, float *output, int rows, int cols);
__global__ void softmax_optimized(const float *input, float *output, int rows, int cols);

/**
 * Softmax layer forward pass for CNN
 *
 * @param logits: Input logits from final FC layer [batch_size, num_classes]
 * @param probabilities: Output probabilities [batch_size, num_classes]
 * @param batch_size: Number of samples in batch
 * @param num_classes: Number of output classes
 */
void cnn_softmax_forward(const float *logits, float *probabilities,
                         int batch_size, int num_classes)
{
    // Use online softmax for best performance
    int block_size = 256;
    int num_blocks = batch_size; // One block per sample
    size_t shared_mem = 2 * block_size * sizeof(float);

    softmax_online<<<num_blocks, block_size, shared_mem>>>(
        logits, probabilities, batch_size, num_classes);
}

/**
 * Combined Softmax + Cross-Entropy Loss (Fused Kernel)
 *
 * This kernel fuses softmax computation with cross-entropy loss calculation,
 * saving memory bandwidth and kernel launch overhead.
 *
 * Loss = -sum(target[i] * log(softmax(logits[i])))
 */
__global__ void softmax_cross_entropy_fused(
    const float *logits,  // [batch_size, num_classes]
    const int *targets,   // [batch_size] - class indices
    float *losses,        // [batch_size] - output losses
    float *probabilities, // [batch_size, num_classes] - optional output
    int batch_size,
    int num_classes,
    bool output_probs = false)
{
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size)
        return;

    const float *sample_logits = logits + sample_idx * num_classes;
    int target_class = targets[sample_idx];

    extern __shared__ float shared[];
    float *max_vals = shared;
    float *sum_vals = shared + blockDim.x;

    int tid = threadIdx.x;

    // Online algorithm: compute max and sum
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    for (int i = tid; i < num_classes; i += blockDim.x)
    {
        float val = sample_logits[i];
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, val);
        thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
    }

    max_vals[tid] = thread_max;
    sum_vals[tid] = thread_sum;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            float old_max = max_vals[tid];
            float other_max = max_vals[tid + s];
            float new_max = fmaxf(old_max, other_max);

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

    // Compute loss for target class (only need one thread)
    if (tid == 0)
    {
        float target_logit = sample_logits[target_class];
        float log_prob = (target_logit - max_val) - logf(sum);
        losses[sample_idx] = -log_prob;
    }

    // Optionally output full probability distribution
    if (output_probs && probabilities != nullptr)
    {
        float *sample_probs = probabilities + sample_idx * num_classes;
        for (int i = tid; i < num_classes; i += blockDim.x)
        {
            sample_probs[i] = expf(sample_logits[i] - max_val) / sum;
        }
    }
}

/**
 * Training step combining forward pass and loss computation
 */
void cnn_train_step(
    const float *logits,  // Output from FC layer
    const int *targets,   // Ground truth labels
    float *probabilities, // Softmax output
    float *losses,        // Per-sample losses
    float *total_loss,    // Average loss (host pointer)
    int batch_size,
    int num_classes)
{
    // Launch fused kernel
    int block_size = 256;
    size_t shared_mem = 2 * block_size * sizeof(float);

    softmax_cross_entropy_fused<<<batch_size, block_size, shared_mem>>>(
        logits, targets, losses, probabilities, batch_size, num_classes, true);

    // Compute average loss (simplified - use reduction kernel in practice)
    float *h_losses = new float[batch_size];
    cudaMemcpy(h_losses, losses, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < batch_size; i++)
    {
        sum += h_losses[i];
    }
    *total_loss = sum / batch_size;

    delete[] h_losses;
}

/**
 * Inference function (forward pass only, optimized)
 */
void cnn_inference(
    const float *logits,
    float *probabilities,
    int *predictions, // [batch_size] - predicted class indices
    int batch_size,
    int num_classes)
{
    // Compute softmax
    cnn_softmax_forward(logits, probabilities, batch_size, num_classes);

    // Find argmax (can be done on GPU with reduction, shown on CPU for simplicity)
    float *h_probs = new float[batch_size * num_classes];
    cudaMemcpy(h_probs, probabilities, batch_size * num_classes * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int b = 0; b < batch_size; b++)
    {
        float max_prob = h_probs[b * num_classes];
        int max_idx = 0;
        for (int c = 1; c < num_classes; c++)
        {
            if (h_probs[b * num_classes + c] > max_prob)
            {
                max_prob = h_probs[b * num_classes + c];
                max_idx = c;
            }
        }
        predictions[b] = max_idx;
    }

    delete[] h_probs;
}

#endif // SOFTMAX_CNN_H

/*
 * USAGE EXAMPLE:
 *
 * // In your training loop:
 * for (int epoch = 0; epoch < num_epochs; epoch++) {
 *     for (int batch = 0; batch < num_batches; batch++) {
 *         // 1. Forward pass through conv/fc layers (not shown)
 *         //    float* d_logits = fc_layer_forward(...);
 *
 *         // 2. Compute softmax + loss
 *         float avg_loss;
 *         cnn_train_step(d_logits, d_targets, d_probs, d_losses,
 *                        &avg_loss, batch_size, num_classes);
 *
 *         // 3. Backward pass (not shown)
 *         //    backward_pass(d_probs, d_targets, ...);
 *     }
 * }
 *
 * // For inference:
 * int* h_predictions = new int[batch_size];
 * cnn_inference(d_logits, d_probs, h_predictions, batch_size, num_classes);
 */
