#include "tensor.h"

struct Linear
{
    Tensor weights;
    Tensor biases;

    // Constructor
    Linear(size_t input_features, size_t output_features)
        : weights(1, 1, output_features, input_features), // Weight shape: (output_features, input_features)
          biases(1, 1, 1, output_features)                // Bias shape: (output_features,)
    {
        // Initialize weights and biases if necessary
    }
};