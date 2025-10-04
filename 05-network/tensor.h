#include <vector>
#include <cstddef>

struct Tensor
{
    std::vector<float> data;
    size_t batch;
    size_t channels;
    size_t height;
    size_t width;

    // This is the constructor 👇
    Tensor(size_t b, size_t c, size_t h, size_t w)
    {
        batch = b;
        channels = c;
        height = h;
        width = w;
        // Calculate total size for the flattened vector
        size_t size = b * c * h * w;
        data.resize(size); // Allocate the memory
    }
};
