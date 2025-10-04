# Extension Guide: Integrating with Real Neural Networks

This guide helps you extend the softmax optimization tutorial to work with real datasets and neural networks.

## 🎯 Part 1: Tiny-ImageNet Dataset Setup

### Download and Prepare Dataset

```bash
# Create data directory
mkdir -p data
cd data

# Download Tiny-ImageNet (takes a few minutes)
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

# Extract
unzip tiny-imagenet-200.zip

# Dataset structure:
# tiny-imagenet-200/
#   ├── train/          # 200 classes, 500 images each
#   ├── val/            # 10,000 validation images
#   └── test/           # 10,000 test images
```

### Dataset Statistics
- **Classes**: 200
- **Training images**: 100,000 (500 per class)
- **Validation images**: 10,000
- **Image size**: 64×64×3 (RGB)
- **Total size**: ~250MB

## 🧠 Part 2: CNN Architecture for Tiny-ImageNet

### Recommended Architecture

```
Input (64×64×3)
    ↓
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓ (32×32×32)
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓ (16×16×64)
Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓ (8×8×128)
Flatten → (8192)
    ↓
Dense(512) → ReLU → Dropout(0.5)
    ↓
Dense(200) → Softmax ← YOUR OPTIMIZED KERNEL!
    ↓
Output (200 classes)
```

### Expected Performance
- **Baseline accuracy**: 35-45% top-1
- **Well-tuned**: 50-60% top-1
- **State-of-the-art**: 70%+ top-1

## 🔧 Part 3: Implementation Options

### Option A: PyTorch Integration (Recommended for Beginners)

**Pros**: Easy to set up, well-tested layers
**Cons**: Overhead from PyTorch-CUDA interface

```python
# File: train_pytorch.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ctypes

# Load your custom CUDA softmax
custom_softmax = ctypes.CDLL('./libsoftmax.so')

class CustomSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Call your CUDA kernel
        output = torch.empty_like(input)
        # ... call custom kernel via ctypes or PyBind11
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass
        return grad_output

class TinyImageNetCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        # Use custom softmax here
        return CustomSoftmax.apply(logits)

# Training loop
def train():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(
        'data/tiny-imagenet-200/train', 
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, 
                             shuffle=True, num_workers=4)
    
    model = TinyImageNetCNN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train()
```

### Option B: Pure CUDA Implementation (Advanced)

**Pros**: Full control, maximum performance
**Cons**: More code to write, harder to debug

You'll need to implement:
1. ✅ Softmax (done!)
2. ⬜ Convolution kernels
3. ⬜ Pooling kernels
4. ⬜ Batch normalization
5. ⬜ ReLU activation
6. ⬜ Fully connected layers
7. ⬜ Cross-entropy loss
8. ⬜ Backward pass (gradients)
9. ⬜ Optimizer (SGD/Adam)
10. ⬜ Data loading

**Recommended**: Use cuDNN for conv/pooling, focus on optimizing softmax!

### Option C: cuDNN with Custom Softmax (Balanced)

**Pros**: Best of both worlds
**Cons**: Requires cuDNN knowledge

```cuda
// File: cnn_cudnn.cu
#include <cudnn.h>
#include "your_softmax.cuh"

class CNNWithCustomSoftmax {
private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    // ... more descriptors
    
public:
    void forward(float* input, float* output, int batch_size) {
        // 1. Conv layers with cuDNN
        cudnnConvolutionForward(cudnn, ...);
        
        // 2. Pooling with cuDNN
        cudnnPoolingForward(cudnn, ...);
        
        // 3. Final FC layer (can use cuBLAS)
        cublasSgemm(...);
        
        // 4. YOUR CUSTOM SOFTMAX!
        softmax_online<<<batch_size, 256>>>(logits, output, batch_size, 200);
    }
};
```

## 📊 Part 4: Benchmarking in Real Networks

### Metrics to Track

1. **Forward Pass Time**
   - Total network inference time
   - Softmax contribution (should be 1-5%)

2. **Training Throughput**
   - Images per second
   - Impact of softmax optimization

3. **Memory Usage**
   - Peak GPU memory
   - Bandwidth utilization

### Profiling Strategy

```bash
# Profile entire network
nsys profile --stats=true python train_pytorch.py

# Focus on softmax layer
ncu --set full --kernel-name softmax_online python train_pytorch.py

# Compare with PyTorch's softmax
ncu --set full --kernel-name ".*softmax.*" python train_pytorch.py
```

### Expected Results

For batch_size=128, num_classes=200:

| Implementation | Time (μs) | Speedup |
|----------------|-----------|---------|
| PyTorch CPU | ~2000 | 1.0x |
| PyTorch CUDA | ~150 | 13.3x |
| Your Naive | ~80 | 25.0x |
| Your Optimized | ~35 | 57.1x |

**Note**: In full network, softmax is <5% of total time!

## 🚀 Part 5: Kernel Fusion Opportunities

### Fusion 1: Softmax + CrossEntropy

Already implemented in `cnn_integration.cuh`!

**Benefits**:
- Eliminates intermediate probability storage
- Reduces memory bandwidth by 50%
- Faster for training

### Fusion 2: BatchNorm + ReLU + Softmax

For architectures with BN before softmax:

```cuda
__global__ void batch_norm_relu_softmax_fused(
    const float* logits,
    const float* bn_weight,
    const float* bn_bias,
    float* output,
    int batch_size,
    int num_classes
) {
    // 1. Apply batch norm
    // 2. Apply ReLU
    // 3. Compute softmax
    // All in one kernel!
}
```

### Fusion 3: Dropout + Softmax

For inference with dropout:

```cuda
__global__ void softmax_dropout_fused(
    const float* input,
    float* output,
    const float* dropout_mask,
    float dropout_prob,
    int batch_size,
    int num_classes
) {
    // Compute softmax and apply dropout in single pass
}
```

## 🎓 Part 6: Transformer Extension

### Attention Softmax Characteristics

| Aspect | CNN Softmax | Attention Softmax |
|--------|-------------|-------------------|
| Input shape | [batch, classes] | [batch, heads, seq, seq] |
| Typical size | 128×200 | 8×12×512×512 |
| Memory bound | Moderate | Critical |
| Optimization priority | Medium | High |

### Why Attention Softmax is Harder

1. **Much larger**: 512×512 vs 200
2. **More frequent**: Every layer vs final layer only
3. **Memory intensive**: Quadratic in sequence length
4. **IO bound**: Cannot fit in SRAM

### FlashAttention Strategy

```
Standard Attention:
1. Compute QK^T: O(N²d) memory
2. Softmax: O(N²) memory  ← BOTTLENECK
3. Multiply by V: O(N²d) memory
Total: 3 passes over O(N²) data

FlashAttention:
1. Tile Q, K, V to fit in SRAM
2. Compute attention in blocks
3. Use online softmax (your kernel!)
4. Recompute in backward pass
Total: ~1 pass with much less IO
```

## 📈 Part 7: Performance Targets

### Realistic Goals

| Task | Target | Notes |
|------|--------|-------|
| Softmax speedup | 5-10x | vs naive implementation |
| CNN training | 500-1000 img/s | On V100 |
| Tiny-ImageNet accuracy | 50-60% | With basic CNN |
| Profiler efficiency | >80% | Memory bandwidth |

### When to Optimize Further

✅ Optimize if softmax is >10% of total time
✅ Optimize for very large num_classes (>10K)
✅ Optimize for transformers (attention is bottleneck)
❌ Don't over-optimize if <1% of total time

## 🔬 Part 8: Debugging Tips

### Common Issues

**Issue**: Results don't match PyTorch
```bash
# Compare outputs element-wise
python compare_outputs.py --tolerance 1e-5
```

**Issue**: Slower than expected
```bash
# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active
```

**Issue**: Numerical instability
```bash
# Verify max subtraction is working
# Check for NaN/Inf in output
```

## 📚 Part 9: Additional Resources

### Datasets Beyond Tiny-ImageNet

1. **CIFAR-100**: 100 classes, 32×32 images (smaller, faster)
2. **ImageNet-1K**: 1000 classes, 224×224 (standard benchmark)
3. **Places-365**: 365 scene categories (scene classification)

### Frameworks to Learn From

1. **PyTorch**: `torch.nn.functional.softmax`
2. **cuDNN**: `cudnnSoftmaxForward`
3. **FlashAttention**: https://github.com/Dao-AILab/flash-attention
4. **xFormers**: https://github.com/facebookresearch/xformers

### Papers to Read

1. **Online softmax**: https://arxiv.org/pdf/1805.02867
2. **FlashAttention**: https://arxiv.org/abs/2205.14135
3. **FlashAttention-2**: https://arxiv.org/abs/2307.08691

## ✅ Next Steps Checklist

- [ ] Download Tiny-ImageNet dataset
- [ ] Set up PyTorch environment
- [ ] Implement PyBind11 wrapper for your kernels
- [ ] Create simple CNN architecture
- [ ] Train baseline model with PyTorch softmax
- [ ] Integrate your optimized softmax
- [ ] Benchmark and compare
- [ ] Profile with Nsight tools
- [ ] Try kernel fusion
- [ ] Extend to transformer attention

Good luck with your CUDA journey! 🚀
