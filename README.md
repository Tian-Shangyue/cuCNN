# cuCNN — A CNN Training Framework in CUDA C++ from Scratch

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-14-blue?logo=c%2B%2B)](https://en.cppreference.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A minimal CNN framework written entirely in CUDA C++, built from the ground up as a **learning project**. No cuDNN, no cuBLAS — just raw CUDA kernels, hand-optimized with shared memory tiling, template metaprogramming, and grid-stride loops.


> **⚠️ This is a simple learning project with many limitations.** It only supports 2D convolution for image tasks — no BatchNorm, no Dropout, no residual connections, no RNN/Transformer. The backward kernels use naive implementations without deep optimization. Currently only tested on MNIST and FashionMNIST. It is not comparable to production frameworks like PyTorch or TensorFlow. If you're also learning CUDA and CNN internals, hope this project can be a useful reference.

---

## Features

### Forward Propagation (Inference)

| Layer | Implementation | Highlights |
|---|---|---|
| **Conv2D** | 3-level fallback | Static tiled (template) → dynamic tiled (shared mem) → naive |
| **Linear** | Tiled GEMM | Shared memory blocking over K dimension |
| **ReLU** | Grid-stride | Single kernel handles arbitrary sizes |
| **MaxPool2D** | Per-thread window | Stores argmax indices for backward |
| **AvgPool2D** | Per-thread window | Uniform gradient redistribution |
| **Softmax** | Numerically stable | Block reduction for max & sum |

### Backward Propagation (Training)

Every forward operator has a corresponding backward kernel:

| Layer | Backward Kernels |
|---|---|
| **Conv2D** | `grad_input` (transposed conv), `grad_weight` (cross-correlation), `grad_bias` |
| **Linear** | `grad_input` (W^T matmul), `grad_weight` (X^T matmul), `grad_bias` (column sum) |
| **ReLU** | Masked gradient pass-through |
| **MaxPool2D** | Gradient routing via stored argmax (`atomicAdd` for overlapping windows) |
| **AvgPool2D** | Uniform redistribution (`atomicAdd` for overlapping windows) |

### Training Pipeline

- **Loss:** `CrossEntropyLoss` — combined Softmax + NLL, O(D) joint gradient (avoids the O(D²) Softmax Jacobian)
- **Optimizer:** SGD with Momentum + Weight Decay (L2 regularization)
- **Trainer:** Epoch loop, per-batch loss tracking, evaluation with accuracy
- **DataLoader:** Pinned memory H→D transfer, Fisher-Yates shuffle, batch iteration

### Data Loading

- **MNIST** — direct IDX binary parsing in C++ (big-endian → little-endian, uint8 → float32)
- **FashionMNIST** — same IDX format, drop-in compatible

---

## Architecture

```
┌─────────────────────────────────────────┐
│               Trainer                    │
│  train_epoch() / evaluate() / train()   │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
  Sequential   Loss      Optimizer
  (forward +   (CE +     (SGD +
   backward)   Softmax)  Momentum)
       │                     │
  ┌────┴────┬────────┐       │
  │         │        │       │
Conv2D   Linear   Pool2D   ReLU
(3 strats)(tiled) (max/avg)(grid-stride)
  │         │        │       │
  └────┴────┴────────┴───────┘
              │
          Tensor
    (GPU memory, NCHW, move semantics)
              │
         DataLoader
    (shuffle, batch, pinned H→D)
              │
        MNISTDataset
    (IDX parser, big-endian)
```

### Memory Layout

All tensors use **NCHW** (batch, channels, height, width) layout. Weights use **KCRS** (output channels, input channels, kernel height, kernel width). This is consistent with PyTorch's default and optimal for CUDA coalesced memory access.

### Conv2D Optimization Strategy

```
┌──────────────┐    hit?    ┌──────────────────┐
│ Static Tiled │ ────────→  │ Template kernel   │  Compile-time R,S,stride
│ (try_launch) │            │ Multidim shm array│  Zero index overhead
└──────┬───────┘            └──────────────────┘
       │ miss                        shm ≤ 48KB?
       ▼                    ┌──────────────────┐
┌──────────────┐   yes     │ Dynamic Tiled     │  Runtime R,S,stride
│ shm ≤ 48KB?  │ ────────→ │ Flat 1D shm array │  Manual offset calc
└──────┬───────┘           └──────────────────┘
       │ no
       ▼
┌──────────────┐
│ Naive Conv   │              Ultimate fallback
│ Global mem   │              Always works
└──────────────┘
```

---

## Project Structure

```
cuCNN/
├── tensor/                    # GPU Tensor with move semantics
│   ├── tensor.h
│   └── tensor.cpp
├── layer/                     # Abstract Layer interface
│   └── layer.h                #   forward() + backward() + parameters()
├── sequential/                # Sequential container
│   └── sequential.h           #   Ordered forward, reversed backward
├── conv/                      # 2D Convolution
│   ├── conv.h
│   └── conv.cu                #   3-level fallback (static→dynamic→naive)
├── linear/                    # Fully Connected (Tiled GEMM)
│   ├── linear.h
│   └── linear.cu
├── relu/                      # ReLU Activation
│   ├── relu.h
│   └── relu.cu
├── pool/                      # MaxPool2D & AvgPool2D
│   ├── pool.h
│   └── pool.cu
├── softmax/                   # Softmax (block reduction)
│   ├── softmax.h
│   └── softmax.cu
├── loss/                      # CrossEntropyLoss
│   ├── loss.h
│   └── loss.cu
├── optimizer/                 # SGD with Momentum
│   ├── optimizer.h
│   └── optimizer.cu
├── dataloader/                # Data loading
│   ├── dataset.h              #   Abstract Dataset interface
│   ├── dataset.cpp
│   ├── mnist_dataset.h        #   MNIST/FashionMNIST IDX parser
│   ├── mnist_dataset.cpp
│   ├── dataloader.h           #   Shuffle + batch + pinned H→D
│   └── dataloader.cpp
├── train/                     # Training loop
│   ├── trainer.h
│   └── trainer.cpp
├── kernels/                   # All CUDA kernels
│   ├── kernels.h              #   Kernel declarations + helper macros
│   ├── conv_kernel.cu         #   forward: native/tiled/dynamic, backward: input/weight/bias
│   ├── linear_kernel.cu       #   forward: tiled matmul, backward: input/weight/bias
│   ├── pool_kernel.cu         #   forward: max/avg, backward: max/avg
│   ├── relu_kernel.cu         #   forward: grid-stride, backward: masked
│   ├── softmax_kernel.cu      #   forward: block reduction
│   ├── loss_kernel.cu         #   CrossEntropy: joint forward+gradient
│   └── optimizer_kernel.cu    #   SGD step with momentum
├── test/                      # Unit tests (CPU vs GPU correctness)
│   ├── test_conv2d_backward.cu
│   ├── test_linear_backward.cu
│   ├── test_pool_backward.cu
│   ├── test_relu.cu
│   ├── test_loss.cu
│   └── test_optimizer.cu
├── main.cu                    # Inference demo
├── TRAINING_FRAMEWORK_DESIGN.md  # Full design documentation
├── Makefile
└── README.md
```

---

## Getting Started

### Prerequisites

- **NVIDIA GPU** with Compute Capability ≥ 5.0
- **CUDA Toolkit** ≥ 11.0
- **GNU Make** (or compatible)
- **GPU Driver** supporting your CUDA version

### Build

```bash
# Clone
git clone https://github.com/Tian-Shangyue/cuCNN.git
cd cuCNN

# Build (inference demo + all tests)
make all

# Run inference demo (SimpleCNN on random input)
./cnn_demo
```

### Run Tests

Each test compares GPU kernel output against a CPU reference implementation:

```bash
# Build and run a specific test
make test_conv2d_backward && ./test_conv2d_backward
make test_linear_backward && ./test_linear_backward
make test_pool_backward && ./test_pool_backward
make test_relu && ./test_relu
make test_loss && ./test_loss
make test_optimizer && ./test_optimizer
```

### Train on MNIST

1. Download [MNIST](http://yann.lecun.com/exdb/mnist/) and extract to `data/MNIST/`:

```
data/MNIST/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

2. Build and run:

```bash
make all
./cnn_demo          # Quick inference test
```

---

## Design Principles

### 1. No External Dependencies (beyond CUDA Runtime)

Every operation — convolution, matrix multiply, pooling, softmax — is a hand-written CUDA kernel. No cuDNN, no cuBLAS. This is intentional: you can't understand what you don't implement.

### 2. Progressive Optimization

Kernels evolve from naive to optimized:
- **Conv2D:** naive (global memory) → dynamic tiled (shared memory, runtime shapes) → static tiled (shared memory, compile-time shapes)
- **Linear:** naive (per-element thread) → tiled (shared memory blocking)

### 3. Separation of Concerns

| Component | Responsibility |
|---|---|
| `Tensor` | GPU memory lifecycle, move semantics |
| `Layer` | `forward()` / `backward()` interface |
| `Sequential` | Layer composition, traversal order |
| `Loss` | Loss value + initial gradient |
| `Optimizer` | Parameter update rule |
| `Dataset` | Data format parsing |
| `DataLoader` | Batching, shuffling, GPU upload |
| `Trainer` | Epoch loop orchestration |

### 4. Training-Mode Awareness

Layers only cache intermediate values (`input_cache_`, `max_indices_`) when `is_training_ = true`. In inference mode, these allocations are skipped to save GPU memory.

---

## Performance Notes

This is an educational framework — it prioritizes clarity and correctness over raw speed. 
For production use, frameworks like PyTorch/TensorFlow with cuDNN backend are recommended.

---

## License

MIT — feel free to use, modify, and learn from this code.
