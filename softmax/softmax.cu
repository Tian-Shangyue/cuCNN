#include "softmax.h"
#include "../kernels/kernels.h"
#include <stdexcept>


Tensor Softmax::forward(const Tensor& input)
{
    // 解析输入形状
    // 支持 2D [batch, class_num] 或 4D（自动展平）
    if (input.ndim() < 2) {
        throw std::runtime_error(
            "Softmax::forward: input must be at least 2D");
    }

    int batch_size = input.shape[0];
    int class_num = 1;
    for (size_t i = 1; i < input.shape.size(); ++i) {
        class_num *= input.shape[i];
    }

    // 分配输出张量（与输入同形状）
    Tensor output(input.shape);

    // 启动 Softmax 核函数
    // 每个 block 处理一个样本，block 内线程协作完成归约
    // blockDim 必须是 2 的幂
    int threads = 128;
    // 确保 threads >= 能覆盖 class_num 的 2 的幂
    while (threads < class_num && threads < 1024) {
        threads *= 2;
    }

    int blocks = batch_size;
    size_t shared_mem = threads * sizeof(float);

    softmax_forward<<<blocks, threads, shared_mem>>>(
        input.data, output.data,
        batch_size, class_num);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return output;
}

Tensor Softmax::backward(const Tensor& grad_output) {
    return Tensor();
}