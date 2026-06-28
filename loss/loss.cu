#include "loss.h"
#include "../kernels/kernels.h"
#include <cmath>
#include <stdexcept>

// 返回当前batch的平均损失
float CrossEntropyLoss::forward(const Tensor& input,    // [N, num_classes] — 未经 softmax 的原始输出
                  const Tensor& targets)  // [N] — 整数类别标签 (0 ~ num_classes-1)
{
    // 解析输入形状
    // 支持 2D [batch, class_num] 或 4D（自动展平）
    if (input.ndim() < 2) {
        throw std::runtime_error(
            "CrossEntropyLoss::forward: input must be at least 2D");
    }
    if (targets.ndim() > 1) {
        throw std::runtime_error(
            "CrossEntropyLoss::forward: targets must be 1D");
    }

    int batch_size = input.shape[0];
    int class_num = 1;
    for (size_t i = 1; i < input.shape.size(); ++i) {
        class_num *= input.shape[i];
    }

    if (class_num <= 0){
        throw std::runtime_error("CrossEntropyLoss::forward: class_num must be > 0");
    }
        
    if (targets.shape[0] != batch_size){
        throw std::runtime_error("CrossEntropyLoss::forward: targets batch size mismatch");
    }
        

    // 分配 grad_output（前向时创建，反向时直接返回）
    this->grad_output_ = Tensor({batch_size, class_num});  // 移动构造

    // 每个 block 处理一个样本，block 内线程协作完成归约
    // blockDim 必须是 2 的幂
    int threads = 128;
    // 确保 threads >= 能覆盖 class_num 的 2 的幂
    while (threads < class_num && threads < 1024) {
        threads *= 2;
    }

    int blocks = batch_size;
    size_t shared_mem = threads * sizeof(float);

    Tensor loss_tensor({batch_size});  // 每个样本的 loss

    cross_entropy_forward<<<blocks, threads, shared_mem>>>(
        input.data, targets.data,
        this->grad_output_.data,   // 直接写入梯度！前向中同时计算了 softmax - onehot
        loss_tensor.data,
        batch_size, class_num);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

     // 求平均损失
    std::vector<float> h_loss = loss_tensor.copy_to_host();
    float sum = 0.0f;
    for (float v : h_loss) sum += v;
    this->loss_ = sum / batch_size;
    return this->loss_;
}

const Tensor& CrossEntropyLoss::backward()             // 返回 dL/d(logits) = softmax(logits) - onehot(targets)
{
    return this->grad_output_;
}