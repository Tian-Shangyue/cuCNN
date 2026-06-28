#include "kernels.h"

/**
 * SGD with Momentum + weight decay 参数更新核函数
 * 每个线程跨步更新多个参数
 * 对每个参数元素（grid-stride 模式，支持任意大张量）：
 *   float g = grad[i];
 *   if (weight_decay > 0) g += weight_decay * param[i];
 *   velocity[i] = momentum * velocity[i] + lr * g;
 *   param[i] = param[i] - velocity[i];
 *   grad[i] = 0.0f;     // 更新后立即清零，为下个 batch 准备
 */
__global__ void sgd_step(
    float* param,          // 参数 W，将被原地更新
    float* grad,           // 梯度 dW，将被清零
    float* velocity,       // 动量累积 v，将被更新
    float lr,
    float momentum,
    float weight_decay,
    int total)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += blockDim.x * gridDim.x)
    {
        float g = grad[i];
        if (weight_decay > 0.0f) {
            g += weight_decay * param[i];
        }
        velocity[i] = momentum * velocity[i] + lr * g;
        param[i] = param[i] - velocity[i];
        grad[i] = 0.0f;   // 清零，避免下个 batch 的梯度累积到本次
    }
}