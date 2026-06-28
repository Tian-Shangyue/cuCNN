#pragma once
#include "../layer/layer.h"

/**
 * ReLU 激活函数层
 * 使用 grid-stride 优化的 CUDA 核函数：
 * 一个线程处理多个元素，减少线程调度开销
 */
class ReLU : public Layer
{
public:

    Tensor input_cache_;  // 缓存前向输入，用于反向判断 mask

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};
