#pragma once
#include "../layer/layer.h"

/**
 * Softmax 层（无状态）
 *
 * 输入布局：[batch, class_num]
 * 输出布局：[batch, class_num]
 *
 * 使用数值稳定的 CUDA 实现：
 * - 先减去每行最大值防止 exp 溢出
 * - 块内二分归约求 max 和 sum
 */
class Softmax : public Layer
{
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};