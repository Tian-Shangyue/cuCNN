#pragma once
#include "../tensor/tensor.h"
#include <cstdio>

/**
 * CrossEntropyLoss = Softmax + NLLLoss 联合实现
 * 训练前向：先做 softmax（数值稳定版），再算交叉熵损失
 * 训练反向：直接返回 y_pred - y_true（联合梯度，避免 O(D²)）
 */
class CrossEntropyLoss {
public:
    float loss_;                    // 当前 batch 的平均损失
    Tensor grad_output_;            // 存储反向传播的初始梯度, 平均损失每个位置的梯度也同样要除batchsize

    float forward(const Tensor& input,    // [N, num_classes] — 未经 softmax 的原始输出
                  const Tensor& targets);  // [N] — 整数类别标签 (0 ~ num_classes-1)

    const Tensor& backward();              // 返回 dL/d(logits) = softmax(logits) - onehot(targets)
};