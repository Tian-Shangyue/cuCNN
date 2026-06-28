#pragma once
#include "../layer/layer.h"

/**
 * 全连接层（矩阵乘法） Y = X @ W + b
 *
 * 输入可以是任意维张量，前向传播时会自动展平为 2D：
 *   [batch, D1, D2, ...]  →  [batch, in_features]
 *
 * 权重布局：W [in_features, out_features]
 * 偏置布局：b [out_features]
 * 输出布局：[batch, out_features]
 */
class Linear : public Layer
{
public:
    int in_features;    //K
    int out_features;   //M
    Tensor weight;  // shape: [in_features, out_features] K*M
    Tensor bias;    // shape: [out_features] M

    // 梯度存储, 用于更新权重
    Tensor grad_weight;  // [in_features, out_features]
    Tensor grad_bias;    // [out_features]

    // 前向传播中间存储
    Tensor input_cache_; // [N, in_features] — 反向计算 grad_weight 时需要
    std::vector<int> input_shape_; // 保存原始输入形状

    Linear(int in_f, int out_f);

    void init_weights(float w_val = 0.01f, float b_val = 0.0f);
    void load_weights(const float* w_host, const float* b_host);

    // 随机初始化（训练用）
    void init_weights_random(unsigned int seed = 42);


    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::vector<Tensor*> parameters() override {
        return {&weight, &bias};
    }
    std::vector<Tensor*> gradients() override {
        return {&grad_weight, &grad_bias};
    }
};
