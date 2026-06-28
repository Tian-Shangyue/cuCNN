#pragma once
#include "../tensor/tensor.h"
#include "../layer/layer.h"
#include <vector>
#include <cuda_runtime.h>

/**
 * SGD with Momentum
 *
 * 更新规则：
 *   v = momentum * v + learning_rate * grad
 *   w = w - v
 */
class SGD {
public:
    float learning_rate_;   // 学习率
    float momentum_;    // 动量系数
    float weight_decay_;  // L2 正则化系数


    /// @param lr            学习率
    /// @param momentum      动量系数 [0, 1)，典型值 0.9
    /// @param weight_decay  L2 正则化系数，典型值 5e-4
    SGD(float lr = 0.01f, float momentum = 0.9f, float weight_decay = 0.0f);

    /// 注册可训练参数及其对应梯度（按相同顺序一一配对）
    /// @param params  parameters() 的返回值
    /// @param grads   gradients()  的返回值
    /// @pre  params.size() == grads.size()
    /// @pre  对于每个 i，params[i] 和 grads[i] 的 shape 必须一致
    void register_parameters(const std::vector<Tensor*>& params,
                             const std::vector<Tensor*>& grads);

    /// 执行一步参数更新（step 内部会清零梯度，无需再单独调用 zero_grad）
    void step();

    /// 手动清零所有梯度（用于梯度累积场景，正常训练通常不需要单独调用）
    void zero_grad();

    // 公有 getter，供测试访问动量
    const std::vector<Tensor>& get_velocities() const { return velocities_; }

private:
    std::vector<Tensor*> params_;      // 参数指针，size = N
    std::vector<Tensor*> grads_;       // 梯度指针，size = N，grads_[i] 对应 params_[i] 的梯度
    std::vector<Tensor> velocities_;   // 动量累积张量，size = N，每个与对应参数同形状
};