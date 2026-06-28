#pragma once
#include "../tensor/tensor.h"

/**
神经网络层的抽象基类 Layer
Layer基类，所有层都继承它
 */
class Layer{
    public:
        //纯虚函数 制定统一的「前向传播」接口
        virtual Tensor forward(const Tensor& input)=0;

        virtual ~Layer(){}

        //反向传播
        // @param grad_output  来自上一层的梯度（∂L/∂Y）更靠近网络的最后一层，形状与本层网络 forward 的输出一致
        // @return grad_input  传递给下一层的梯度（∂L/∂X）更靠近网络的第一层，形状与本层网络 forward 的输入一致
        virtual Tensor backward(const Tensor& grad_output) = 0;

        //训练模式切换
        // train=true:  启用中间值存储，支持 backward
        // train=false: 只做推理，不存储中间值，节省显存
        virtual void set_train_mode(bool train) { is_training_ = train; }
        bool is_training() const { return is_training_; }

        // 返回该层所有可训练参数的指针列表（weight, bias 等）
        virtual std::vector<Tensor*> parameters() { return {}; }
        // 返回该层所有参数的梯度指针列表
        virtual std::vector<Tensor*> gradients()  { return {}; }
    protected:
        bool is_training_ = false;
};