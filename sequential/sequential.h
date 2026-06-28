#pragma once
#include "../layer/layer.h"
#include <vector>
#include <memory>
#include <stdexcept>

/**
 * 顺序容器：按添加顺序依次执行各层的前向传播
 *
 * 用法：
 *   Sequential model;
 *   model.add<Conv2D>(1, 6, 5, 5);
 *   model.add<ReLU>();
 *   model.add<MaxPool2D>(2, 2);
 *   Tensor output = model.forward(input);
 */
class Sequential : public Layer
{
    public:
        std::vector<std::unique_ptr<Layer>> layers;

        Sequential() = default;

        /// 添加一个层（类型 T，构造参数 Args...）
        template<typename T, typename... Args>
        T* add(Args&&... args)
        {
            auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
            T* raw = ptr.get();
            layers.push_back(std::move(ptr));
            return raw;
        }

        /// 直接添加一个已构造的层指针
        void add(Layer* layer)
        {
            layers.push_back(std::unique_ptr<Layer>(layer));
        }

        /// 前向传播：输入依次流经所有层
        Tensor forward(const Tensor& input) override
        {
            if (layers.empty()) {
                throw std::runtime_error("Sequential: empty model");
            }

            // 第一层
            Tensor current = layers[0]->forward(input);

            // 后续层：上一层的输出作为下一层的输入
            for (size_t i = 1; i < layers.size(); ++i) {
                current = layers[i]->forward(current);
            }

            return current;
        }


        // 反向传播：逆序遍历各层
        // grad_output和每层前向时的输出shape一样
        Tensor backward(const Tensor& grad_output) override {
            if (layers.empty()){
                throw std::runtime_error("Sequential: empty model");
            }

            Tensor grad(grad_output.shape);
            cudaError_t err = cudaMemcpy(grad.data, grad_output.data, grad_output.bytes(),
                                  cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(err));
            }
             // 逆序：从最后一层反向传播到第一层
            for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
                grad = layers[i]->backward(grad);   //输出和每层前向是的输入shape一样
            }
            return grad;
        }

        //训练模式切换
        void set_train_mode(bool train) override {
            //不能直接调用set_train_mode(train)修改自身的训练模式，不这样调用基类的，编译器会调用类中重写的方法，导致无限递归
            Layer::set_train_mode(train);
            for (auto& layer : layers) {
                layer->set_train_mode(train);
            }
        }
        
        //返回模型所有可训练参数的指针列表（weight, bias 等）
        std::vector<Tensor*> parameters() override {
            std::vector<Tensor*> params;
            for (auto& layer : layers) {
                auto p = layer->parameters();
                params.insert(params.end(), p.begin(), p.end());
            }
            return params;
        }

        // 返回模型所有参数的梯度指针列表
        std::vector<Tensor*> gradients() override {
            std::vector<Tensor*> grads;
            for (auto& layer : layers) {
                auto g = layer->gradients();
                grads.insert(grads.end(), g.begin(), g.end());
            }
            return grads;
        }

        /// 返回层数
        size_t size() const { return layers.size(); }
};