#include "optimizer.h"
#include "../kernels/kernels.h"
#include <stdexcept>
#include <cstdio>

SGD::SGD(float lr, float momentum, float weight_decay)
    : learning_rate_(lr), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::register_parameters(const std::vector<Tensor*>& params,
                               const std::vector<Tensor*>& grads) {
    if (params.size() != grads.size()) {
        throw std::runtime_error(
            "SGD::register_parameters: params.size() != grads.size()");
    }

    params_.clear();
    grads_.clear();
    velocities_.clear();

    size_t n = params.size();
    params_.reserve(n);
    grads_.reserve(n);
    velocities_.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        Tensor* p = params[i];
        Tensor* g = grads[i];

        // 参数和梯度的形状必须一致
        if (p->shape != g->shape) {
            throw std::runtime_error(
                "SGD::register_parameters: shape mismatch between "
                "param[" + std::to_string(i) + "] and grad[" +
                std::to_string(i) + "]");
        }

        params_.push_back(p);
        grads_.push_back(g);

        // 为每个参数创建一个同形状的速度张量，并初始化为 0
        Tensor v(p->shape);
        cudaError_t err = cudaMemset(v.data, 0, v.bytes());
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        velocities_.push_back(std::move(v));
    }

    printf("SGD: registered %zu parameter(s)\n", n);
}

void SGD::zero_grad() {
    for (size_t i = 0; i < grads_.size(); ++i) {
        cudaError_t err = cudaMemset(grads_[i]->data, 0, grads_[i]->bytes());
        if (err != cudaSuccess) {
            fprintf(stderr, "SGD::zero_grad[%zu] failed: %s\n",
                    i, cudaGetErrorString(err));
        }
    }
}

void SGD::step() {
    if (params_.empty()) {
        fprintf(stderr, "SGD::step: no parameters registered, skip\n");
        return;
    }

    for (size_t i = 0; i < params_.size(); ++i) {
        int total = static_cast<int>(params_[i]->numel());
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;
        if (grid_size > 65535) grid_size = 65535;

        sgd_step<<<grid_size, block_size>>>(
            params_[i]->data,        // 参数
            grads_[i]->data,         // 梯度
            velocities_[i].data,     // 动量
            learning_rate_,
            momentum_,
            weight_decay_,
            total);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}