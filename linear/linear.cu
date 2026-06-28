#include "linear.h"
#include "../kernels/kernels.h"
#include <stdexcept>
#include <vector>
#include <cstdlib>


Linear::Linear(int in_f, int out_f)
    :in_features(in_f),
    out_features(out_f),
    weight({in_f, out_f}),
    bias({out_f}),
    grad_weight({in_f, out_f}),
    grad_bias({out_f})
{
    init_weights_random();
}

void Linear::init_weights(float w_val, float b_val)
{
    // 权重初始化
    size_t w_num = weight.numel();
    std::vector<float> h_weight(w_num);
    for (size_t i = 0; i < w_num; ++i) {
        h_weight[i] = w_val;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(weight.data, h_weight.data(),
                                 weight.bytes(), cudaMemcpyHostToDevice));

    //偏置初始化
    size_t b_num = bias.numel();
    std::vector<float> h_bias(b_num, b_val);
    CHECK_CUDA_ERROR(cudaMemcpy(bias.data, h_bias.data(),
                                 bias.bytes(), cudaMemcpyHostToDevice));
}
void Linear::load_weights(const float* w_host, const float* b_host)
{
    CHECK_CUDA_ERROR(cudaMemcpy(weight.data, w_host,
                                 weight.bytes(), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(bias.data, b_host,
                                 bias.bytes(), cudaMemcpyHostToDevice));
}

Tensor Linear::forward(const Tensor& input)
{
    
    // 输入可能是 4D [N, C, H, W] 或 2D [N, in_features]
    // 展平除 batch 外的所有维度
    this->input_shape_ = input.shape;   // 保存原始输入形状
    int N = input.shape[0];
    int K = 1;  // 展平后的特征数
    for (size_t i = 1; i < input.shape.size(); ++i) {
        K *= input.shape[i];
    }

    if (K != in_features) {
        throw std::runtime_error(
            "Linear::forward: input feature size mismatch. "
            "Expected " + std::to_string(this->in_features) +
            ", got " + std::to_string(K));
    }

    int M = this->out_features;

    // 分配输出张量 [N, M]
    Tensor output({N, M});

    // 如果处于训练模式，需要缓存输入用于后续的反向传播（形状展平成 [N, K]）
    if (this->is_training_) {
        this->input_cache_ = Tensor({N, K});
        CHECK_CUDA_ERROR(cudaMemcpy(this->input_cache_.data, input.data,
                                     this->input_cache_.bytes(), cudaMemcpyDeviceToDevice));
    }


    // 启动 tiled 矩阵乘法核函数
    // 一个线程块负责 TILE_N × TILE_M 的输出区域
    // K 维度同样按 TILE_K 分块，在共享内存中累加
    dim3 block(TILE_M, TILE_N);   // x → M, y → N
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );
    

    linear_forward_tiled<<<grid, block>>>(
        input.data, weight.data, bias.data, output.data,
        N, K, M);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return output;
}


// 随机初始化（训练用）
void Linear::init_weights_random(unsigned int seed)
{
    // Kaiming Uniform 初始化（适用于 ReLU 激活后接的 Linear）
    float bound = sqrtf(6.0f / in_features);
    size_t w_num = weight.numel();
    std::vector<float> h_weight(w_num);
    srand(seed);
    for (size_t i = 0; i < w_num; ++i) {
        h_weight[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * bound;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(weight.data, h_weight.data(),
                                 weight.bytes(), cudaMemcpyHostToDevice));

    // 偏置初始化为 0
    std::vector<float> h_bias(out_features, 0.0f);
    CHECK_CUDA_ERROR(cudaMemcpy(bias.data, h_bias.data(),
                                 bias.bytes(), cudaMemcpyHostToDevice));

    // 梯度清零
    CHECK_CUDA_ERROR(cudaMemset(grad_weight.data, 0, grad_weight.bytes()));
    CHECK_CUDA_ERROR(cudaMemset(grad_bias.data, 0, grad_bias.bytes()));
}

Tensor Linear::backward(const Tensor& grad_output)
{
    //grad_output与本层网络的输出和下一层网络的输入形状一样，方向传播时由上一层传入作为输入 N*M
    int K = this->in_features;
    //grad_output 必须是 2D 张量
    if (grad_output.shape.size() != 2) {
        throw std::runtime_error("Linear::backward: grad_output must be 2D tensor");
    }

    int N = grad_output.shape[0];
    int M = grad_output.shape[1];

    //输出维度必须和层的 out_features 一致
    if (M != this->out_features) {
        throw std::runtime_error(
            "Linear::backward: grad_output feature size mismatch. "
            "Expected " + std::to_string(this->out_features) + 
            ", got " + std::to_string(M));
    }

    // batch 大小必须和前向缓存的输入一致
    if (N != this->input_cache_.shape[0]) {
        throw std::runtime_error(
            "Linear::backward: batch size mismatch between forward and backward. "
            "Forward N=" + std::to_string(this->input_cache_.shape[0]) + 
            ", backward N=" + std::to_string(N));
    }

    // 求grad_input = grad_output @ W^T  →  [N, K]，作为反向传播时下一层的输入
    Tensor grad_input({N, K});
    {
        dim3 block(TILE_K, TILE_N); //x->k,y->n
        dim3 grid((K + TILE_K - 1) / TILE_K, (N + TILE_N -1 ) / TILE_N);
        linear_backward_input<<<grid, block>>>(
            grad_output.data, this->weight.data, grad_input.data, N, K, M);
    }

    // 求grad_weight += input^T @ grad_output  →  [K, M]，用于更新weight
    // 注意：核函数中梯度要累加（+=），显存不足时，一个batch可能会分成多个mini-batch 实现一个完整的batch，所以这时梯度应该累积
    {
        dim3 block(TILE_M, TILE_K);
        dim3 grid((M + TILE_M - 1) / TILE_M, (K + TILE_K - 1) / TILE_K);
        linear_backward_weight<<<grid, block>>>(
            this->input_cache_.data, grad_output.data, this->grad_weight.data, N, K, M);
    }

    // grad_bias += sum over N of grad_output  →  [M], 用于更新bias
    // 同上，要使用累加
    {
        int threads = 256;
        int blocks = (M + threads - 1) / threads;
        linear_backward_bias<<<blocks, threads>>>(
            grad_output.data, this->grad_bias.data, N, M);
    }

    //将grad_input形状还原为上一层的输出形状
    grad_input.reshape(this->input_shape_);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return grad_input;

}
