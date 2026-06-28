#include "conv.h"
#include "../kernels/kernels.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <random>

//构造函数
Conv2D::Conv2D(int in_c, int out_c,
               int k_h, int k_w,
               int s_h, int s_w,
               int p_h, int p_w,
               bool tiled)
    : in_channels(in_c),
      out_channels(out_c),
      kernel_h(k_h), kernel_w(k_w),
      stride_h(s_h), stride_w(s_w),
      pad_h(p_h), pad_w(p_w),
      use_tiled(tiled),
      weight({out_c, in_c, k_h, k_w}),
      bias({out_c}),
      grad_weight({out_c, in_c, k_h, k_w}),
      grad_bias({out_c})
{
    init_weights_kaiming();
}

//初始化权重
void Conv2D::init_weights(float w_val, float b_val)
{
    size_t w_num = weight.numel();
    std::vector<float> h_weight(w_num);
    for (size_t i = 0; i < w_num; ++i) {
        h_weight[i] = w_val;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(weight.data, h_weight.data(),
                                 weight.bytes(), cudaMemcpyHostToDevice));

    size_t b_num = bias.numel();
    std::vector<float> h_bias(b_num, b_val);
    CHECK_CUDA_ERROR(cudaMemcpy(bias.data, h_bias.data(),
                                 bias.bytes(), cudaMemcpyHostToDevice));
}


void Conv2D::init_weights_kaiming(unsigned int seed)
{
    // 权重形状 [out_channels, in_channels, kernel_h, kernel_w]
    size_t w_num = weight.numel();
    float fan_in = static_cast<float>(in_channels * kernel_h * kernel_w);
    float std = std::sqrt(2.0f / fan_in);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, std);

    std::vector<float> h_weight(w_num);
    for (size_t i = 0; i < w_num; ++i) {
        h_weight[i] = dist(rng);
    }
    CHECK_CUDA_ERROR(cudaMemcpy(weight.data, h_weight.data(),
                                weight.bytes(), cudaMemcpyHostToDevice));

    // 偏置初始化为 0
    std::vector<float> h_bias(bias.numel(), 0.0f);
    CHECK_CUDA_ERROR(cudaMemcpy(bias.data, h_bias.data(),
                                bias.bytes(), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemset(grad_weight.data, 0, grad_weight.bytes()));
    CHECK_CUDA_ERROR(cudaMemset(grad_bias.data, 0, grad_bias.bytes()));
}

//加载权重
void Conv2D::load_weights(const float* w_host, const float* b_host)
{
    CHECK_CUDA_ERROR(cudaMemcpy(weight.data, w_host,
                                 weight.bytes(), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(bias.data, b_host,
                                 bias.bytes(), cudaMemcpyHostToDevice));
}

// 计算 tiled_dynamic 所需的共享内存大小（字节）
static size_t calc_dynamic_shm_bytes(int R, int S, int stride_h, int stride_w)
{
    int IN_H = TILE_SIZE * stride_h + R - stride_h;
    int IN_W = TILE_SIZE * stride_w + S - stride_w;

    size_t input_shm  = (size_t)IN_H * IN_W * C_TILE;
    size_t kernel_shm = (size_t)K_TILE * C_TILE * R * S;

    return (input_shm + kernel_shm) * sizeof(float);
}

// 静态模板版本（多维共享内存，编译期索引)
template<int R, int S, int stride_h, int stride_w>
static void launch_conv2d_tiled_static(
    const float* input, const float* kernel, const float* bias,
    float* output,
    int N, int C, int H, int W, int K,
    int pad_h, int pad_w,
    int H_out, int W_out)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (W_out + TILE_SIZE - 1) / TILE_SIZE,
        (H_out + TILE_SIZE - 1) / TILE_SIZE,
        N * ((K + K_TILE - 1) / K_TILE)
    );

    conv2d_tiled<R, S, stride_h, stride_w><<<grid, block>>>(
        input, kernel, bias, output,
        N, C, H, W, K,
        pad_h, pad_w,
        H_out, W_out);
}


/// 尝试匹配常见 (R,S,stride) 组合，命中则启动模板版本
/// @return true 表示匹配成功
static bool try_launch_static(
    const float* input, const float* kernel, const float* bias,
    float* output,
    int N, int C, int H, int W, int K,
    int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out)
{
    if (R == 5 && S == 5 && stride_h == 1 && stride_w == 1) {
        launch_conv2d_tiled_static<5, 5, 1, 1>(
            input, kernel, bias, output,
            N, C, H, W, K, pad_h, pad_w, H_out, W_out);
        return true;
    }
    if (R == 3 && S == 3 && stride_h == 1 && stride_w == 1) {
        launch_conv2d_tiled_static<3, 3, 1, 1>(
            input, kernel, bias, output,
            N, C, H, W, K, pad_h, pad_w, H_out, W_out);
        return true;
    }
    if (R == 3 && S == 3 && stride_h == 2 && stride_w == 2) {
        launch_conv2d_tiled_static<3, 3, 2, 2>(
            input, kernel, bias, output,
            N, C, H, W, K, pad_h, pad_w, H_out, W_out);
        return true;
    }
    if (R == 1 && S == 1 && stride_h == 1 && stride_w == 1) {
        launch_conv2d_tiled_static<1, 1, 1, 1>(
            input, kernel, bias, output,
            N, C, H, W, K, pad_h, pad_w, H_out, W_out);
        return true;
    }
    if (R == 7 && S == 7 && stride_h == 1 && stride_w == 1) {
        launch_conv2d_tiled_static<7, 7, 1, 1>(
            input, kernel, bias, output,
            N, C, H, W, K, pad_h, pad_w, H_out, W_out);
        return true;
    }
    if (R == 5 && S == 5 && stride_h == 2 && stride_w == 2) {
        launch_conv2d_tiled_static<5, 5, 2, 2>(
            input, kernel, bias, output,
            N, C, H, W, K, pad_h, pad_w, H_out, W_out);
        return true;
    }

    return false;
}


// 动态共享内存版本（1D 扁平化，任意尺寸）
static void launch_conv2d_tiled_dynamic(
    const float* input, const float* kernel, const float* bias,
    float* output,
    int N, int C, int H, int W, int K,
    int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (W_out + TILE_SIZE - 1) / TILE_SIZE,
        (H_out + TILE_SIZE - 1) / TILE_SIZE,
        N * ((K + K_TILE - 1) / K_TILE)
    );

    size_t shm_bytes = calc_dynamic_shm_bytes(R, S, stride_h, stride_w);

    conv2d_tiled_dynamic<<<grid, block, shm_bytes>>>(
        input, kernel, bias, output,
        N, C, H, W, K, R, S,
        pad_h, pad_w, stride_h, stride_w,
        H_out, W_out);
}


//朴素卷积（无共享内存，最终回退）
static void launch_conv2d_native(
    const float* input, const float* kernel, const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out)
{
    dim3 block(16, 16);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * K
    );

    conv2d_native<<<grid, block>>>(
        input, kernel, bias, output,
        N, C, H, W, K, R, S,
        pad_h, pad_w, stride_h, stride_w,
        H_out, W_out);
}


// 前向传播实现
Tensor Conv2D::forward(const Tensor& input)
{
    //解析输入形状 [N, C, H, W] 
    if (input.ndim() != 4) {
        throw std::runtime_error(
            "Conv2D::forward: input must be 4D (N, C, H, W)");
    }

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    if (C != this->in_channels) {
        throw std::runtime_error(
            "Conv2D::forward: input channel mismatch");
    }

    // 如果处于训练模式，需要缓存输入用于后续的反向传播
    if (this->is_training_) {
        this->input_cache_ = Tensor(input.shape);
        CHECK_CUDA_ERROR(cudaMemcpy(this->input_cache_.data, input.data,
                                     input.bytes(), cudaMemcpyDeviceToDevice));
    }

    int K = this->out_channels;
    int R = this->kernel_h;
    int S = this->kernel_w;

    // 计算输出尺寸
    int H_out, W_out;
    compute_output_size(H, W, R, S,
                        this->pad_h, this->pad_w, this->stride_h, this->stride_w,
                        &H_out, &W_out);

    // 分配输出张量
    Tensor output({N, K, H_out, W_out});

    // 三级回退策略
    if (use_tiled) {
        // 尝试编译期已知组合 → 多维共享内存，零额外寻址开销
        if (try_launch_static(
                input.data, this->weight.data, this->bias.data, output.data,
                N, C, H, W, K, R, S,
                this->pad_h, this->pad_w, this->stride_h, this->stride_w,
                H_out, W_out)) {
            goto launch_done;
        }

        // 未命中 → 动态共享内存版本，支持任意尺寸
        size_t shm_bytes = calc_dynamic_shm_bytes(R, S, stride_h, stride_w);
        if (shm_bytes <= 48 * 1024) {
            launch_conv2d_tiled_dynamic(
                input.data, this->weight.data, this->bias.data, output.data,
                N, C, H, W, K, R, S,
                this->pad_h, this->pad_w, this->stride_h, this->stride_w,
                H_out, W_out);
            goto launch_done;
        }
    }

    // 最终回退 → 朴素卷积
    launch_conv2d_native(
        input.data, this->weight.data, this->bias.data, output.data,
        N, C, H, W, K, R, S,
        this->pad_h, this->pad_w, this->stride_h, this->stride_w,
        H_out, W_out);

launch_done:
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) 
{
    //解析grad_output形状 [N, K, H_out, W_out]
    if (grad_output.ndim() != 4) {
        throw std::runtime_error(
            "Conv2D::backward: grad_output must be 4D (N, K, H_out, W_out)");
    }

    int N = grad_output.shape[0];
    int K = grad_output.shape[1];
    int H_out = grad_output.shape[2];
    int W_out = grad_output.shape[3];

    if (K != this->out_channels) {
        throw std::runtime_error(
            "Conv2D::backward: grad_output channel mismatch");
    }

    int C = this->in_channels;
    int H = this->input_cache_.shape[2];
    int W = this->input_cache_.shape[3];

    int R = this->kernel_h;
    int S = this->kernel_w;

    // // 1. grad_input: [N, C, H, W]
    // Tensor grad_input({N, C, H, W});
    // {
    //     if (this->use_tiled) {
    //         dim3 block(TILE_SIZE, TILE_SIZE);                
    //         dim3 grid((W + block.x - 1) / block.x,
    //                 (H + block.y - 1) / block.y,
    //                 N * ((C + C_TILE - 1) / C_TILE));     // z: 每个样本按 c_tile 分组
    //         size_t shm_size = K_TILE * C_TILE * kernel_h * kernel_w * sizeof(float); // 共享内存存放 kernel tile
    //         conv2d_backward_input_tiled<<<grid, block, shm_size>>>(
    //             grad_output.data, weight.data, grad_input.data,
    //             N, C, H, W, K, kernel_h, kernel_w,
    //             pad_h, pad_w, stride_h, stride_w,
    //             H_out, W_out);
    //     }else{
    //         dim3 block(16, 16);
    //         dim3 grid((W + block.x - 1) / block.x,
    //                 (H + block.y - 1) / block.y,
    //                 N * C);
    //         conv2d_backward_input<<<grid, block>>>(
    //             grad_output.data, weight.data, grad_input.data,
    //             N, C, H, W, K, kernel_h, kernel_w,
    //             pad_h, pad_w, stride_h, stride_w,
    //             H_out, W_out);
    //     }
        
    // }

    // 1. 计算 grad_input
    Tensor grad_input({N, C, H, W});
    {
        // 1.1 上采样 grad_output，尺寸变为 (H_out-1)*stride + 1
        int H_up = (H_out - 1) * stride_h + 1;
        int W_up = (W_out - 1) * stride_w + 1;
        Tensor grad_output_up({N, K, H_up, W_up});
        CHECK_CUDA_ERROR(cudaMemset(grad_output_up.data, 0, grad_output_up.bytes()));

        {
            dim3 block(16, 16);
            dim3 grid((W_out + block.x - 1) / block.x,
                    (H_out + block.y - 1) / block.y,
                    N * K);
            upsample_grad_output_kernel<<<grid, block>>>(
                grad_output.data, grad_output_up.data,
                N, K, H_out, W_out, stride_h, stride_w, H_up, W_up);
        }

        // 1.2 旋转卷积核并转置维度： [K,C,R,S] -> [C,K,R,S]
        Tensor kernel_rot({C, K, R, S});
        {
            dim3 block(16, 16);
            dim3 grid((S + block.x - 1) / block.x,
                    (R + block.y - 1) / block.y,
                    C * K);
            rotate_transpose_kernel_kernel<<<grid, block>>>(
                this->weight.data, kernel_rot.data, K, C, R, S);
        }

        // 1.3 计算转置卷积所需的 padding
        int new_pad_h = R - 1 - pad_h;
        int new_pad_w = S - 1 - pad_w;

        // 1.4 分配全零偏置（前向卷积会累加 bias，我们需要去掉 bias 的影响）
        float* d_zero_bias = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&d_zero_bias, C * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_zero_bias, 0, C * sizeof(float)));

        // 1.5 复用前向卷积（三级回退策略），此时 stride = 1，输出尺寸应为 H,W
        if (use_tiled) {
            // 尝试匹配静态模板版本（stride = 1）
            if (try_launch_static(
                    grad_output_up.data, kernel_rot.data, d_zero_bias, grad_input.data,
                    N, K, H_up, W_up, C, R, S,
                    new_pad_h, new_pad_w, 1, 1,
                    H, W)) {
                goto grad_input_done;
            }

            // 动态共享内存版本
            size_t shm_bytes = calc_dynamic_shm_bytes(R, S, 1, 1);
            if (shm_bytes <= 48 * 1024) {
                launch_conv2d_tiled_dynamic(
                    grad_output_up.data, kernel_rot.data, d_zero_bias, grad_input.data,
                    N, K, H_up, W_up, C, R, S,
                    new_pad_h, new_pad_w, 1, 1,
                    H, W);
                goto grad_input_done;
            }
        }

        // 回退到朴素卷积
        launch_conv2d_native(
            grad_output_up.data, kernel_rot.data, d_zero_bias, grad_input.data,
            N, K, H_up, W_up, C, R, S,
            new_pad_h, new_pad_w, 1, 1,
            H, W);

    grad_input_done:
        cudaFree(d_zero_bias);
    }

    // 2. grad_weight: [K, C, R, S]
    {
        //  if (this->use_tiled) {
        //     dim3 block(16, 16);                             // 每个线程负责一个 (r,s)
        //     dim3 grid((kernel_w + block.x - 1) / block.x,
        //             (kernel_h + block.y - 1) / block.y,
        //             K * ((C + C_TILE - 1) / C_TILE));    // z: 每个输出通道按 c_tile 分组
        //     conv2d_backward_weight_tiled<<<grid, block>>>(
        //         input_cache_.data, grad_output.data, grad_weight.data,
        //         N, C, H, W, K, kernel_h, kernel_w,
        //         pad_h, pad_w, stride_h, stride_w,
        //         H_out, W_out);
        //  }else{
        //     dim3 block(16, 16);
        //     dim3 grid((kernel_w + block.x - 1) / block.x,
        //             (kernel_h + block.y - 1) / block.y,
        //             K * C);
        //     conv2d_backward_weight<<<grid, block>>>(
        //         input_cache_.data, grad_output.data, grad_weight.data,
        //         N, C, H, W, K, kernel_h, kernel_w,
        //         pad_h, pad_w, stride_h, stride_w,
        //         H_out, W_out);
        //  }
        // 手动用朴素版速度比tiled版更快
        dim3 block(16, 16);
        dim3 grid((kernel_w + block.x - 1) / block.x,
                (kernel_h + block.y - 1) / block.y,
                K * C);
        conv2d_backward_weight<<<grid, block>>>(
            input_cache_.data, grad_output.data, grad_weight.data,
            N, C, H, W, K, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w,
            H_out, W_out);
    }

    // 3. grad_bias: [K]
    {
        int threads = 256;
        int blocks = (K + threads - 1) / threads;
        conv2d_backward_bias<<<blocks, threads>>>(
            grad_output.data, grad_bias.data,
            N, K, H_out, W_out);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return grad_input;
}