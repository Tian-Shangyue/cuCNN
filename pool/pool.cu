#include "pool.h"
#include "../kernels/kernels.h"
#include <stdexcept>

MaxPool2D::MaxPool2D(int p_h, int p_w,
              int s_h , int s_w ,
              int ph , int pw )
    :pool_h(p_h),
    pool_w(p_w),
    stride_h(s_h == -1 ? p_h : s_h),
    stride_w(s_w == -1 ? p_w : s_w),
    pad_h(ph),
    pad_w(pw)
{
}

Tensor MaxPool2D::forward(const Tensor& input)
{
    // 解析输入形状 [N, C, H, W]
    if (input.ndim() != 4) {
        throw std::runtime_error(
            "MaxPool2D::forward: input must be 4D (N, C, H, W)");
    }

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    //计算输出尺寸
    int H_out, W_out;
    compute_output_size(H, W, pool_h, pool_w,
                        pad_h, pad_w, stride_h, stride_w,
                        &H_out, &W_out);

    // 分配输出张量 [N, C, H_out, W_out]
    Tensor output({N, C, H_out, W_out});

    //启动 MaxPool 核函数
    dim3 block(16, 16);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C
    );
    // 训练时分配 max_indices_
    if (is_training_) {
        this->max_indices_ = Tensor({N, C, H_out, W_out});
        this->input_shape_ = input.shape;
        maxpool2d_with_index<<<grid, block>>>(
        input.data, output.data,
        reinterpret_cast<int*>(max_indices_.data),
        N, C, H, W,
        pool_h, pool_w, pad_h, pad_w,
        stride_h, stride_w, H_out, W_out);
    }else{
        maxpool2d<<<grid, block>>>(
        input.data, output.data,
        N, C, H, W,
        pool_h, pool_w,
        pad_h, pad_w,
        stride_h, stride_w,
        H_out, W_out);
    }



    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    // 解析输入形状 [N, C, H_out, W_out](前向时的输出形状)
    if (grad_output.ndim() != 4) {
        throw std::runtime_error(
            "MaxPool2D::backword: grad_output must be 4D [N, C, H_out, W_out]");
    }
    // grad_output: [N, C, H_out, W_out]
    int N = grad_output.shape[0], C = grad_output.shape[1];
    int H_out = grad_output.shape[2], W_out = grad_output.shape[3];

   
    Tensor grad_input(this->input_shape_);
    CHECK_CUDA_ERROR(cudaMemset(grad_input.data, 0, grad_input.bytes()));

    int total = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    maxpool2d_backward<<<grid_size, block_size>>>(
        grad_output.data,
        reinterpret_cast<int*>(max_indices_.data),
        grad_input.data,
        total);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return grad_input;
}


AvgPool2D::AvgPool2D(int p_h, int p_w,
              int s_h , int s_w ,
              int ph , int pw )
        :pool_h(p_h),
        pool_w(p_w),
        stride_h(s_h == -1 ? p_h : s_h),
        stride_w(s_w == -1 ? p_w : s_w),
        pad_h(ph),
        pad_w(pw)
{

}
Tensor AvgPool2D::forward(const Tensor& input)
{
    // 解析输入形状 [N, C, H, W]
    if (input.ndim() != 4) {
        throw std::runtime_error(
            "AvgPool2D::forward input must be 4D (N, C, H, W)");
    }

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    if(this->is_training_){
        this->input_shape_ = input.shape;
    }

    //计算输出尺寸
    int H_out, W_out;
    compute_output_size(H, W, pool_h, pool_w,
                        pad_h, pad_w, stride_h, stride_w,
                        &H_out, &W_out);

    // 分配输出张量 [N, C, H_out, W_out]
    Tensor output({N, C, H_out, W_out});

    //启动核函数
    dim3 block(16, 16);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C
    );

    avgpool2d<<<grid, block>>>(
        input.data, output.data,
        N, C, H, W,
        pool_h, pool_w,
        pad_h, pad_w,
        stride_h, stride_w,
        H_out, W_out);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return output;
}

Tensor AvgPool2D::backward(const Tensor& grad_output)
{
    // 解析输入形状 [N, C, H_out, W_out]
    if (grad_output.ndim() != 4) {
        throw std::runtime_error(
            "AvgPool2D::backward: grad_output must be 4D [N, C, H_out, W_out]");
    }

    int N = grad_output.shape[0];
    int C = grad_output.shape[1];
    int H_out = grad_output.shape[2];
    int W_out = grad_output.shape[3];

    if (N != this->input_shape_[0] || C != this->input_shape_[1]) {
        throw std::runtime_error(
            "AvgPool2D::backward: N or C mismatch between grad_output and saved input shape");
    }
    //原始输入尺寸
    int H = this->input_shape_[2];
    int W = this->input_shape_[3];
    // 分配输出张量 [N, C, H, W]
    Tensor grad_input(this->input_shape_);
    CHECK_CUDA_ERROR(cudaMemset(grad_input.data, 0, grad_input.bytes()));

    //启动核函数
    dim3 block(16, 16);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C
    );

    avgpool2d_backward<<<grid, block>>>(
        grad_output.data, grad_input.data,
        N, C, H, W,
        pool_h, pool_w,
        pad_h, pad_w,
        stride_h, stride_w,
        H_out, W_out
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return grad_input;

}