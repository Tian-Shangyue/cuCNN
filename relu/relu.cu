#include "relu.h"
#include "../kernels/kernels.h"
#include <stdexcept>


Tensor ReLU::forward(const Tensor& input)
{
    //创建输出张量，形状与输入张量相同
    Tensor output(input.shape);


    //拷贝输入到输出
    CHECK_CUDA_ERROR(cudaMemcpy(output.data, input.data, input.bytes(), cudaMemcpyDeviceToDevice));

    // 如果训练，缓存输入
    if (this->is_training_) {
        this->input_cache_ = Tensor(input.shape);
        CHECK_CUDA_ERROR(cudaMemcpy(this->input_cache_.data, input.data,
                                     input.bytes(), cudaMemcpyDeviceToDevice));
    }

    int total = static_cast<int>(input.numel());
    if (total == 0) return output;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    

    // 限制 grid 大小，让 grid-stride 循环发挥作用
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    relu_grid_stride_forward<<<grid_size, block_size>>>(output.data, total);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return output;
}


Tensor ReLU::backward(const Tensor& grad_output)
{
    //创建输出张量，形状与输入张量相同, dL/dX
    Tensor grad_input(grad_output.shape);

    int total = static_cast<int>(grad_output.numel());
    if (total == 0) return grad_input;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    // 限制 grid 大小，让 grid-stride 循环发挥作用
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    relu_backward<<<grid_size, block_size>>>(
        this->input_cache_.data,    // 前向输入
        grad_output.data,     // dL/dY
        grad_input.data,      // dL/dX
        total);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return grad_input;
}