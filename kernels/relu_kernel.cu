#include "kernels.h"

/*
激活函数（ReLU）
*/
__global__ void relu_forward(float* data, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

/*
激活函数（ReLU）grid-stride优化版，一个线程加载多个，减少线程开销
*/
__global__ void relu_grid_stride_forward(float* data, int size){
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        data[i] = fmaxf(0.0f, data[i]);   
    }
}

/**
 * ReLU 反向传播
 * grad_input[i] = grad_output[i]  if input[i] > 0
 *               = 0.0f              otherwise
 ∂X/∂L = ∂Y/∂L⋅1[X>0]
 一个线程加载处理多个，减少线程开销
 */
__global__ void relu_backward(
    const float* input,        // 前向时的输入 X
    const float* grad_output,  // dL/dY
    float* grad_input,         // dL/dX（输出）
    int size)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;   
    }
}

