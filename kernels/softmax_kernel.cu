#include "kernels.h"


/**
Softmax 的 CUDA 实现
为数值稳定性，需要先减去最大值
一个线程块负责一个样本的输出
由于一个样本的class_num可能大于线程块的最大线程数量，所以需要采用跨步的方式才能处理所有元素
一个线程可能需要处理多个元素
确保blockDim是2幂
 */
__global__ void softmax_forward(
        const float* input, 
        float* output,
        int batch_size,
        int class_num){
     extern __shared__ float shared[];  // 动态共享内存，大小由启动时指定
     int tid = threadIdx.x;
     int n = blockIdx.x;    //当前线程块处理第几个样本
     if (n >= batch_size) return;
     int start =  n * class_num; //线程块负责样本的起始索引

     //找最大值
     float max_val = -INFINITY;
     //跨步找当前线程负责的几个元素的最大值
     for(int i = tid; i < class_num; i += blockDim.x){
        max_val = fmaxf(max_val, input[start + i]);
     }
     shared[tid] = max_val;
    //同步线程确保所有线程都找到自己负责元素的最大值
    __syncthreads();
    // 块内二分归约，得到整个样本的全局最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads(); // 每一轮归约都必须同步
    }
    max_val = shared[0]; // 最终全局最大值存在shared[0]

    //计算当前线程负责的所有元素的exp值和sum和
    float sum = 0.0f;
    for(int i = tid; i < class_num; i += blockDim.x){
        float e = expf(input[start + i] - max_val);
        output[start + i] = e;
        sum += e;
    }
    shared[tid] = sum;
    //同步确保所有线程都计算完负责的元素的exp值和sum和
    __syncthreads();
    // 块内二分归约，得到整个样本的sum和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = shared[tid] + shared[tid + s];
        }
        __syncthreads(); // 每一轮归约都必须同步
    }
    sum = shared[0]; // 最终全局sum和存在shared[0]

    //跨步对本线程负责的所有元素进行归一化并输出
    for(int i = tid;i < class_num; i += blockDim.x){
        output[start + i] = output[start + i] / sum;
    }
}