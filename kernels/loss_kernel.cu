#include "kernels.h"

/**
 * CrossEntropyLoss 前向 + 梯度计算 联合核函数
 * 每个 block 处理一个样本
 *
 * 步骤：
 *   1. 跨步找 max(logits)，确保数值稳定
 *   2. 块内二分归约得到全局 max
 *   3. 跨步计算 exp(logits - max)，归约得到 sum
 *   4. 跨步计算 softmax，同时将 softmax - onehot 写入 grad
 *   5. loss = -log(softmax[true_class])
 确保blockDim是2幂
 */
__global__ void cross_entropy_forward(
        const float* input,      // [N, M]
        const float* targets,     // [N] — float 格式的类别索引
        float* grad,              // [N, M] — 输出梯度 = softmax - onehot
        float* losses,            // [N] — 每个样本的 loss
        int batch_size,
        int class_num)
{
    extern __shared__ float shared[];  // 动态共享内存，大小由启动时指定
    int tid = threadIdx.x;
    int n = blockIdx.x;    //当前线程块处理第几个样本
    if (n >= batch_size) return;
    int start =  n * class_num; //线程块负责样本的起始索引
    int target_idx = (int)targets[n]; //第几个样本的真实类别

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
        grad[start + i] = e;    // 暂存 exp 值
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
    __shared__ float softmax_target;    //真实标签对应类别的 softmax 概率
    for(int i = tid;i < class_num; i += blockDim.x){
        float prob = grad[start + i] / sum;
        grad[start + i] = (prob - ((i == target_idx) ? 1.0f : 0.0f)) / float(batch_size); //直接计算并写入梯度,平均损失所以梯度要除于batchsize
        if (i == target_idx){
            softmax_target = prob;  //当线程刚好处理到真实标签对应的类别时记录目标类概率用于后续计算损失
        }
    }
    __syncthreads();

    if (tid == 0) {
        losses[n] = -logf(fmaxf(softmax_target, 1e-8f));    //记录当前样本的交叉熵损失
    }
}