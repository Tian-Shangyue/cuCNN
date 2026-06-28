#pragma once
#include<math.h>
#include<stdio.h>
#define CHECK_CUDA_ERROR(val) do { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define TILE_SIZE 16  // 输出 tile 大小
#define K_TILE 8      // 一次处理 8 个输出通道
#define C_TILE 4      // 一次处理 4 个输入通道
#define TILE_N 16
#define TILE_M 16
#define TILE_K 16

 /*
 朴素直接卷积:每个线程计算输出特征图的一个元素 (n, k, h_out, w_out)。
 */
__global__ void conv2d_native(
    const float* input, //输入特征图，NCHW布局
    const float* kernel, //卷积核，KCRS布局
    const float* bias,   //偏置，K维
    float* output,      //输出特征图，NKHW布局
    int N, int C, int H, int W,     //输入特征图的维度number, channel, height, width
    int K, int R, int S,     //卷积核的维度number, height, width
    int pad_h, int pad_w,     //卷积的padding
    int stride_h, int stride_w,  //卷积的stride
    int H_out, int W_out   //输出特征图的维度height, width
);


/*
 * 基于静态多维共享内存的 Tiled 卷积核 —— 首选路径
 * R, S, stride_h, stride_w 作为模板参数，编译器生成高效多维数组索引
 */
template<int R, int S, int stride_h, int stride_w>
__global__ void conv2d_tiled(
    const float* input, //输入特征图，NCHW布局
    const float* kernel, //卷积核，KCRS布局
    const float* bias,   //偏置，K维
    float* output,      //输出特征图，NKHW布局
    int N, int C, int H, int W,     //输入特征图的维度number, channel, height, width
    int K, // int R, int S,     //卷积核的维度number, height, width（已作为模板参数）
    int pad_h, int pad_w,     //卷积的padding
    // int stride_h, int stride_w,  //卷积的stride（已作为模板参数）
    int H_out, int W_out);   //输出特征图的维度height, width




/*
 * 基于动态共享内存的 Tiled 卷积核 —— 通用回退路径
 * R, S, stride_h, stride_w 作为运行时参数，共享内存大小由启动时指定：
 *   shm_bytes = (IN_H * IN_W * C_TILE + K_TILE * C_TILE * R * S) * sizeof(float)
 *   其中 IN_H = TILE_SIZE * stride_h + R - stride_h
 *        IN_W = TILE_SIZE * stride_w + S - stride_w
 */
__global__ void conv2d_tiled_dynamic(
    const float* input, //输入特征图，NCHW布局
    const float* kernel, //卷积核，KCRS布局
    const float* bias,   //偏置，K维
    float* output,      //输出特征图，NKHW布局
    int N, int C, int H, int W,     //输入特征图的维度number, channel, height, width
    int K, int R, int S,     //卷积核的维度number, height, width
    int pad_h, int pad_w,     //卷积的padding
    int stride_h, int stride_w,  //卷积的stride
    int H_out, int W_out);   //输出特征图的维度height, width
    

/**
 * Conv2D 反向：计算 grad_input = dL/dX
 *
 * 手工逐点实现，直接通过链式法则计算每个输入像素的梯度：
 *   对每个输入位置 (n,c,h,w)，找到所有用到该位置的前向输出 (n,k,ho,wo)
 *   并乘以对应的原始卷积核权重 kernel[k][c][r][s]。
 * 即：
 *   h = ho * stride_h - pad_h + r   → 反向解出 ho = (h + pad_h - r) / stride_h
 *   w = wo * stride_w - pad_w + s   → 反向解出 wo = (w + pad_w - s) / stride_w
 * 然后累加： grad_input[n][c][h][w] += grad_output[n][k][ho][wo] * kernel[k][c][r][s]
 *
 * 注意：这是直接求和，不需要翻转 kernel。
 * 若要复用前向互相关卷积核（如 conv2d_native / conv2d_tiled） ===
 * 要1. 对 grad_output 进行上采样（仅在 stride > 1 时需要）：
 *      在高度和宽度方向相邻元素间插入 (stride_h - 1) 和 (stride_w - 1) 个零。
 *      上采样后尺寸：
 *   2. 将原始卷积核 kernel [K, C, R, S] 处理为：
 *       - 旋转最后两维 180°：kernel_rot[k][c][r][s] = kernel[k][c][R-1-r][S-1-s]
 *       - 交换 K 和 C 维度，得到 new_kernel [C, K, R, S]
 *   3. 调用前向卷积函数，传入：
 *       函数内参数对应关系：原函数中的 C 应填入 K，K 应填入 C
 *   调用后得到的 output 即为 grad_input。
 *   注意：当 stride == 1 时，无需上采样，只需翻转核并调整 padding 即可。
 *   该方法本质上是用互相关模拟了转置卷积，适用于任何 stride。
 * === 本 kernel 特点 ===
 * - 每个线程计算 grad_input 的一个元素 [n, c, h, w]
 * - 直接基于链式法则，无需上采样、无需翻转核、无需额外内存
 * - 支持任意 stride 和 padding
 *       如果要用卷积函数（如 conv2d）来等价实现，才需要预先将 kernel 旋转 180°，
 *       但本 kernel 没用那种方式。
 * 每个线程计算 grad_input 的一个元素 [n, c, h, w]
 */
__global__ void conv2d_backward_input(
    const float* grad_output,   // [N, K, H_out, W_out]
    const float* kernel,        // [K, C, R, S]
    float* grad_input,          // [N, C, H, W]
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out);

/**
 * 用tile和共享内存，只对kernel进行tile和加载共享内存
 同一个 block 内的线程访问的 grad_output 位置几乎不重叠。
 一个线程块处理一个样本的c_tile个通道的tile_size*tile_size个元素
 一个线程处理一个样本的c_tile个通道的一个元素
 * 实测用tile后更慢了！！！
 */
__global__ void conv2d_backward_input_tiled(
    const float* grad_output,   // [N, K, H_out, W_out]
    const float* kernel,        // [K, C, R, S]
    float* grad_input,          // [N, C, H, W]
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out);


/**
 * Conv2D 反向：计算 grad_weight = dL/dW
 *
 * 等价于 conv2d(input, grad_output) 但交换了 N 和 K 的角色。概念上的类比，（计算结构类似，
 * 但实际实现需要一个专门的权重梯度核，不能直接调用标准的 conv2d 函数。）
 *要求第k个卷积核的第c个通道的一个位置的梯度，需要把所有用到这个位置的输出梯度求和，
 所以要将n个样本kc的相关梯度求和得到第k个卷积核的第c个通道的一个位置的梯度
 * 每个线程计算 grad_weight 的一个元素 [k, c, r, s]
 */
__global__ void conv2d_backward_weight(
    const float* input,         // [N, C, H, W]
    const float* grad_output,   // [N, K, H_out, W_out]
    float* grad_weight,         // [K, C, R, S] — 注意：是累加，需预清零
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out);

 /**
 * 用tile和共享内存，共享内存存储grad_output,input在同一个block中重叠率低
 * 一个线程块负责一个卷积核的c_tile个通道的R*S区域
 * 每个线程负责一个卷积核的c_tile个通道的r，s位置
 * 共享内存设为存gradoutput [hout][wout], k是固定的，n循环加载，每层加载一个n进入共享内存
 * 实测用tile后更慢了！！！
 */
__global__ void conv2d_backward_weight_tiled(
    const float* input,         // [N, C, H, W]
    const float* grad_output,   // [N, K, H_out, W_out]
    float* grad_weight,         // [K, C, R, S] — 注意：是累加，需预清零
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out);


/**
 * Conv2D 反向：计算 grad_bias = dL/db
 * grad_output 在 N、H_out、W_out 三个维度上的求和
 * grad_bias[k] = Σ_n Σ_ho Σ_wo grad_output[n][k][ho][wo]
 * 每个线程处理一个输出通道的偏置梯度
 */
__global__ void conv2d_backward_bias(
    const float* grad_output,   // [N, K, H_out, W_out]
    float* grad_bias,           // [K]
    int N, int K, int H_out, int W_out);



// 上采样 scatter 核：将 grad_output 的元素按照 stride 分散到更大的张量中（其余位置已通过 memset 置零）
__global__ void upsample_grad_output_kernel(
    const float* grad_output,   // [N, K, H_out, W_out]
    float* upsampled,          // [N, K, H_up, W_up]
    int N, int K, int H_out, int W_out,
    int stride_h, int stride_w,
    int H_up, int W_up);

// 旋转卷积核并交换维度：kernel[K][C][R][S] -> kernel_rot[C][K][R][S]
__global__ void rotate_transpose_kernel_kernel(
    const float* kernel,       // [K, C, R, S]
    float* kernel_rot,        // [C, K, R, S]
    int K, int C, int R, int S);


/*
激活函数（ReLU）
*/
__global__ void relu_forward(float* data, int size);

/*
激活函数（ReLU）grid-stride优化版，一个线程加载多个，减少线程开销
*/
__global__ void relu_grid_stride_forward(float* data, int size);

/**
 * ReLU 反向传播
 * grad_input[i] = grad_output[i]  if input[i] > 0
 *               = 0.0f              otherwise
 ∂X/∂L = ∂Y/∂L⋅1[X>0]
 */
__global__ void relu_backward(
    const float* input,        // 前向时的输入 X
    const float* grad_output,  // dL/dY
    float* grad_input,         // dL/dX（输出）
    int size);


/*
池化层（MaxPooling）的实现
池化核没有c维度！！！
每个线程输出一个样本的一个通道的一个位置的元素（[n][c][h][w]）
*/
__global__ void maxpool2d(
    const float* input, //输入特征图，NCHW布局
    float* output,      //输出特征图，NCHW布局
    int N, int C, int H, int W,     //输入特征图的维度number, channel, height, width
    int pool_h, int pool_w,     //池化核的height, width
    int pad_h, int pad_w,     //填充的padding
    int stride_h, int stride_w,  //height和weight方向的stride
    int H_out, int W_out   //输出特征图的维度height, width
);

/*
池化层（MaxPooling）训练时前向的实现
池化核没有c维度！！！
每个线程输出一个样本的一个通道的一个位置的元素（[n][c][h][w]）
记录每个窗口的最大值坐标
*/
__global__ void maxpool2d_with_index(
    const float* input, //输入特征图，NCHW布局
    float* output,      //输出特征图，NCHW布局
    int* indices,   //窗口的最大值坐标
    int N, int C, int H, int W,     //输入特征图的维度number, channel, height, width
    int pool_h, int pool_w,     //池化核的height, width
    int pad_h, int pad_w,     //填充的padding
    int stride_h, int stride_w,  //height和weight方向的stride
    int H_out, int W_out   //输出特征图的维度height, width
);

// MaxPool2D 反向核函数
__global__ void maxpool2d_backward(
    const float* grad_output,   // [N * C * H_out * W_out] 展平
    const int* indices,         // [N * C * H_out * W_out] 展平，最大值在输入中的位置
    float* grad_input,          // [N * C * H * W] 展平，预初始化为 0
    int total);


/*
池化层（AvgPooling）的实现
池化核没有c维度！！！
每个线程输出一个样本的一个通道的一个位置的元素（[n][c][h][w]）
*/
__global__ void avgpool2d(
    const float* input, //输入特征图，NCHW布局
    float* output,      //输出特征图，NCHW布局
    int N, int C, int H, int W,     //输入特征图的维度number, channel, height, width
    int pool_h, int pool_w,     //池化核的height, width
    int pad_h, int pad_w,     //填充的padding
    int stride_h, int stride_w,  //height和weight方向的stride
    int H_out, int W_out   //输出特征图的维度height, width
);

/**
 * AvgPool2D 反向传播
 * 将梯度均匀分配到池化窗口内的每个位置
 * 每个线程负责 grad_output 的一个元素
 */
__global__ void avgpool2d_backward(
    const float* grad_output,   // [N, C, H_out, W_out]
    float* grad_input,          // [N, C, H, W] — 预初始化为 0
    int N, int C, int H, int W,
    int pool_h, int pool_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out);


/*
全连接层（矩阵乘法）Y=X⋅W+b
全连接层要求输入是二维矩阵 
输入矩阵 X 的形状必须是 [batch_size, in_features]（二维）
权重矩阵 W 的形状必须是 [in_features, out_features]（二维）
偏置的形状为[out_features]
因此in_features实际上是要被展平成一维的
但由于输入特征本身存储的形式其实就相当于被展平过了，只需要改变解释方式
每个线程输出一个位置的元素（相当于总的线程排布为n*m）
*/
__global__ void linear_forward(
    const float* input, // X，N*K
    const float* weight, // W, K*M
    const float* bias,  //b,M(广播会将m个元素复制n行，然后对应位置相加)
    float* output,  // 输出，大小为N*M
    int N, int K, int M //N为batch_size大小，K为in_features大小，M为out_features大小
);


/*
全连接层（矩阵乘法）,使用tile进行优化，Y=X⋅W+b
全连接层要求输入是二维矩阵 
输入矩阵 X 的形状必须是 [batch_size, in_features]（二维）
权重矩阵 W 的形状必须是 [in_features, out_features]（二维）
偏置的形状为[out_features]
因此in_features实际上是要被展平成一维的
但由于输入特征本身存储的形式其实就相当于被展平过了，只需要改变解释方式
一个线程块负责TILE_N*TILE_M，大小的输出
 K维度同样进行分块（Tile）：K -> TILE_K 每轮从全局内存加载：X Tile : [TILE_N, TILE_K] W Tile : [TILE_K, TILE_M]到共享内存（Shared Memory）。然后利用共享内存中的数据完成当前Tile的部分乘加，并在K维度上累加所有Tile的结果。
保持 TILE_M == TILE_K == TILE_N
*/
__global__ void linear_forward_tiled(
    const float* input, // X，N*K
    const float* weight, // W, K*M
    const float* bias,  //b,M(广播会将m个元素复制n行，然后对应位置相加)
    float* output,  // 输出，大小为N*M
    int N, int K, int M //N为batch_size大小，K为in_features大小，M为out_features大小
);


/**
 * Linear 反向：计算 grad_input = grad_output @ W^T
 *
 * 每个线程块负责 grad_input 的一个 TILE_N × TILE_K 区域
 * 沿 M 维度分块累加
 *
 * 计算: grad_input[n][k] = sum_m grad_output[n][m] * W[k][m]
 *                             (等价于 grad_output @ W^T)
 */
__global__ void linear_backward_input(
    const float* grad_output,   // [N, M]
    const float* weight,        // [K, M] — 注意：W 的存储是 [K, M]
    float* grad_input,          // [N, K]
    int N, int K, int M);


/**
 * Linear 反向：计算 grad_weight = X^T @ grad_output
 *
 * 每个线程块负责 grad_weight 的一个 TILE_K × TILE_M 区域
 * 沿 N 维度分块累加
 *
 * 计算: grad_weight[k][m] = sum_n input[n][k] * grad_output[n][m]
 *                               (等价于 input^T @ grad_output)
 */
__global__ void linear_backward_weight(
    const float* input,         // [N, K]
    const float* grad_output,   // [N, M]
    float* grad_weight,         // [K, M]
    int N, int K, int M);


/**
 * Linear 反向：计算 grad_bias = sum over N of grad_output
 */
__global__ void linear_backward_bias(
    const float* grad_output,   // [N, M]
    float* grad_bias,           // [M]
    int N, int M);

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
        int class_num);


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
        int class_num);


/**
 * SGD with Momentum + weight decay 参数更新核函数
 * 每个线程跨步更新多个参数
 * 对每个参数元素（grid-stride 模式，支持任意大张量）：
 *   float g = grad[i];
 *   if (weight_decay > 0) g += weight_decay * param[i];
 *   velocity[i] = momentum * velocity[i] + lr * g;
 *   param[i] = param[i] - velocity[i];
 *   grad[i] = 0.0f;     // 更新后立即清零，为下个 batch 准备
 */
__global__ void sgd_step(
    float* param,          // 参数 W，将被原地更新
    float* grad,           // 梯度 dW，将被清零
    float* velocity,       // 动量累积 v，将被更新
    float lr,
    float momentum,
    float weight_decay,
    int total);


// CPU端朴素卷积实现，与GPU核函数逻辑完全一致
void conv2d_cpu(
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int H_out, int W_out);


// 计算输出特征图尺寸
inline void compute_output_size(
    int H, int W, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int* H_out, int* W_out)
{
    *H_out = (H + 2 * pad_h - R) / stride_h + 1;
    *W_out = (W + 2 * pad_w - S) / stride_w + 1;
}

// 初始化数据（使用固定种子保证可复现）
inline void initialize_data(float* data, int size, float value) {
    for(int i = 0; i < size; ++i) {
        data[i] = value;
    }
}
