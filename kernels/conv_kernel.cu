#include "kernels.h"

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
)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nk = blockIdx.z; // 将 N*K 折叠到 grid z 维度
    int n = nk / K; 
    int k = nk % K;

    if(n >= N || k >= K || h_out >=H_out || w_out >= W_out) return; //边界检查

    float sum = 0.0f;

    for(int c = 0; c< C; ++c){  //遍历输入特征图的每个通道
        for(int r = 0; r < R; ++r){ //遍历卷积核的height维度
            for(int s = 0; s<S; ++s){   //遍历卷积核的width维度
                //h_out 每增加 1，h_in 就增加 stride_h。如果 stride_h = 2，那么输出坐标走 1 步，输入坐标走 2 步
                int h_in = h_out * stride_h - pad_h + r; //本次计算对应的输入特征图像素在height维度上的位置, h*stride相当于无填充时的起始位置，有填充得减去才能得到真正的其实位置，下面的w同样如此
                int w_in = w_out * stride_w - pad_w + s; //本次计算对应的输入特征图像素在width维度上的位置

                if(h_in >= 0 && w_in >=0 && h_in < H && w_in < W){ //边界检查，确保访问的输入特征图像素在有效范围内,同时对于填充为0的情况无需计算
                    sum += input[((n * C + c) * H + h_in) * W + w_in] 
                        // 输入张量(NCHW布局)索引计算：
                        // n*C*H*W → 前n个样本的所有元素
                        // + c*H*W → 加上当前样本中前c个输入通道的所有元素
                        // + h_in*W → 加上当前通道中前h_in行的所有元素
                        // + w_in → 加上当前行的第w_in列元素
                        * kernel[((k * C + c) * R + r) * S + s];
                        // 权重张量(KCRS布局)索引计算：
                        // k*C*R*S → 前k个输出通道的所有卷积核元素
                        // + c*R*S → 加上当前输出通道中前c个输入通道的所有元素
                        // + r*S → 加上当前输入通道中前r行的所有元素
                        // + s → 加上当前行的第s列元素
                }
            }
        }
    }

    sum += bias[k]; //加上第k个输出通道(卷积核)的偏置
    output[((n * K + k) * H_out + h_out) * W_out + w_out] = sum; // 输出张量(NKHW布局)索引计算：
    //n * K * H_out * H_out → 前n个样本的所有元素
    // + k * H_out * H_out → 加上当前样本中前k个输出通道的所有元素
    // + h_out * H_out → 加上当前输出通道中前h_out行的所有元素
    // + w_out → 加上当前行的第w_out列元素
}


/*
 * 基于静态多维共享内存的 Tiled 卷积核（分块卷积）—— 首选优化路径
 * 核心优化：将全局内存访问转化为高速共享内存访问，最大化数据复用率
 * 线程映射：每个线程块(Block)负责计算1个样本的K_TILE个输出通道的TILE_SIZE×TILE_SIZE空间区域
 * 每个线程(Thread)负责计算该区域内1个空间位置的K_TILE个输出元素
 *
 * 模板参数 R, S, stride_h, stride_w 在编译期确定，编译器可直接生成
 * 高效的多维共享内存数组索引（无额外地址计算开销）
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
    int H_out, int W_out)   //输出特征图的维度height, width
{
    // 共享内存尺寸由模板参数在编译期确定
    __shared__ float shm_input[TILE_SIZE * stride_h + R - stride_h]
                                [TILE_SIZE * stride_w + S - stride_w]
                                [C_TILE];
    __shared__ float shm_kernel[K_TILE][C_TILE][R][S];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h_out = by * TILE_SIZE + ty;
    int w_out = bx * TILE_SIZE + tx;

    int n = blockIdx.z / ((K + K_TILE - 1) / K_TILE);
    int k_tile = blockIdx.z % ((K + K_TILE - 1) / K_TILE);
    int k_start = k_tile * K_TILE;

    float acc[K_TILE] = {0.0f};

    for (int c_tile = 0; c_tile < ((C + C_TILE - 1) / C_TILE); ++c_tile) {
        int c_start = c_tile * C_TILE;

        // 协作加载输入 tile
        for (int c_offset = 0; c_offset < C_TILE; ++c_offset) {
            int c = c_start + c_offset;
            for (int i = ty; i < (TILE_SIZE * stride_h + R - stride_h); i += TILE_SIZE) {
                for (int j = tx; j < (TILE_SIZE * stride_w + S - stride_w); j += TILE_SIZE) {
                    int h_in = (by * TILE_SIZE * stride_h - pad_h) + i;
                    int w_in = (bx * TILE_SIZE * stride_w - pad_w) + j;

                    if (h_in < H && h_in >= 0 && w_in < W && w_in >= 0
                            && c < C && n < N) {
                        shm_input[i][j][c_offset]
                            = input[((n * C + c) * H + h_in) * W + w_in];
                    } else {
                        shm_input[i][j][c_offset] = 0.0f;
                    }
                }
            }
        }

        // 加载权重 tile
        for (int k_offset = 0; k_offset < K_TILE; ++k_offset) {
            int k = k_start + k_offset;
            for (int c_offset = 0; c_offset < C_TILE; ++c_offset) {
                int c = c_start + c_offset;
                for (int r = ty; r < R; r += TILE_SIZE) {
                    for (int s = tx; s < S; s += TILE_SIZE) {
                        if (k < K && c < C && r < R && s < S) {
                            shm_kernel[k_offset][c_offset][r][s]
                                = kernel[((k * C + c) * R + r) * S + s];
                        } else {
                            shm_kernel[k_offset][c_offset][r][s] = 0.0f;
                        }
                    }
                }
            }
        }

        __syncthreads();

        // 卷积计算
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = ty * stride_h + r;
                int w_in = tx * stride_w + s;
                for (int c_offset = 0; c_offset < C_TILE; ++c_offset) {
                    float input_val = shm_input[h_in][w_in][c_offset];
                    for (int k_offset = 0; k_offset < K_TILE; ++k_offset) {
                        acc[k_offset] += input_val
                            * shm_kernel[k_offset][c_offset][r][s];
                    }
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
    for (int k_offset = 0; k_offset < K_TILE; ++k_offset) {
        int k = k_start + k_offset;
        if (n < N && k < K && h_out < H_out && w_out < W_out) {
            output[((n * K + k) * H_out + h_out) * W_out + w_out]
                = acc[k_offset] + bias[k];
        }
    }
}


/*
 * 基于动态共享内存的 Tiled 卷积核（分块卷积）—— 通用回退路径
 * 与上面的模板版本逻辑完全一致，区别在于：
 * - R, S, stride_h, stride_w 作为运行时参数传入（不再是模板参数）
 * - 共享内存使用 extern __shared__ 1D 扁平化分配
 * - 多维索引通过手动计算偏移量实现
 *
 * 动态共享内存布局：
 *   [0      .. IN_H*IN_W*C_TILE-1]      shm_input  [IN_H][IN_W][C_TILE]
 *   [offset .. offset+K_TILE*C_TILE*R*S-1] shm_kernel [K_TILE][C_TILE][R][S]
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
    int H_out, int W_out)   //输出特征图的维度height, width
{
    // ---- 输入 tile 尺寸（运行时计算）----
    int IN_H = TILE_SIZE * stride_h + R - stride_h;
    int IN_W = TILE_SIZE * stride_w + S - stride_w;

    // ---- 1D 索引步长 ----
    int IN_STRIDE_H = IN_W * C_TILE;       // shm_input[h][w][c] → h*IN_STRIDE_H + w*IN_STRIDE_W + c
    int IN_STRIDE_W = C_TILE;

    int KER_BASE   = IN_H * IN_W * C_TILE; // shm_kernel 起始偏移
    int KER_STRIDE_K = C_TILE * R * S;     // shm_kernel[k][c][r][s] → KER_BASE + k*KER_STRIDE_K + c*KER_STRIDE_C + r*KER_STRIDE_R + s
    int KER_STRIDE_C = R * S;
    int KER_STRIDE_R = S;

    // ---- 动态共享内存 ----
    extern __shared__ float shm[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h_out = by * TILE_SIZE + ty;
    int w_out = bx * TILE_SIZE + tx;

    int k_tiles_per_sample = (K + K_TILE - 1) / K_TILE;
    int n   = blockIdx.z / k_tiles_per_sample;
    int k_tile = blockIdx.z % k_tiles_per_sample;
    int k_start = k_tile * K_TILE;

    // 寄存器累加器
    float acc[K_TILE] = {0.0f};

    // 遍历所有输入通道，每次处理 C_TILE 个
    int c_tiles = (C + C_TILE - 1) / C_TILE;
    for (int c_tile = 0; c_tile < c_tiles; ++c_tile) {
        int c_start = c_tile * C_TILE;

        // ============ 协作加载输入 tile ============
        for (int c_offset = 0; c_offset < C_TILE; ++c_offset) {
            int c = c_start + c_offset;
            for (int i = ty; i < IN_H; i += TILE_SIZE) {
                for (int j = tx; j < IN_W; j += TILE_SIZE) {
                    int h_in = (by * TILE_SIZE * stride_h - pad_h) + i;
                    int w_in = (bx * TILE_SIZE * stride_w - pad_w) + j;

                    int idx = i * IN_STRIDE_H + j * IN_STRIDE_W + c_offset;

                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W
                            && c < C && n < N) {
                        shm[idx] = input[((n * C + c) * H + h_in) * W + w_in];
                    } else {
                        shm[idx] = 0.0f;
                    }
                }
            }
        }

        // ============ 协作加载权重 tile ============
        for (int k_offset = 0; k_offset < K_TILE; ++k_offset) {
            int k = k_start + k_offset;
            for (int c_offset = 0; c_offset < C_TILE; ++c_offset) {
                int c = c_start + c_offset;
                for (int r = ty; r < R; r += TILE_SIZE) {
                    for (int s = tx; s < S; s += TILE_SIZE) {
                        int idx = KER_BASE
                                + k_offset * KER_STRIDE_K
                                + c_offset * KER_STRIDE_C
                                + r * KER_STRIDE_R + s;
                        if (k < K && c < C && r < R && s < S) {
                            shm[idx] = kernel[((k * C + c) * R + r) * S + s];
                        } else {
                            shm[idx] = 0.0f;
                        }
                    }
                }
            }
        }

        __syncthreads();

        // ============ 卷积计算 ============
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = ty * stride_h + r;
                int w_in = tx * stride_w + s;
                for (int c_offset = 0; c_offset < C_TILE; ++c_offset) {
                    float input_val = shm[h_in * IN_STRIDE_H + w_in * IN_STRIDE_W + c_offset];
                    for (int k_offset = 0; k_offset < K_TILE; ++k_offset) {
                        acc[k_offset] += input_val * shm[KER_BASE
                                + k_offset * KER_STRIDE_K
                                + c_offset * KER_STRIDE_C
                                + r * KER_STRIDE_R + s];
                    }
                }
            }
        }

        __syncthreads();
    }

    // ============ 写回结果 ============
    for (int k_offset = 0; k_offset < K_TILE; ++k_offset) {
        int k = k_start + k_offset;
        if (n < N && k < K && h_out < H_out && w_out < W_out) {
            output[((n * K + k) * H_out + h_out) * W_out + w_out]
                = acc[k_offset] + bias[k];
        }
    }
}




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
    int H_out, int W_out)
{
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if(n >= N || c >= C || h >= H || w >= W){
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                // 根据公式 h=h_out * stride_h - pad_h + r, 反解出grad_output中的hout位置，w同理
                int h_s_h = h + pad_h - r;
                int w_s_w = w + pad_w - s;
                if (h_s_h >= 0 && w_s_w >= 0 && h_s_h % stride_h == 0 && w_s_w % stride_w == 0){
                    int h_out = h_s_h / stride_h;
                    int w_out = w_s_w / stride_w;
                    if(h_out < H_out && w_out < W_out){
                        sum += grad_output[((n * K + k) * H_out +h_out) * W_out + w_out] * 
                                kernel[((k * C + c) * R + r) * S + s];
                    }
                }
            }
        }
    }


    grad_input[((n * C + c) * H + h) * W + w] = sum;


}


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
    int H_out, int W_out)
{
    extern __shared__ float shm_kernel[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h = by * TILE_SIZE + ty;
    int w = bx * TILE_SIZE + tx;
    int n = blockIdx.z / ((C + C_TILE - 1) / C_TILE);

    int c_tile = blockIdx.z % ((C + C_TILE - 1) / C_TILE);
    int c_start = c_tile * C_TILE;

    float acc[C_TILE] = {0.0f};

    // 循环处理所有k_tile
    for(int k_tile = 0; k_tile < (K + K_TILE -1) / K_TILE; ++k_tile){
            //协助加载当前ktile的kernel
            int k_start = k_tile * K_TILE;
            for(int k_offset = 0; k_offset < K_TILE; ++k_offset){
                int k = k_start + k_offset;
                for(int c_offset = 0; c_offset < C_TILE; ++c_offset){
                    int c = c_start + c_offset;
                    for(int r = ty; r < R; r += TILE_SIZE){
                        for(int s = tx; s < S; s += TILE_SIZE){
                            if(k < K && c < C && r < R && s <S){
                                shm_kernel[((k_offset * C_TILE + c_offset) * R + r) * S +s] = kernel[((k * C + c) * R + r) * S + s];
                            }else{
                                shm_kernel[((k_offset * C_TILE + c_offset) * R + r) * S +s] = 0.0f;
                            }
                        }
                    }
                }
            }

            // 同步确保所有线程加载完成
            __syncthreads();

            // 计算
            for(int k_offset = 0; k_offset < K_TILE; ++k_offset){
                int k = k_start + k_offset;
                if(k >= K)
                    continue;
                for (int r = 0; r < R; ++r) {
                    for (int s = 0; s < S; ++s) {
                        // 根据公式 h=h_out * stride_h - pad_h + r, 反解出grad_output中的hout位置，w同理
                        int h_s_h = h + pad_h - r;
                        int w_s_w = w + pad_w - s;
                        if (h_s_h >= 0 && w_s_w >= 0 && h_s_h % stride_h == 0 && w_s_w % stride_w == 0){
                            int h_out = h_s_h / stride_h;
                            int w_out = w_s_w / stride_w;
                            if(h_out < H_out && w_out < W_out){
                                float grad_val = grad_output[((n * K + k) * H_out +h_out) * W_out + w_out];
                                for(int c_offset = 0; c_offset < C_TILE; ++c_offset){
                                    acc[c_offset] += grad_val * shm_kernel[((k_offset * C_TILE + c_offset) * R + r) * S +s];
                                }
                            }
                            
                        }
                    }
                }
            }

            // 同步确保所有线程加载完成
            __syncthreads();

    }

    // 写回结果
    for(int c_offset = 0; c_offset < C_TILE; ++c_offset){
        int c = c_start + c_offset;
        if (n < N && c < C && h < H && w < W){
            grad_input[((n * C + c) * H + h) * W + w] = acc[c_offset];
        } 
    }
}


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
    int H_out, int W_out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int kc = blockIdx.z;
    int k = kc / C;
    int c = kc % C;

    if (k >= K || c >= C || r >= R || s >= S){
        return;
    }

    float sum = 0.0f;

    for(int n = 0; n < N ; ++n){
        for(int h_out = 0 ;h_out < H_out; ++h_out){
            for(int w_out = 0; w_out < W_out; ++w_out){
                // 根据公式计算input的h位置
                int h = h_out * stride_h -pad_h + r;
                int w = w_out * stride_w - pad_w + s;
                if(h >= 0 && h < H && w >= 0 && w < W){
                    sum += grad_output[((n * K + k) * H_out + h_out) * W_out + w_out] * 
                            input[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }

    grad_weight[((k * C + c) * R + r) * S + s] = sum;
}


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
    int H_out, int W_out)
{
    __shared__ float shm_gout[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int r = by * blockDim.y + ty;
    int s = bx * blockDim.x + tx;
    bool valid_rs = (r < R && s < S);
    int k = blockIdx.z / ((C + C_TILE - 1) / C_TILE);

    int c_tile = blockIdx.z % ((C + C_TILE - 1) / C_TILE);
    int c_start = c_tile * C_TILE;

    float acc[C_TILE] = {0.0f};
    
    for(int n = 0; n < N; ++n){
        // 分块遍历 grad_output 的空间维度
        for (int h_tile = 0; h_tile < H_out; h_tile += TILE_SIZE) {
            for (int w_tile = 0; w_tile < W_out; w_tile += TILE_SIZE) {
                // 每个线程跨步加载多个元素
                for(int h_offset = ty; h_offset < TILE_SIZE; h_offset += blockDim.y){
                    int h_out = h_tile + h_offset;
                    for(int w_offset = tx; w_offset < TILE_SIZE; w_offset += blockDim.x){
                        int w_out = w_tile + w_offset;
                        if(h_out < H_out && w_out < W_out){
                            shm_gout[h_offset][w_offset] = grad_output[((n * K + k) * H_out + h_out) * W_out + w_out];
                        }else{
                            shm_gout[h_offset][w_offset] = 0.0f;
                        }
                    }
                }

                // 同步确保所有线程完成
                __syncthreads();


                // 计算
                for (int c_offset = 0; c_offset < C_TILE; ++c_offset) {
                    int c = c_start + c_offset;
                    if (c >= C) continue;

                    for (int h_offset = 0; h_offset < TILE_SIZE; ++h_offset) {
                        int h_out = h_tile + h_offset;
                        if (h_out >= H_out) continue;
                        int h_in = h_out * stride_h - pad_h + r;
                        if (h_in < 0 || h_in >= H) continue;

                        for (int w_offset = 0; w_offset < TILE_SIZE; ++w_offset) {
                            int w_out = w_tile + w_offset;
                            if (w_out >= W_out) continue;
                            int w_in = w_out * stride_w - pad_w + s;
                            if (w_in < 0 || w_in >= W) continue;

                            float go = shm_gout[h_offset][w_offset];
                            acc[c_offset] += go * input[((n * C + c) * H + h_in) * W + w_in];
                        }
                    }
                }
                 // 同步确保所有线程完成
                __syncthreads();

            }
        }
    }

    for(int c_offset = 0; c_offset < C_TILE; ++c_offset){
        int c = c_start + c_offset;
        if(c < C && valid_rs){
            grad_weight[((k * C + c) * R + r) * S + s] = acc[c_offset];
        }
    }
}


/**
 * Conv2D 反向：计算 grad_bias = dL/db
 * grad_output 在 N、H_out、W_out 三个维度上的求和
 * grad_bias[k] = Σ_n Σ_ho Σ_wo grad_output[n][k][ho][wo]
 * 每个线程跨步处理输出通道的偏置梯度
 */
__global__ void conv2d_backward_bias(
    const float* grad_output,   // [N, K, H_out, W_out]
    float* grad_bias,           // [K]
    int N, int K, int H_out, int W_out)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= K){
        return;
    }

    int size = H_out * W_out;
    float sum = 0.0f;

    for(; k < K; k += gridDim.x * blockDim.x){
        for(int n = 0; n < N; ++n){
            int start = (n * K + k) * H_out * W_out;
            for(int idx = 0; idx < size; ++idx){
                sum += grad_output[start + idx];
            }
        }

        grad_bias[k] = sum;
        sum = 0.0f;
        
    }

}

// 上采样 scatter 核：将 grad_output 的元素按照 stride 分散到更大的张量中（其余位置已通过 memset 置零）
__global__ void upsample_grad_output_kernel(
    const float* grad_output,   // [N, K, H_out, W_out]
    float* upsampled,          // [N, K, H_up, W_up]
    int N, int K, int H_out, int W_out,
    int stride_h, int stride_w,
    int H_up, int W_up)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nk = blockIdx.z;
    int n = nk / K;
    int k = nk % K;

    if (n >= N || k >= K || h_out >= H_out || w_out >= W_out) return;

    int h_up = h_out * stride_h;
    int w_up = w_out * stride_w;
    upsampled[((n * K + k) * H_up + h_up) * W_up + w_up] =
        grad_output[((n * K + k) * H_out + h_out) * W_out + w_out];
}

// 旋转卷积核并交换维度：kernel[K][C][R][S] -> kernel_rot[C][K][R][S]
__global__ void rotate_transpose_kernel_kernel(
    const float* kernel,       // [K, C, R, S]
    float* kernel_rot,        // [C, K, R, S]
    int K, int C, int R, int S)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int ck = blockIdx.z;
    int c = ck / K;
    int k = ck % K;

    if (c >= C || k >= K || r >= R || s >= S) return;

    kernel_rot[((c * K + k) * R + r) * S + s] =
        kernel[((k * C + c) * R + (R - 1 - r)) * S + (S - 1 - s)];
}


// 显式实例化，确保链接器能找到这些符号
template __global__ void conv2d_tiled<3, 3, 2, 2>(
    const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, int, int);

template __global__ void conv2d_tiled<1, 1, 1, 1>(
    const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, int, int);

template __global__ void conv2d_tiled<5, 5, 1, 1>(
    const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, int, int);

template __global__ void conv2d_tiled<3, 3, 1, 1>(
    const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, int, int);

template __global__ void conv2d_tiled<5, 5, 2, 2>(
    const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, int, int);

template __global__ void conv2d_tiled<7, 7, 1, 1>(
    const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, int, int);