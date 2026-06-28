#include "kernels.h"


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
){
    int n = blockIdx.y * blockDim.y + threadIdx.y;  //对应输出矩阵的第n行
    int m = blockIdx.x * blockDim.x + threadIdx.x;  //对应输出矩阵的第m列

    if(n >= N || m >= M){
        return;
    }

    float acc = 0.0f;
    //第n行 * 第m列，（n行每行长K，m列每列长K）
    for(int k = 0; k< K; ++k){
        acc += input[(n * K) + k] * weight[(k * M) + m]; 
    }

    output[(n * M) + m] = acc + bias[m];

}


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
){
    // 共享内存：分别存储 X 和 W 的子块
    __shared__ float shm_X[TILE_N][TILE_K];
    __shared__ float shm_W[TILE_K][TILE_M];

    int n_start = blockIdx.y * TILE_N;  //当前tile的n起始位置
    int m_start = blockIdx.x * TILE_M;  //当前tile的m起始位置
    
    int n = n_start + threadIdx.y;  //对应输出矩阵的第n行
    int m = m_start + threadIdx.x;  //对应输出矩阵的第m列


    float acc = 0.0f;
    
    for(int k_tile = 0; k_tile < K; k_tile += TILE_K){
        //协助加载本轮的X tile
        int k1 = k_tile + threadIdx.x; //当前线程本轮加载元素在实际输入的k位置
        if(k1 < K && n < N){
            shm_X[threadIdx.y][threadIdx.x] = input[(n * K) + k1];
        }else{
            shm_X[threadIdx.y][threadIdx.x] = 0.0f;
        }

        //协助加载本轮的W tile
        int k2 = k_tile + threadIdx.y; //当前线程加载元素载实际权重的k位置
        if(k2 < K && m < M){
            shm_W[threadIdx.y][threadIdx.x] = weight[(k2 * M) + m];
        }else{
            shm_W[threadIdx.y][threadIdx.x] = 0.0f;
        }

        //进行线程同步确保全部加载完成
        __syncthreads();

        // 计算当前tile的乘积
        for(int k = 0; k < TILE_K; ++k){
            acc += shm_X[threadIdx.y][k] * shm_W[k][threadIdx.x];
        }

        //进行线程同步确保全部计算完成
        __syncthreads();

    }


    //写回输出，第n行 * 第m列，（n行每行长K，m列每列长K）
    if(n < N && m < M){
        output[(n * M) + m] = acc + bias[m];
    }
    
}

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
    int N, int K, int M)
{
    __shared__ float shm_go[TILE_N][TILE_M];   // grad_output tile
    __shared__ float shm_w[TILE_M][TILE_K];    // W^T tile

    int n = blockIdx.y * TILE_N + threadIdx.y;  //对应输出（grad_input）的第n行
    int k = blockIdx.x * TILE_K + threadIdx.x;  //对应输出（grad_input）的第k列

    float acc = 0.0f;

    for(int m_tile = 0; m_tile < (M + TILE_M -1 ) / TILE_M; ++m_tile){
        //协助加载本轮的grad_output tile
        int m1 = m_tile * TILE_M + threadIdx.x; //当前线程本轮加载元素在实际输入的m位置
        if(n < N && m1 < M){
            shm_go[threadIdx.y][threadIdx.x] = grad_output[n * M + m1];
        }else{
            shm_go[threadIdx.y][threadIdx.x] = 0.0f;
        }
        //协助加载本轮的W^T tile
        int m2 = m_tile * TILE_M + threadIdx.y; ///当前线程本轮加载元素在实际w的m位置
        if(m2 < M && k < K){
            shm_w[threadIdx.y][threadIdx.x] = weight[k * M + m2];
        }else{
            shm_w[threadIdx.y][threadIdx.x] = 0.0f;
        }

        //同步确保线程块内所有线程都完成
        __syncthreads();

        for(int m = 0; m < TILE_M; ++m){
            acc += shm_go[threadIdx.y][m] * shm_w[m][threadIdx.x];
        }
        //同步确保线程块内所有线程都完成
        __syncthreads();
    }
    //写回输出
    if(n < N && k < K){
        grad_input[n * K + k] = acc;
    }


}


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
    int N, int K, int M)
{
    __shared__ float shm_in[TILE_K][TILE_N];    // input(前向时的输入) tile转置
    __shared__ float shm_go[TILE_N][TILE_M];   // grad_output tile

    int k = blockIdx.y * TILE_K + threadIdx.y;   //对应输出（grad_weight）的第k行
    int m = blockIdx.x * TILE_M + threadIdx.x;   //对应输出（grad_weight）的第m列
    float acc = 0.0f;

    for(int n_tile = 0; n_tile < (N + TILE_N -1 ) / TILE_N; ++n_tile){
        //协助加载本轮的input tile
        int n1 = n_tile * TILE_N + threadIdx.x; //当前线程本轮加载元素在实际输入的n位置
        if(n1 < N && k < K){
            shm_in[threadIdx.y][threadIdx.x] = input[n1 * K + k];
        }else{
            shm_in[threadIdx.y][threadIdx.x] = 0.0f;
        }
        //协助加载本轮的grad_output tile
        int n2 = n_tile * TILE_N + threadIdx.y; ///当前线程本轮加载元素在实际grad_output的n位置
        if(n2 < N && m < M){
            shm_go[threadIdx.y][threadIdx.x] = grad_output[n2 * M + m];
        }else{
            shm_go[threadIdx.y][threadIdx.x] = 0.0f;
        }

        //同步确保线程块内所有线程都完成
        __syncthreads();

        for(int n = 0; n < TILE_N; ++n){
            acc += shm_in[threadIdx.y][n] * shm_go[n][threadIdx.x];
        }
        //同步确保线程块内所有线程都完成
        __syncthreads();
    }
    //写回输出
    if(m < M && k < K){
        grad_weight[k * M + m] += acc;
    }

}


/**
 * Linear 反向：计算 grad_bias = sum over N of grad_output
每个线程负责计算 1 个输出维度 m 对应的偏置梯度
 */
__global__ void linear_backward_bias(
    const float* grad_output,   // [N, M]
    float* grad_bias,           // [M]
    int N, int M)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;   //当前线程负责的输出维度索引 m
    if(m >= M){
        return;
    }

    float sum = 0.0f;
    //累加每行m位置的梯度
    for(int n = 0; n < N; ++n){
        sum += grad_output[n * M + m];
    }

    grad_bias[m] += sum;
}