#include "kernels.h"

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
)
{   
    //负责的输出特征图的h方向位置
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    //负责的输出特征图的w方向位置
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    //第几个样本
    int n = blockIdx.z / C;
    //第几个通道
    int c = blockIdx.z % C;

    if(h_out >= H_out || w_out >= W_out || n >= N || c >= C){
        return;
    }

    int h_start = h_out * stride_h - pad_h; //实际计算在输入图上的h起始位置
    int w_start = w_out * stride_w - pad_w; //实际计算在输入图上的w起始位置
    float max_val = -INFINITY;

    for(int i = 0; i < pool_h; ++i){
        int h_in = h_start + i;
        for(int j = 0; j < pool_w; ++j){
            int w_in = w_start + j;
            if(h_in < H && w_in < W && h_in >= 0 && w_in >= 0){
                max_val = fmaxf(max_val, input[((n * C + c) * H + h_in) * W + w_in]);
            }
        }
    }

    output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
}

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
)
{   
    //负责的输出特征图的h方向位置
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    //负责的输出特征图的w方向位置
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    //第几个样本
    int n = blockIdx.z / C;
    //第几个通道
    int c = blockIdx.z % C;

    if(h_out >= H_out || w_out >= W_out || n >= N || c >= C){
        return;
    }

    int h_start = h_out * stride_h - pad_h; //实际计算在输入图上的h起始位置
    int w_start = w_out * stride_w - pad_w; //实际计算在输入图上的w起始位置
    float max_val = -INFINITY;
    int max_idx = -1;  // 展平索引（相对于整个输入张量）

    for(int i = 0; i < pool_h; ++i){
        int h_in = h_start + i;
        for(int j = 0; j < pool_w; ++j){
            int w_in = w_start + j;
            if(h_in < H && w_in < W && h_in >= 0 && w_in >= 0){
                float val = input[((n * C + c) * H + h_in) * W + w_in];
                if(val > max_val){
                    max_val = val;
                    max_idx = ((n * C + c) * H + h_in) * W + w_in;
                }
            }
        }
    }

    output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
    indices[((n * C + c) * H_out + h_out) * W_out + w_out] = max_idx;
}

// MaxPool2D 反向核函数
//一个线程跨步处理展平后一个grad_output位置的最大值梯度
__global__ void maxpool2d_backward(
    const float* grad_output,   // [N * C * H_out * W_out] 展平
    const int* indices,         // [N * C * H_out * W_out] 展平，最大值在输入中的位置
    float* grad_input,          // [N * C * H * W] 展平，预初始化为 0
    int total)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x){
        int input_id = indices[i];  // 该输出位置对应的最大值在输入中的展平位置
        if(input_id >= 0){
            atomicAdd(&grad_input[input_id], grad_output[i]);   //不同输出位置的最大值在前向的窗口中的位置可能是同一个，所以梯度需要累加，为防止竞争，使用原子加
        }
    }
}


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
)
{   
    //负责的输出特征图的h方向位置
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    //负责的输出特征图的w方向位置
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    //第几个样本
    int n = blockIdx.z / C;
    //第几个通道
    int c = blockIdx.z % C;

    if(h_out >= H_out || w_out >= W_out || n >= N || c >= C){
        return;
    }

    int h_start = h_out * stride_h - pad_h; //实际计算在输入图上的h起始位置
    int w_start = w_out * stride_w - pad_w; //实际计算在输入图上的w起始位置
    float acc = 0.0f;
    int pool_size = pool_h * pool_w;

    for(int i = 0; i < pool_h; ++i){
        int h_in = h_start + i;
        for(int j = 0; j < pool_w; ++j){
            int w_in = w_start + j;
            if(h_in < H && w_in < W && h_in >= 0 && w_in >= 0){
                acc += input[((n * C + c) * H + h_in) * W + w_in];
            }
        }
    }

    output[((n * C + c) * H_out + h_out) * W_out + w_out] = acc / pool_size;
}


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
    int H_out, int W_out)
{

    //  当前负责的grad_output[N, C, H_out, W_out]位置元素
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;

    if(h_out >= H_out || w_out >= W_out || n >= N || c >= C){
        return;
    }

    float grad_val = grad_output[((n * C + c) * H_out + h_out) * W_out + w_out];
    float grad_per_element = grad_val / (float)(pool_h * pool_w);

    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;

    for(int i = 0; i < pool_h; ++i){
        int h = h_start + i;
        if(h >= 0 && h < H){
            for(int j = 0; j < pool_w; ++j){
                int w = w_start + j;
                if(w >= 0 && w < W){
                     // atomicAdd：与 MaxPool 同理，可能会由多个输出梯度位置的平均梯度进行累加
                    atomicAdd(&grad_input[((n * C + c) * H + h) * W + w], grad_per_element);
                }
            }
        }
    }
}