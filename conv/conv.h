#include "../layer/layer.h"

/**
 * 二维卷积层
 *
 * 输入布局：NCHW  [batch, in_channels, H, W]
 * 权重布局：KCRS  [out_channels, in_channels, kernel_h, kernel_w]
 * 偏置布局：K     [out_channels]
 * 输出布局：NKHW  [batch, out_channels, H_out, W_out]
 *
 * 支持两种 CUDA 核函数：
 * - conv2d_native  朴素实现，适用于任意尺寸
 * - conv2d_tiled   基于共享内存的分块优化，需要编译期确定 R, S, stride
 */
class Conv2D : public Layer
{
public:
    int in_channels;    //C
    int out_channels;   //K
    int kernel_h, kernel_w; //R,S
    int stride_h, stride_w;
    int pad_h, pad_w;   
    bool use_tiled; //是否使用tild方法进行卷积

    Tensor weight;  // shape: [K, C, R, S]
    Tensor bias;    // shape: [K]

    // 梯度存储
    Tensor grad_weight;  // [K, C, R, S]
    Tensor grad_bias;    // [K]

    // 前向中间值缓存, 用于反向计算
    Tensor input_cache_;  // [N, C, H, W]

    /// 构造卷积层
    /// @param in_c   输入通道数
    /// @param out_c  输出通道数
    /// @param k_h    卷积核高度
    /// @param k_w    卷积核宽度
    /// @param s_h    stride 高度 (default 1)
    /// @param s_w    stride 宽度 (default 1)
    /// @param p_h    padding 高度 (default 0)
    /// @param p_w    padding 宽度 (default 0)
    /// @param tiled  是否优先使用 tiled 核函数 (default true)
    Conv2D(int in_c, int out_c,
           int k_h, int k_w,
           int s_h = 1, int s_w = 1,
           int p_h = 0, int p_w = 0,
           bool tiled = true);

    /// 初始化权重（默认用固定值，方便调试）
    void init_weights(float w_val = 0.01f, float b_val = 0.0f);

    void init_weights_kaiming(unsigned int seed = 42);
    
    /// 从主机内存加载预训练权重
    void load_weights(const float* w_host, const float* b_host);

    /// 前向传播
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::vector<Tensor*> parameters() override {
        return {&weight, &bias};
    }
    std::vector<Tensor*> gradients() override {
        return {&grad_weight, &grad_bias};
    }
};
