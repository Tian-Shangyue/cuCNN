#pragma once
#include "../layer/layer.h"

/**
 * 二维最大池化层
 *
 * 输入/输出布局：NCHW
 * 池化核在 H, W 空间维度上滑动，不跨越 C 通道
 */
class MaxPool2D : public Layer
{
public:
    int pool_h, pool_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    std::vector<int> input_shape_; //反向时需要知道输入的 H, W，缓存 input shape
    Tensor max_indices_;  // [N, C, H_out, W_out] — 存储每个输出位置对应最大值的展平索引,反向：梯度只流向最大值所在的位置，其他位置梯度为 0

    /// @param p_h  池化窗口高度
    /// @param p_w  池化窗口宽度
    /// @param s_h  stride 高度（-1 表示与窗口相同，即无重叠）
    /// @param s_w  stride 宽度（-1 表示与窗口相同）
    /// @param ph   padding 高度
    /// @param pw   padding 宽度
    MaxPool2D(int p_h, int p_w,
              int s_h = -1, int s_w = -1,
              int ph = 0, int pw = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};

/**
 * 二维平均池化层
 */
class AvgPool2D : public Layer
{
public:
    int pool_h, pool_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    std::vector<int> input_shape_; //反向时需要知道输入的 H, W，缓存 input shape

    AvgPool2D(int p_h, int p_w,
              int s_h = -1, int s_w = -1,
              int ph = 0, int pw = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};
