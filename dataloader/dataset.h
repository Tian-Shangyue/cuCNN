#pragma once
#include <vector>
#include <string>
#include <cstdint>

/**
 * 数据集抽象基类
 *
 * 所有数据集统一提供 NCHW float32 图像 + int32 标签。
 * 子类负责从具体文件格式解析，基类定义访问接口。
 */
class Dataset {
public:
    int num_samples_ = 0;
    int channels_    = 0;
    int height_      = 0;
    int width_       = 0;
    int num_classes_ = 0;

    virtual ~Dataset() {}

    /// 获取单个样本图像数据
    /// @param dst  预分配 C*H*W 个 float 的缓冲区
    virtual void get_image(int index, float* dst) const = 0;

    /// 获取单个样本标签
    virtual int get_label(int index) const = 0;

    /// 批量获取图像（子类可重写优化，如 memcpy）
    /// @param indices  样本索引数组，长度 batch_size
    /// @param dst      输出缓冲区，大小 batch_size * C * H * W
    virtual void get_images_batch(const std::vector<int>& indices,
                                  float* dst) const;

    // 便捷访问
    int num_samples() const { return num_samples_; }
    int img_size()     const { return channels_ * height_ * width_; }
    int img_bytes()    const { return img_size() * sizeof(float); }
};
