#pragma once
#include "dataset.h"
#include <vector>
#include <string>

/**
 * MNIST / FashionMNIST 数据集 — C++ 直接解析 IDX 二进制格式
 *
 * IDX 文件格式（big-endian）：
 *   [2B zero] [1B type: 0x08=uint8] [1B ndims]
 *   [ndims × 4B dim sizes]
 *   [raw data...]   (uint8, no inter-element padding)
 *
 * 图像文件：magic=0x00000803, ndims=3, dims=[N, 28, 28]
 * 标签文件：magic=0x00000801, ndims=1, dims=[N]
 *
 * 用法：
 *   MNISTDataset train(".../train-images-idx3-ubyte",
 *                      ".../train-labels-idx1-ubyte");
 *   // train.num_samples() == 60000
 *   // train.channels() == 1, train.height() == 28, train.width() == 28
 */
class MNISTDataset : public Dataset {
public:
    /// @param images_path  图像 IDX 文件路径
    /// @param labels_path  标签 IDX 文件路径
    /// @param normalize    归一化到 [0, 1]（默认 true）
    MNISTDataset(const std::string& images_path,
                 const std::string& labels_path,
                 bool normalize = true);

    void get_image(int index, float* dst) const override;
    int  get_label(int index) const override;

    // 批量获取：用 memcpy 加速
    void get_images_batch(const std::vector<int>& indices,
                          float* dst) const override;

private:
    std::vector<float> images_;  // [N * C * H * W], float32, NCHW
    std::vector<int>   labels_;  // [N], int32

    /// 读取 IDX 文件，返回原始 uint8 数据 + 维度信息
    static std::vector<uint8_t> load_idx_file(const std::string& path,
                                              std::vector<int>& dims);
    /// 大端 → 小端（IDX 头部使用 big-endian）
    static uint32_t be32_to_host(const uint8_t* be);
};
