#pragma once
#include "../tensor/tensor.h"
#include "dataset.h"
#include <vector>
#include <random>

/**
 * 数据加载器：shuffle → batching → GPU 上传
 *
 * 不负责文件解析，只从 Dataset 读数据并按 batch 输出到 GPU。
 * 使用 pinned memory (cudaMallocHost) 加速 H→D 传输。
 *
 * 用法：
 *   MNISTDataset dataset("train-images", "train-labels");
 *   DataLoader loader(&dataset, 128, true);
 *
 *   Tensor input, labels;
 *   while (loader.has_next()) {
 *       loader.next_batch(input, labels);
 *       // input:  [B, C, H, W] on GPU
 *       // labels: [B] on GPU (类别索引存为 float)
 *       ...
 *   }
 *   loader.reset();  // 下一个 epoch
 */
class DataLoader {
public:
    /// @param dataset     外部管理生命周期，DataLoader 不负责释放
    /// @param batch_size  每批样本数
    /// @param shuffle     每个 epoch 是否打乱索引
    DataLoader(Dataset* dataset, int batch_size, bool shuffle = true);

    ~DataLoader();

    // ── 迭代接口 ──
    void reset();                          // 回到开头，可重新 shuffle
    bool has_next() const;                 // 判断是否还有 batch
    void next_batch(Tensor& input,         // [B, C, H, W]
                    Tensor& labels);       // [B]

    // ── 访问器 ──
    int num_batches()   const { return num_batches_; }
    int total_samples() const { return total_samples_; }
    int batch_size()    const { return batch_size_; }
    int channels()      const { return channels_; }
    int height()        const { return height_; }
    int width()         const { return width_; }
    int num_classes()   const { return num_classes_; }

    // ── pinned memory 可用性（调试用）──
    bool has_pinned() const { return pinned_input_ != nullptr; }

private:
    Dataset* dataset_ = nullptr;

    int total_samples_ = 0;
    int batch_size_    = 0;
    bool shuffle_      = true;

    int channels_    = 0;
    int height_      = 0;
    int width_       = 0;
    int num_classes_ = 0;

    int num_batches_    = 0;
    int current_batch_  = 0;

    std::vector<int> indices_;    // shuffle 后的索引序列
    std::mt19937     rng_;        // 随机数生成器

    // 页锁定内存（pinned memory）：DMA 直接访问，H→D 高速
    float* pinned_input_  = nullptr;
    float* pinned_labels_ = nullptr;

    void alloc_pinned(int B);
    void free_pinned();
};
