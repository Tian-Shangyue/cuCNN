#include "dataloader.h"
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

// 复用 kernels.h 的 CHECK_CUDA_ERROR 宏
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(val) do { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif


DataLoader::DataLoader(Dataset* dataset, int batch_size, bool shuffle)
    : dataset_(dataset),
      total_samples_(dataset->num_samples()),
      batch_size_(batch_size),
      shuffle_(shuffle),
      channels_(dataset->channels_),
      height_(dataset->height_),
      width_(dataset->width_),
      num_classes_(dataset->num_classes_) {

    if (dataset_ == nullptr) {
        throw std::runtime_error("DataLoader: dataset is null");
    }
    if (batch_size_ <= 0) {
        throw std::runtime_error("DataLoader: batch_size must be > 0");
    }

    num_batches_ = total_samples_ / batch_size_;
    if (num_batches_ == 0) {
        fprintf(stderr,
                "[DataLoader] WARNING: batch_size=%d > total_samples=%d, "
                "num_batches=0\n",
                batch_size_, total_samples_);
    }

    // 初始化索引序列 [0, 1, 2, ..., N-1]
    indices_.resize(total_samples_);
    for (int i = 0; i < total_samples_; ++i) indices_[i] = i;

    // 随机数种子（用时钟）
    auto seed = static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    rng_.seed(seed);

    // 预分配 pinned memory
    alloc_pinned(batch_size_);

    // 首次 shuffle
    reset();

    printf("[DataLoader] batch_size=%d, batches=%d, shuffle=%d, "
           "pinned=%s\n",
           batch_size_, num_batches_, shuffle_,
           has_pinned() ? "yes" : "no");
}


DataLoader::~DataLoader() {
    free_pinned();
}


/* ================================================================
 *  Pinned memory
 * ================================================================ */

void DataLoader::alloc_pinned(int B) {
    size_t img_bytes = static_cast<size_t>(B) * channels_ *
                       height_ * width_ * sizeof(float);
    size_t lbl_bytes = static_cast<size_t>(B) * sizeof(float);

    cudaError_t e1 = cudaMallocHost(&pinned_input_, img_bytes);
    if (e1 != cudaSuccess) {
        fprintf(stderr,
                "[DataLoader] cudaMallocHost(input) failed: %s\n",
                cudaGetErrorString(e1));
        pinned_input_ = nullptr;
    }

    cudaError_t e2 = cudaMallocHost(&pinned_labels_, lbl_bytes);
    if (e2 != cudaSuccess) {
        fprintf(stderr,
                "[DataLoader] cudaMallocHost(labels) failed: %s\n",
                cudaGetErrorString(e2));
        pinned_labels_ = nullptr;
    }
}


void DataLoader::free_pinned() {
    if (pinned_input_) {
        cudaFreeHost(pinned_input_);
        pinned_input_ = nullptr;
    }
    if (pinned_labels_) {
        cudaFreeHost(pinned_labels_);
        pinned_labels_ = nullptr;
    }
}


/* ================================================================
 *  迭代
 * ================================================================ */

void DataLoader::reset() {
    current_batch_ = 0;
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}


bool DataLoader::has_next() const {
    return current_batch_ < num_batches_;
}


void DataLoader::next_batch(Tensor& input, Tensor& labels) {
    if (!has_next()) {
        throw std::runtime_error("DataLoader: no more batches");
    }

    int B = batch_size_;
    int img_sz = channels_ * height_ * width_;

    // 提取本 batch 的索引
    int start = current_batch_ * B;
    std::vector<int> batch_idx(
        indices_.begin() + start,
        indices_.begin() + start + B);

    if (pinned_input_ && pinned_labels_) {
        // ── 快速路径：写入 pinned memory，H→D 传输 ──
        dataset_->get_images_batch(batch_idx, pinned_input_);
        for (int i = 0; i < B; ++i) {
            pinned_labels_[i] =
                static_cast<float>(dataset_->get_label(batch_idx[i]));
        }

        CHECK_CUDA_ERROR(cudaMemcpy(
            input.data, pinned_input_,
            static_cast<size_t>(B) * img_sz * sizeof(float),
            cudaMemcpyHostToDevice));

        CHECK_CUDA_ERROR(cudaMemcpy(
            labels.data, pinned_labels_,
            static_cast<size_t>(B) * sizeof(float),
            cudaMemcpyHostToDevice));
    } else {
        // ── 回退路径：普通 host 内存 ──
        std::vector<float> h_input(B * img_sz);
        std::vector<float> h_labels(B);
        dataset_->get_images_batch(batch_idx, h_input.data());
        for (int i = 0; i < B; ++i) {
            h_labels[i] =
                static_cast<float>(dataset_->get_label(batch_idx[i]));
        }
        input.copy_from_host(h_input);
        labels.copy_from_host(h_labels);
    }

    current_batch_++;
}
