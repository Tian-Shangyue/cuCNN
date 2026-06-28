#include "mnist_dataset.h"
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <cstring>

/* ================================================================
 *  IDX 文件加载
 * ================================================================ */

uint32_t MNISTDataset::be32_to_host(const uint8_t* be) {
    // big-endian 4 字节 → 本机字节序 (x86 为 little-endian)
    return (static_cast<uint32_t>(be[0]) << 24) |
           (static_cast<uint32_t>(be[1]) << 16) |
           (static_cast<uint32_t>(be[2]) <<  8) |
           (static_cast<uint32_t>(be[3]));
}

std::vector<uint8_t> MNISTDataset::load_idx_file(
        const std::string& path, std::vector<int>& dims) {

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open IDX file: " + path);
    }

    // ── 头部 (big-endian) ──
    uint8_t magic_buf[4];
    file.read(reinterpret_cast<char*>(magic_buf), 4);
    if (!file.good()) {
        throw std::runtime_error("Cannot read IDX magic: " + path);
    }

    // 前两字节必须为 0
    if (magic_buf[0] != 0 || magic_buf[1] != 0) {
        throw std::runtime_error("Invalid IDX magic number in " + path);
    }

    uint8_t data_type = magic_buf[2];
    uint8_t ndims     = magic_buf[3];
    if (data_type != 0x08) {
        throw std::runtime_error(
            "Unsupported IDX data type 0x" +
            std::to_string(data_type) + " in " + path +
            " (only uint8 is supported)");
    }

    // ── 维度信息 ──
    dims.resize(ndims);
    size_t total_elements = 1;
    for (uint8_t d = 0; d < ndims; ++d) {
        uint8_t dim_buf[4];
        file.read(reinterpret_cast<char*>(dim_buf), 4);
        if (!file.good()) {
            throw std::runtime_error("Truncated IDX dims: " + path);
        }
        dims[d] = static_cast<int>(be32_to_host(dim_buf));
        total_elements *= dims[d];
    }

    // ── 数据体 (uint8, 无需字节序翻转) ──
    std::vector<uint8_t> data(total_elements);
    file.read(reinterpret_cast<char*>(data.data()), total_elements);
    if (!file.good()) {
        throw std::runtime_error("Truncated IDX data: " + path);
    }

    return data;
}


/* ================================================================
 *  构造函数
 * ================================================================ */

MNISTDataset::MNISTDataset(const std::string& images_path,
                           const std::string& labels_path,
                           bool normalize) {
    // ── 加载图像 ──
    std::vector<int> img_dims;
    auto img_raw = load_idx_file(images_path, img_dims);

    if (img_dims.size() != 3) {
        throw std::runtime_error(
            "MNIST image file expected 3 dims (N, H, W), got " +
            std::to_string(img_dims.size()) + " in " + images_path);
    }

    num_samples_ = img_dims[0];
    height_      = img_dims[1];
    width_       = img_dims[2];
    channels_    = 1;   // MNIST 是灰度图

    // uint8 → float32，可选归一化
    int img_sz = num_samples_ * height_ * width_;
    images_.resize(img_sz);
    float scale = normalize ? (1.0f / 255.0f) : 1.0f;
    for (int i = 0; i < img_sz; ++i) {
        images_[i] = static_cast<float>(img_raw[i]) * scale;
    }

    // ── 加载标签 ──
    std::vector<int> lbl_dims;
    auto lbl_raw = load_idx_file(labels_path, lbl_dims);

    if (lbl_dims.size() != 1) {
        throw std::runtime_error(
            "MNIST label file expected 1 dim (N), got " +
            std::to_string(lbl_dims.size()) + " in " + labels_path);
    }
    if (lbl_dims[0] != num_samples_) {
        throw std::runtime_error(
            "MNIST image/label count mismatch: " +
            std::to_string(num_samples_) + " images vs " +
            std::to_string(lbl_dims[0]) + " labels");
    }

    labels_.resize(num_samples_);
    for (int i = 0; i < num_samples_; ++i) {
        labels_[i] = static_cast<int>(lbl_raw[i]);
    }

    // 推断类别数 = 最大标签值 + 1
    int max_label = 0;
    for (int l : labels_) {
        if (l > max_label) max_label = l;
    }
    num_classes_ = max_label + 1;

    printf("[MNISTDataset] Loaded: %s\n", images_path.c_str());
    printf("  Samples: %d,  Shape: %dx%dx%d,  Classes: %d,  "
           "Normalize: %s\n",
           num_samples_, channels_, height_, width_, num_classes_,
           normalize ? "yes" : "no");
}


/* ================================================================
 *  数据访问
 * ================================================================ */

void MNISTDataset::get_image(int index, float* dst) const {
    int plane = height_ * width_;   // C=1，单通道
    const float* src = images_.data() + static_cast<size_t>(index) * plane;
    std::memcpy(dst, src, plane * sizeof(float));
}

int MNISTDataset::get_label(int index) const {
    return labels_[index];
}

void MNISTDataset::get_images_batch(const std::vector<int>& indices,
                                     float* dst) const {
    int plane = height_ * width_;   // 单通道的空间大小
    for (size_t i = 0; i < indices.size(); ++i) {
        const float* src = images_.data() +
                           static_cast<size_t>(indices[i]) * plane;
        std::memcpy(dst + i * plane, src, plane * sizeof(float));
    }
}
