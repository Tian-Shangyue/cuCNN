#pragma once

#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

class Tensor
{
public:

    // GPU数据指针
    float* data;

    // 任意维度Shape
    std::vector<int> shape;

public:

    Tensor();

    // 按形状构造并分配显存
    explicit Tensor(
        const std::vector<int>& shape
    );

    ~Tensor();

    // 禁止拷贝构造、拷贝赋值
    //如果允许默认浅拷贝，两个 Tensor 对象会持有同一块显存地址
    Tensor(const Tensor&) = delete;
    Tensor& operator=(
        const Tensor&
    ) = delete;

    // 允许移动构造、移动赋值
    //把一个张量的显存所有权转移给另一个对象，原对象置空
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(
        Tensor&& other
    ) noexcept;

public:

    //手动分配内存
    void allocate();

    //手动释放内存
    void release();

    //返回张量的元素总个数
    size_t numel() const;

    //返回张量的总字节数
    size_t bytes() const;

    //返回张量的维度数（秩）
    int ndim() const;

    //修改张量形状
    void reshape(
        const std::vector<int>& new_shape
    );

    //判断张量是否为空，是否未分配显存
    bool empty() const;

    /// 从主机内存拷贝到 GPU 显存
    void copy_from_host(const float* src);
    void copy_from_host(const std::vector<float>& src);

    /// 从 GPU 显存拷贝到主机内存
    void copy_to_host(float* dst) const;
    std::vector<float> copy_to_host() const;
};