#include "tensor.h"
#include <stdexcept>

//默认构造
Tensor::Tensor()
{
    this->data = nullptr;
}

// 按形状构造并分配显存
Tensor::Tensor(const std::vector<int>& shape)
{
    this->data = nullptr;
    this->shape = shape;
    allocate(); // 保存形状后立即分配GPU显存
}

// 析构函数
Tensor::~Tensor()
{
    release();  //释放显存
}

// 移动构造，赋值
Tensor::Tensor(Tensor&& other) noexcept
{
    // Tensor&&是右值引用，&& 是专门用来绑定右值对象的引用类型
    this->data = other.data;
    this->shape = std::move(other.shape);
    other.data = nullptr;
}

Tensor& Tensor::operator=(
        Tensor&& other
    ) noexcept
{
    if(this != &other){
        //先释放旧资源，防止直接覆盖 data 指针，旧显存的地址会彻底丢失，造成无法挽回的显存泄漏。
        release();
        this->data = other.data;
        this->shape = std::move(other.shape);
        other.data = nullptr;
    }

    return *this;
}


//手动分配内存
void Tensor::allocate()
{
    if(data != nullptr){
        return;
    }

    if(shape.empty()){
        return;
    }

    cudaError_t err =
        cudaMalloc(
            &data,
            this->bytes()
        );
    
    if(err != cudaSuccess){
        throw std::runtime_error(
            cudaGetErrorString(err)
        );
    }
}

//手动释放内存
void Tensor::release()
{
    if(data){
        cudaFree(data);
        data = nullptr;
    }
}

//返回张量的元素总个数
size_t Tensor::numel() const
{
    if(shape.empty()){
        return 0;
    }

    size_t total = 1;
    for(int dim: shape){
        total *= dim;
    }
    return total;
}

//返回张量的总字节数
size_t Tensor::bytes() const
{
    return this->numel() * sizeof(float);
}

//返回张量的维度数（秩）
int Tensor::ndim() const
{
    //size() 是 vector 的标准成员方法，返回值类型是 size_t（无符号整数）
    //static_cast<int>(...) 是 C++ 中的静态类型转换语法
    return static_cast<int>(this->shape.size());
}

//修改张量形状
void Tensor::reshape(
        const std::vector<int>& new_shape
    )
{
    size_t old_numel = numel();

    size_t new_numel = 1;

    for(int dim : new_shape)
    {
        new_numel *= dim;
    }

    if(old_numel != new_numel)
    {
        throw std::runtime_error(
            "Tensor reshape failed: numel mismatch."
        );
    }

    shape = new_shape;
}

//判断张量是否为空，是否未分配显存
bool Tensor::empty() const
{
    return this->data == nullptr;
}


void Tensor::copy_from_host(const float* src)
{
    if (data == nullptr || src == nullptr) {
        throw std::runtime_error("Tensor::copy_from_host: null pointer");
    }
    cudaError_t err = cudaMemcpy(data, src, this->bytes(),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void Tensor::copy_from_host(const std::vector<float>& src)
{
    if (src.size() != numel()) {
        throw std::runtime_error(
            "Tensor::copy_from_host: size mismatch");
    }
    copy_from_host(src.data());
}

void Tensor::copy_to_host(float* dst) const
{
    if (data == nullptr || dst == nullptr) {
        throw std::runtime_error("Tensor::copy_to_host: null pointer");
    }
    cudaError_t err = cudaMemcpy(dst, data, this->bytes(),
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

std::vector<float> Tensor::copy_to_host() const
{
    std::vector<float> result(numel());
    copy_to_host(result.data());
    return result;
}