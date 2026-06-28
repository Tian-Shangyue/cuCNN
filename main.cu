#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#include "./sequential/sequential.h"
#include "./conv/conv.h"
#include "./relu/relu.h"
#include "./pool/pool.h"
#include "./linear/linear.h"
#include "./softmax/softmax.h"
#include "./loss/loss.h"
#include "./optimizer/optimizer.h"
#include "./dataloader/mnist_dataset.h"
#include "./dataloader/dataloader.h"
#include "./train/trainer.h"


/// 打印张量信息
void print_tensor(const char* name, const Tensor& t)
{
    printf("  %-12s  shape: [", name);
    for (size_t i = 0; i < t.shape.size(); ++i) {
        printf("%d%s", t.shape[i],
               i + 1 < t.shape.size() ? ", " : "");
    }
    printf("]  numel: %zu\n", t.numel());
}


/// 创建填充了指定值的 4D 张量
Tensor make_input(int N, int C, int H, int W, float val = 1.0f)
{
    Tensor t({N, C, H, W});
    size_t n = t.numel();
    std::vector<float> host(n);
    for (size_t i = 0; i < n; ++i) {
        host[i] = val;
    }
    t.copy_from_host(host);
    return t;
}


// simple网络 前向推理
void demo_simpleCNN_forward()
{
    printf("\n");
    printf("  simpleCNN Inference Demo\n");
    //1. 构建模型 
    printf("\n[1] Building model...\n");

    Sequential model;

    //第一层Conv2d 输入3×224×224 7×7 卷积核，输出 64 通道，步长 2，填充 3，带偏置 输出64×112×112
    model.add<Conv2D>(3, 64, 7, 7, 2, 2, 3, 3, true);
    //第2层 ReLU 输入64×112×112 无参数 输出64×112×112
    model.add<ReLU>();
    //第3层 MaxPool2d 输入64×112×112 3×3池化核，步长2，填充1 输出64×56×56
    model.add<MaxPool2D>(3, 3, 2, 2, 1, 1);
    //第4层 Conv2d 输入64×56×56 3×3卷积核，输出64通道，步长1，填充1，带偏置 输出64×56×56
    model.add<Conv2D>(64, 64, 3, 3, 1, 1, 1, 1, true);
    //第5层 ReLU 输入64×56×56 无参数 输出64×56×56
    model.add<ReLU>();
    //第6层 Conv2d 输入64×56×56 3×3卷积核，输出64通道，步长1，填充1，带偏置 输出64×56×56
    model.add<Conv2D>(64, 64, 3, 3, 1, 1, 1, 1, true);
    //第7层 ReLU 输入64×56×56 无参数 输出64×56×56
    model.add<ReLU>();
    //第8层 Conv2d 输入64×56×56 3×3卷积核，输出128通道，步长2，填充1，带偏置 输出128×28×28
    model.add<Conv2D>(64, 128, 3, 3, 2, 2, 1, 1, true);
    //第9层 ReLU 输入128×28×28 无参数 输出128×28×28
    model.add<ReLU>();
    //第10层 Conv2d 输入128×28×28 3×3卷积核，输出128通道，步长1，填充1，带偏置 输出128×28×28
    model.add<Conv2D>(128, 128, 3, 3, 1, 1, 1, 1, true);
    //第11层 ReLU 输入128×28×28 无参数 输出128×28×28
    model.add<ReLU>();
    //第12层 Conv2d 输入128×28×28 3×3卷积核，输出256通道，步长2，填充1，带偏置 输出256×14×14
    model.add<Conv2D>(128, 256, 3, 3, 2, 2, 1, 1, true);
    //第13层 ReLU 输入256×14×14 无参数 输出256×14×14
    model.add<ReLU>();
    //第14层 Conv2d 输入256×14×14 3×3卷积核，输出256通道，步长1，填充1，带偏置 输出256×14×14
    model.add<Conv2D>(256, 256, 3, 3, 1, 1, 1, 1, true);
    //第15层 ReLU 输入256×14×14 无参数 输出256×14×14
    model.add<ReLU>();
    //第16层 AdaptiveAvgPool2d 输入256×14×14 全局平均池化到1×1 输出256×1×1
    model.add<AvgPool2D>(14, 14, 14, 14, 0, 0);
    //第17层 Flatten 输入256×1×1 展平为一维向量 输出256维
    //第18层 Linear 输入256维 输出10维，带偏置 输出10维
    model.add<Linear>(256, 10);
    //第19层 Softmax 输入10维 归一化概率和为1 输出10维
    model.add<Softmax>();

    // 2. 构造输入
    printf("\n[2] Creating input tensor [3, 3, 224, 224]...\n");
    Tensor input = make_input(3, 3, 224, 224, 0.5f);

    //3. 前向推理
    printf("\n[3] Running forward pass...\n");
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = model.forward(input);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

     // 4. 输出结果
    printf("\n[4] Results:\n");
    print_tensor("output", output);

    // 预测类别
    std::vector<float> h_out = output.copy_to_host();
    for(int n = 0; n < 3; ++n){
        int pred_class = 0;
        float max_prob = h_out[n * 10 + 0];

        printf("  Sample %d probs: ", n);
        for (int i = 0; i < 10; ++i) {
            float prob = h_out[n * 10 + i]; // 正确的行优先索引
            printf("%.4f ", prob);
            
            if (prob > max_prob) {
                max_prob = prob;
                pred_class = i;
            }
        }
        printf("\n");
        printf("  Sample %d Predicted class: %d (prob = %.4f)\n\n", n, pred_class, max_prob);
    }
    
}


/**
 * 构建 MNIST 训练模型
 *
 * 网络结构：
 *   Conv2D(1→8, 5×5) → ReLU → MaxPool(2×2)
 *   Conv2D(8→16, 5×5) → ReLU → MaxPool(2×2)
 *   Linear(16*4*4, 128) → ReLU
 *   Linear(128, 10)
 *
 * 输入: [N, 1, 28, 28]
 * 输出: [N, 10]
 */
Sequential build_mnist_model() {
    Sequential model;

    // Block 1: 1x28x28 → 8x24x24 → 8x12x12
    model.add<Conv2D>(1, 8, 5, 5, 1, 1, 0, 0, false);
    model.add<ReLU>();
    model.add<MaxPool2D>(2, 2);

    // Block 2: 8x12x12 → 16x8x8 → 16x4x4
    model.add<Conv2D>(8, 16, 5, 5, 1, 1, 0, 0, false);
    model.add<ReLU>();
    model.add<MaxPool2D>(2, 2);

    // Classifier
    model.add<Linear>(16 * 4 * 4, 128);
    model.add<ReLU>();
    model.add<Linear>(128, 10);

    return model;
}


// int main()
// {   
//     printf("CUDA CNN Inference Framework\n");

//     // 检查 CUDA 设备
//     int device_count = 0;
//     cudaGetDeviceCount(&device_count);
//     if (device_count == 0) {
//         fprintf(stderr, "Error: No CUDA device found!\n");
//         return 1;
//     }

//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     printf("Device: %s\n", prop.name);
//     printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

//     // 运行测试
//     printf("\n--- Tests ---");
//     demo_simpleCNN_forward();
//     return 0;
// }

int main() {
    printf("CUDA CNN Training Framework\n\n");

    // 检查 CUDA 设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "Error: No CUDA device found!\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n\n", prop.name);

    Dataset* train_set = nullptr;
    Dataset* test_set  = nullptr;

   
    train_set = new MNISTDataset(
            "./data/MNIST/raw/train-images-idx3-ubyte",
            "./data/MNIST/raw/train-labels-idx1-ubyte");
    test_set = new MNISTDataset(
            "./data/MNIST/raw/t10k-images-idx3-ubyte",
            "./data/MNIST/raw/t10k-labels-idx1-ubyte");
    // 构建模型
    Sequential model;
    model = build_mnist_model();
    // 损失函数
    CrossEntropyLoss criterion;
    // 优化器
    SGD optimizer(0.01f, 0.9f, 5e-4f);
    optimizer.register_parameters(model.parameters(), model.gradients());
    // 数据加载器
    int batch_size = 128;
    DataLoader train_loader(train_set, batch_size, true);   // shuffle
    DataLoader test_loader(test_set, batch_size, false);    // no shuffle
    // 训练
    Trainer trainer(model, criterion, optimizer, train_loader);
    int num_epochs = 5;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        printf("\n--- Epoch %d/%d ---\n", epoch + 1, num_epochs);

        auto start = std::chrono::high_resolution_clock::now();

        float train_loss = trainer.train_epoch();
        float val_acc = trainer.evaluate(test_loader);

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count();

        printf("Epoch %d: loss=%.4f, val_acc=%.2f%%, time=%lldms\n",
               epoch + 1, train_loss, val_acc * 100.0f, ms);
    }

    // 训练完成后进行推理演示
    printf("\n\n===== Inference Demo (Loaded Model) =====\n");
    model.set_train_mode(false);   // 关闭训练模式，减少缓存

    // 按 Trainer 的习惯，预先分配好 batch 的 Tensor
    int B = test_loader.batch_size();
    int C = test_loader.channels();    // 需要 DataLoader 提供 channels/height/width
    int H = test_loader.height();
    int W = test_loader.width();
    int num_classes = test_loader.num_classes();

    Tensor input({B, C, H, W});
    Tensor labels({B});   // 标签存储为 float，实际是类别索引

    test_loader.reset();
    if (test_loader.has_next()) {
        // 使用与 Trainer 完全一致的加载方式
        test_loader.next_batch(input, labels);

        // 前向推理
        Tensor output = model.forward(input);   // shape: [B, num_classes]
        Softmax softmax_layer;
        // softmax_layer.set_train_mode(false);  // 如果有该接口可调用，无则忽略
        Tensor logits = softmax_layer.forward(output);

        // 将结果拷贝回 CPU
        std::vector<float> h_logits = logits.copy_to_host();
        std::vector<float> h_labels = labels.copy_to_host();

        int show_samples = 10;   // 展示前10个样本
        int correct = 0;
        for (int i = 0; i < show_samples && i < B; ++i) {
            // 寻找最大概率的类别
            float max_prob = h_logits[i * num_classes];
            int pred = 0;
            for (int c = 1; c < num_classes; ++c) {
                float prob = h_logits[i * num_classes + c];
                if (prob > max_prob) {
                    max_prob = prob;
                    pred = c;
                }
            }
            int true_label = static_cast<int>(h_labels[i]);
            if (pred == true_label) correct++;

            printf("Sample %2d: True=%d, Pred=%d (prob=%.4f) %s\n",
                i, true_label, pred, max_prob,
                (pred == true_label) ? "y" : "x");
        }
        printf("Accuracy on first %d samples: %.2f%%\n",
            show_samples, correct * 100.0f / show_samples);
    }


    // 清理
    delete train_set;
    delete test_set;

    printf("\nTraining complete!\n");
    return 0;
}