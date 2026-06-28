#include "trainer.h"
#include <chrono>

float Trainer::train_epoch() {
    model_.set_train_mode(true);

    train_loader_.reset();
    float total_loss = 0.0f;
    int batch_count = 0;

    // 预分配 batch 张量（形状由 DataLoader 确定）
    int B = train_loader_.batch_size();
    int C = train_loader_.channels();
    int H = train_loader_.height();
    int W = train_loader_.width();
    Tensor input({B, C, H, W});
    Tensor labels({B});  // [batch_size]，每个元素是类别索引（存为 float）

    while (train_loader_.has_next()) {
        // 1. 加载一个 batch
        train_loader_.next_batch(input, labels);

        // 2. 前向传播
        Tensor logits = model_.forward(input);

        // 3. 计算损失（CrossEntropyLoss 内部同时计算 dL/d(logits) 作为反向起点）
        float loss = criterion_.forward(logits, labels);

        // 4. 反向传播（从 loss 的梯度起点开始，链式传播到所有层）
        //    backward 后所有层的 grad_weight/grad_bias 已被更新
        const Tensor& grad_start = criterion_.backward();
        model_.backward(grad_start);

        // 5. 参数更新（step 内部会清零梯度，为下个 batch 准备）
        optimizer_.step();

        total_loss += loss;
        batch_count++;

        if (batch_count % 10 == 0) {
            printf("  Batch %d/%d, loss = %.4f\n",
                   batch_count, train_loader_.num_batches(), loss);
        }
    }

    float avg_loss = total_loss / batch_count;
    train_loss_history_.push_back(avg_loss);
    return avg_loss;
}

float Trainer::evaluate(DataLoader& val_loader) {
    model_.set_train_mode(false);  // 推理模式，不缓存中间值

    int batch_size = val_loader.batch_size();
    int C = val_loader.channels();
    int H = val_loader.height();
    int W = val_loader.width();

    Tensor input({batch_size, C, H, W});
    Tensor labels({batch_size});

    val_loader.reset();
    int correct = 0;
    int total = 0;

    while (val_loader.has_next()) {
        val_loader.next_batch(input, labels);

        Tensor logits = model_.forward(input);  // [B, num_classes]

        // 将 GPU 结果拷贝回 host 进行 argmax
        std::vector<float> h_logits = logits.copy_to_host();
        std::vector<float> h_labels = labels.copy_to_host();

        int B = (int)h_labels.size();
        for (int i = 0; i < B; ++i) {
            int true_label = (int)h_labels[i];

            // argmax over class dimension
            int pred = 0;
            float max_val = h_logits[i * val_loader.num_classes()];
            for (int c = 1; c < val_loader.num_classes(); ++c) {
                float val = h_logits[i * val_loader.num_classes() + c];
                if (val > max_val) {
                    max_val = val;
                    pred = c;
                }
            }

            if (pred == true_label) correct++;
            total++;
        }
    }

    float acc = (float)correct / total;
    val_acc_history_.push_back(acc);
    return acc;
}

void Trainer::train(int num_epochs) {
    printf("\n========== Training Started ==========\n");
    printf("Epochs: %d, Batch Size: %d, LR: %.4f, Momentum: %.2f\n",
           num_epochs, train_loader_.batch_size(),
           optimizer_.learning_rate_, optimizer_.momentum_);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        printf("\n--- Epoch %d/%d ---\n", epoch + 1, num_epochs);

        auto start = std::chrono::high_resolution_clock::now();

        float train_loss = train_epoch();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count();

        printf("Epoch %d: loss = %.4f, time = %lld ms\n",
               epoch + 1, train_loss, duration);
    }

    printf("\n========== Training Complete ==========\n");
}