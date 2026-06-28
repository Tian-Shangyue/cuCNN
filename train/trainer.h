#pragma once
#include "../sequential/sequential.h"
#include "../loss/loss.h"
#include "../optimizer/optimizer.h"
#include "../dataloader/dataloader.h"
#include <vector>
#include <cstdio>

/**
 * 训练器：封装完整的训练和评估流程
 */
class Trainer {
public:
    Sequential& model_;
    CrossEntropyLoss& criterion_;
    SGD& optimizer_;
    DataLoader& train_loader_;

    std::vector<float> train_loss_history_;
    std::vector<float> val_acc_history_;

    Trainer(Sequential& model, CrossEntropyLoss& criterion,
            SGD& optimizer, DataLoader& train_loader)
        : model_(model), criterion_(criterion),
          optimizer_(optimizer), train_loader_(train_loader) {}

    /// 训练一个 epoch
    float train_epoch();

    /// 评估准确率（使用 DataLoader）
    /// @param val_loader  验证集 DataLoader（shuffle=false）
    /// @return 准确率 [0, 1]
    float evaluate(DataLoader& val_loader);

    /// 训练 num_epochs 个 epoch
    void train(int num_epochs);
};