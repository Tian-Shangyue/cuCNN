# CNN Inference Framework Makefile
# ================================

NVCC      := nvcc
# -Xcompiler -MMD -MP 让 nvcc 在编译时自动生成 .d 依赖文件
NVCCFLAGS := -std=c++14 -arch=native -O2 -Xcompiler -MMD -Xcompiler -MP
LDFLAGS   :=

# 源文件列表
LAYER_SRCS  := tensor/tensor.cpp conv/conv.cu relu/relu.cu pool/pool.cu linear/linear.cu softmax/softmax.cu
KERNEL_SRCS := kernels/conv_kernel.cu \
               kernels/relu_kernel.cu \
               kernels/pool_kernel.cu \
               kernels/linear_kernel.cu \
		       kernels/loss_kernel.cu \
               kernels/optimizer_kernel.cu \
               kernels/softmax_kernel.cu

# ─── 数据加载源文件 ───
DATA_SRCS   := dataloader/dataset.cpp \
               dataloader/mnist_dataset.cpp \
               dataloader/dataloader.cpp

# ─── 训练相关源文件 ───
TRAIN_SRCS  := loss/loss.cu \
               optimizer/optimizer.cu \
               train/trainer.cpp
MAIN_SRC    := main.cu

# 合并所有源文件，统一转成 .o
SRCS := $(LAYER_SRCS) $(KERNEL_SRCS) $(DATA_SRCS) $(TRAIN_SRCS) $(MAIN_SRC)
OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)          # 注意：先替换 .cpp 再替换 .cu，结果正确

TARGET := cnn_demo

# 由编译器生成的 .d 依赖文件
DEPS := $(OBJS:.o=.d)

# ==================== 规则 ====================

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# .cpp → .o
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# .cu → .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# 自动包含编译器生成的依赖文件（-include 表示文件不存在时忽略错误）
-include $(DEPS)

# 运行测试
test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)
	@echo "Clean complete"