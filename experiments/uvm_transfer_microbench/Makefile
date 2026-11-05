NVCC ?= nvcc
CUDA_ARCH ?= native

TARGET := uvm_transfer_microbench
SOURCE := uvm_transfer_microbench.cu

NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -arch=$(CUDA_ARCH)

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcudart

clean:
	rm -f $(TARGET)

.PHONY: all clean
