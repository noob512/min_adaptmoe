NVCC ?= nvcc
CUDA_ARCH ?= native

TARGET := libuvm_batch_shim.so
SOURCE := uvm_batch_shim.cu
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -fPIC -lineinfo -arch=$(CUDA_ARCH)

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $< -lcudart

clean:
	rm -f $(TARGET)

.PHONY: all clean
