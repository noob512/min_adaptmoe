NVCC ?= nvcc
CUDA_ARCH ?= native

TARGET := libuvm_slot_64k_overwrite_stage_sync_shim.so
SOURCE := slot_batch_shim.cu
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -fPIC -lineinfo -arch=$(CUDA_ARCH)

all: $(TARGET)

$(TARGET): $(SOURCE) ../uvm_512x4m_batch_compare/uvm_batch_shim.cu
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $< -lcudart

clean:
	rm -f $(TARGET)

.PHONY: all clean
