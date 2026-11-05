NVCC ?= nvcc
CUDA_ARCH ?= native

TARGET := libvllm_slot_direct_e2e.so
SOURCE := slot_e2e_shim.cu
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -fPIC -lineinfo -arch=$(CUDA_ARCH)

all: $(TARGET)

$(TARGET): $(SOURCE) \
	../uvm_512x4m_slot_shared_push_compare/slot_batch_shim.cu \
	../uvm_512x4m_batch_compare/uvm_batch_shim.cu
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $< -lcudart

clean:
	rm -f $(TARGET)

.PHONY: all clean
