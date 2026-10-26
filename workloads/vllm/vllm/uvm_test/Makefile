NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDE = $(CUDA_HOME)/include
CUDA_LIB = $(CUDA_HOME)/lib64

NVCCFLAGS = -Xcompiler -fPIC -O2 -std=c++17
CUDA_ARCH ?= native

all: uvm libuvm_slot_cuda_fence_kernels.so

uvm: uvm_allocator.so

uvm_allocator.so: uvm_allocator.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -shared -o $@ $< -lcudart

libuvm_slot_cuda_fence_kernels.so: uvm_slot_cuda_fence_kernels.cu
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fPIC \
		-shared -o $@ $< -L$(CUDA_LIB) -lcudart

clean:
	rm -f uvm_allocator.so libuvm_slot_cuda_fence_kernels.so

.PHONY: all uvm clean
