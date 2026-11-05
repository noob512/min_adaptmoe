NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDE = $(CUDA_HOME)/include
CUDA_LIB = $(CUDA_HOME)/lib64

NVCCFLAGS = -Xcompiler -fPIC -O2 -std=c++17
CUDA_ARCH ?= native

all: uvm uvm_to_uvm_reuse_probe managed_to_cuda_malloc_reuse_probe \
	libuvm_slot_cuda_fence_kernels.so

uvm: uvm_allocator.so

uvm_allocator.so: uvm_allocator.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -shared -o $@ $< -lcudart

uvm_to_uvm_reuse_probe: uvm_to_uvm_reuse_probe.cu uvm_allocator.so
	$(NVCC) -O2 -std=c++17 -I$(CUDA_INCLUDE) -L. \
		-Xlinker -rpath,'$$ORIGIN' -o $@ $< \
		-l:uvm_allocator.so -L$(CUDA_LIB) -lcudart -ldl

managed_to_cuda_malloc_reuse_probe: managed_to_cuda_malloc_reuse_probe.cu uvm_allocator.so
	$(NVCC) -O2 -std=c++17 -I$(CUDA_INCLUDE) -L. \
		-Xlinker -rpath,'$$ORIGIN' -o $@ $< \
		-l:uvm_allocator.so -L$(CUDA_LIB) -lcudart -ldl

libuvm_slot_cuda_fence_kernels.so: uvm_slot_cuda_fence_kernels.cu
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fPIC \
		-shared -o $@ $< -L$(CUDA_LIB) -lcudart

clean:
	rm -f uvm_allocator.so uvm_to_uvm_reuse_probe \
		managed_to_cuda_malloc_reuse_probe \
		libuvm_slot_cuda_fence_kernels.so

.PHONY: all uvm clean
