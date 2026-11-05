NVCC ?= nvcc
CUDA_ARCH ?= native

TARGET := uvm_slot_cuda_fence_probe
SOURCE := uvm_slot_cuda_fence_probe.cu
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo -arch=$(CUDA_ARCH)

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcuda -lcudart

clean:
	rm -f $(TARGET)

test:
	python3 -m unittest -v test_run_registered_plan.py

.PHONY: all clean test
