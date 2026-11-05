NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -fPIC

.PHONY: all clean

all: libuvm_fragment_guard.so

libuvm_fragment_guard.so: partial_guard_shim.cu
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $<

clean:
	rm -f libuvm_fragment_guard.so
