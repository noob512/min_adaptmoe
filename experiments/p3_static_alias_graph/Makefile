NVCC ?= nvcc
TARGET := p3_static_alias_graph
SOURCE := p3_static_alias_graph.cu

NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
ARCH_FLAGS ?= -arch=native
HOST_WARNINGS ?= -Xcompiler=-Wall,-Wextra
LDLIBS ?= -lcuda

.PHONY: all clean smoke

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $(HOST_WARNINGS) $< -o $@ $(LDLIBS)

smoke: $(TARGET)
	./$(TARGET) --layers 4 --banks 2 --regions-per-layer 8 \
		--copy-chunk-kib 2048 --verify-mode canary \
		--verify-blocks 32 --warmups 1 --repeats 2
	./$(TARGET) --layers 4 --banks 2 --regions-per-layer 4 \
		--copy-chunk-kib 512 --verify-mode full \
		--verify-blocks 16 --warmups 0 --repeats 1 \
		--modes graph_region_alias

clean:
	rm -f $(TARGET)
