CLC_FLAGS := --platform_filter "AMD Accelerated Parallel Processing" --device_type GPU --add_headers
KERNEL_FLAGS := -D TILE_COLS=32 -D TILE_ROWS=8
CL_FLAGS := "-I clcommons/include -cl-std=CL1.2 $(KERNEL_FLAGS)"
all:
	clcc $(CLC_FLAGS) --cloptions=$(CL_FLAGS) kernels.cl -o kernels.out
