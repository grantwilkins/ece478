NVCC = nvcc
CUDAFLAGS = -arch=sm_61 -O2

all: dot-product.cu
	$(NVCC) dot-product.cu $(CUDAFLAGS) -o dot-product

clean:
	rm -rf a.out dot-product