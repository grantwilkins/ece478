NVCC = nvcc
CUDAFLAGS = -O2 -g

all: dot-product.cu
	$(NVCC) dot-product.cu $(CUDAFLAGS) -o dot-product

clean:
	rm -rf a.out dot-product