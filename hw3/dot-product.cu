/*
Grant Wilkins
ECE 4780 Fall 2022
Homework 3

In this homework we create a multi-GPU version of 
a floating point vector dot product. 

We utilize a threads/block of 1024, the max value available
for CUDA, however this can be adjusted as necessary.
*/

#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../common.h"

#define THREADS_PER_BLOCK 1024


// Code provided by Dr. Jin
void get_random_vector(float * V, int N) {
      // Populate the vector with random numbers
      for (int i = 0; i < N; i++) V[i] = (float) rand() / (float) rand();
}

__global__ void kernel1(float *A, float *B, float *C, long long N) {
	
	// Share memory across all threads in the block
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Normal indexing

	// Share memory across all threads in the block
	__shared__ float results[THREADS_PER_BLOCK];
	float block_sum = 0.0;
	while(idx < N) {
		block_sum += A[idx] * B[idx];
		idx += blockDim.x * gridDim.x;
	}


	results[threadIdx.x] = block_sum;
	__syncthreads();
	//Do reduction in parallel
	
	for(unsigned int i = blockDim.x/2; i > 0; i/=2)
	{
		if(threadIdx.x < i)
			results[threadIdx.x] += results[threadIdx.x + i];
		__syncthreads();

	}

	//Log our result for this block
	if(threadIdx.x == 0)
		C[blockIdx.x] = results[0];
}


int main(int argc, char ** argv) {

	int ngpus;
	CHECK(cudaGetDeviceCount(&ngpus));
	long long N = 1 << 24;
	long long iSize = N/ngpus;
	long long iBytes = iSize*sizeof(float);
	int i;
	float final_ker1 = 0.0;
	//This is to ensure that with some N < THREADS_PER_BLOCK we will
	//still have a non-zero number of blocks.
    	int numBlocks = ((iSize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);

	//Initial variables
	float **v1, **v2, **result_ker1; // host copies
	float **dev_v1, **dev_v2, **dev_ker1; // device copies

	//Double array allocation
	v1 = (float **) malloc(sizeof(float *)*ngpus);
	v2 = (float **) malloc(sizeof(float *)*ngpus);
	dev_v1 = (float **) malloc(sizeof(float *)*ngpus);
	dev_v2 = (float **) malloc(sizeof(float *)*ngpus);
	result_ker1 = (float **) malloc(sizeof(float *)*ngpus);
	dev_ker1 = (float **) malloc(sizeof(float * )*ngpus);
	cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * ngpus);
	
	double iStart = seconds();
	//Allocate memory on each GPU and generate random vectors;
	for(i = 0; i < ngpus; i++)
	{
		// set current device
		CHECK(cudaSetDevice(i));
		
		// get device memory
		CHECK(cudaMalloc((void **)&dev_v1[i], iBytes));
		CHECK(cudaMalloc((void **)&dev_v2[i], iBytes));
		CHECK(cudaMalloc((void **)&dev_ker1[i], sizeof(float)*numBlocks));

		// allocate page locked host memory for asynchronous data transfer
		CHECK(cudaMallocHost((void **) &v1[i], iBytes));
		CHECK(cudaMallocHost((void **) &v2[i], iBytes));
		CHECK(cudaMallocHost((void **) &result_ker1[i], numBlocks*sizeof(float)));

		// create streams for timing and synchronizing
		CHECK(cudaStreamCreate(&stream[i]));
	}

	for(i = 0; i < ngpus; i++)
	{
		CHECK(cudaSetDevice(i));
		get_random_vector(v1[i], iSize);
		get_random_vector(v2[i], iSize);
	}

	// distributing the workload across multiple devices
	for(i = 0; i < ngpus; i++)
	{
		CHECK(cudaSetDevice(i));
		CHECK(cudaMemcpyAsync(dev_v1[i], v1[i], iBytes, cudaMemcpyHostToDevice, stream[i]));
		CHECK(cudaMemcpyAsync(dev_v2[i], v2[i], iBytes, cudaMemcpyHostToDevice, stream[i]));
		
		kernel1<<<numBlocks,THREADS_PER_BLOCK,0,stream[i]>>>(dev_v1[i], dev_v2[i], dev_ker1[i], iSize);
		
		CHECK(cudaMemcpyAsync(result_ker1[i],dev_ker1[i],numBlocks*sizeof(float),
			cudaMemcpyDeviceToHost, stream[i]));

	}

	for (i = 0; i < ngpus; i++)
    	{
      	CHECK(cudaSetDevice(i));
      	CHECK(cudaStreamSynchronize(stream[i]));
    	}

    	double iElaps = seconds() - iStart;
    	printf("%d GPU timer elapsed: %8.2fms \n", ngpus, iElaps * 1000.0);
	
	for(i = 0; i < ngpus; i++)
	{
		for(int j = 0; j < numBlocks; j++)
			final_ker1 += result_ker1[i][j];
	}
	
	printf("Kernel Result: %e\n",  final_ker1);

	// Cleanup and shutdown
	for (int i = 0; i < ngpus; i++)
	{
		CHECK(cudaSetDevice(i));
		CHECK(cudaFree(dev_v1[i]));
		CHECK(cudaFree(dev_v2[i]));
		CHECK(cudaFree(dev_ker1[i]));

		CHECK(cudaFreeHost(result_ker1[i]));
		CHECK(cudaFreeHost(v1[i]));
		CHECK(cudaFreeHost(v2[i]));
		CHECK(cudaStreamDestroy(stream[i]));

		CHECK(cudaDeviceReset());
	}

	//Cleanup and exit
	free(dev_v1);
	free(dev_v2);
	free(dev_ker1);
	free(v1);
	free(v2);
	free(stream);
	free(result_ker1);
	return 0;
}
