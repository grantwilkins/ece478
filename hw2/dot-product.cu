/*
Grant Wilkins
ECE 4780 Fall 2022
Homework 1

In this homework we create a CPU and GPU version of 
a floating point vector dot product. Afterwards we compare
the results for accuracy and speedup.

We utilize a threads/block of 1024, the max value available
for CUDA, however this can be adjusted as necessary.
*/

#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024


// Code provided by Dr. Jin
float * get_random_vector(int N) {
      if (N < 1) exit(1);
      // Allocate memory for the vector
      float *V = (float *) malloc(N * sizeof(float));
      if (V == NULL) exit(2);
      // Populate the vector with random numbers
      for (int i = 0; i < N; i++) V[i] = (float) rand() / (float) rand();
      // Return the randomized vector
      return V;
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

__global__ void kernel2(float *A, float *B, float *C, long long N) {
	
	// Share memory across all threads in the block
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Normal indexing

	// Share memory across all threads in the block
	__shared__ float results[THREADS_PER_BLOCK];
	results[threadIdx.x] = A[idx] * B[idx]; // find multilpication in parallel

	__syncthreads();
	//Do reduction in parallel
	//float block_sum = 0.0;
	for(unsigned int i = blockDim.x/2; i > 0; i/=2)
	{
		if(threadIdx.x < i)
			results[threadIdx.x] += results[threadIdx.x + i];
		__syncthreads();

	}

	//Protect against race conditions in
	//other blocks accessing C
	if(threadIdx.x == 0)
		atomicAdd(C, results[0]);
}



int main(int argc, char ** argv) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	long long N = 1 << 24;
	//This is to ensure that with some N < THREADS_PER_BLOCK we will
	//still have a non-zero number of blocks.
    	int numBlocks = ((N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);

	//Initial variables
	float kernel1_elapsed, kernel2_elapsed;
	float *v1, *v2, *result_ker1, *result_ker2; // host copies
	float *dev_v1, *dev_v2, *dev_ker1, *dev_ker2; // device copies
	int size = N * sizeof(float);

	//Allocate memory and generate random vectors;

	result_ker1 = (float *) malloc(sizeof(float)*numBlocks);
	result_ker2 = (float *) malloc(sizeof(float));
	*result_ker2 = 0.0;
	v1 = get_random_vector(N);
	v2 = get_random_vector(N);

	//Allocate device memory
	cudaMalloc((void **)&dev_v1, size);
	cudaMalloc((void **)&dev_v2, size);
	cudaMalloc((void **)&dev_ker1, sizeof(float)*numBlocks);
	cudaMalloc((void **)&dev_ker2, sizeof(float));

	//Initialize the result to 0.0
	cudaMemset(dev_ker2, 0.0, sizeof(float));

	printf("\n---TIMING KERNEL 1---\n");
	cudaEventRecord(start);
	//Copy the inputs to device
	cudaMemcpy(dev_v1, v1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, v2, size, cudaMemcpyHostToDevice);
	kernel1<<<numBlocks,THREADS_PER_BLOCK>>>(dev_v1, dev_v2, dev_ker1, N);
	cudaMemcpy(result_ker1,dev_ker1,numBlocks*sizeof(float),cudaMemcpyDeviceToHost);
	
	float final_ker1 = 0.0;
	int i;
	for(i = 0; i < numBlocks; i++)
		final_ker1 += result_ker1[i];

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel1_elapsed, start, stop);
	printf("GPU: Tgpu = %.5f microsec\n", ((float) kernel1_elapsed));


	printf("\n---TIMING KERNEL 2---\n");
	cudaEventRecord(start);
	//Copy the inputs to device
	cudaMemcpy(dev_v1, v1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, v2, size, cudaMemcpyHostToDevice);
	kernel2<<<numBlocks,THREADS_PER_BLOCK>>>(dev_v1, dev_v2, dev_ker2, N);
	cudaMemcpy(result_ker2,dev_ker2,sizeof(float),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel2_elapsed, start, stop);
	printf("GPU: Tgpu = %.5f microsec\n", ((float) kernel2_elapsed));
	

	//STATS REGION
	printf("\n---DATA RESULTS---\n");
	printf("Kernel 1 Result: %e\n",  final_ker1);
	printf("Kernel 2 Result: %e\n", *result_ker2);

	//Cleanup and exit
	cudaFree(dev_v1);
	cudaFree(dev_v2);
	cudaFree(dev_ker1);
	cudaFree(dev_ker2);
	free(v1);
	free(v2);
	free(result_ker1);
	free(result_ker2);
	return 0;
}
