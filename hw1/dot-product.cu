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
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

// Code provided by Dr. Jin
long long stop_timer(long long start_time, char *name) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
	printf("%s: %.5f sec\n", name, ((float) (end_time-start_time)) / (1000 * 1000));
	return (end_time-start_time);
}

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

float CPU_big_dot(float *A, float *B, long long N) {

	//Very typical O(N) dot product
	int i;
	float c = 0.0;
	for(i = 0; i < N; i++)
	{
		c += A[i] * B[i];
	}
	return c;
}

__global__ void GPU_big_dot(float *A, float *B, float *C, long long N) {
	
	// Share memory across all threads in the block
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Normal indexing

	// Share memory across all threads in the block
	__shared__ float results[THREADS_PER_BLOCK];
	results[threadIdx.x] = A[idx] * B[idx]; // find multilpication in parallel

	__syncthreads();
	//Do reduction in parallel
	float block_sum = 0.0;
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

	long long N = 100*1024*1024;
	if(argc == 2)
	{
		N = atoi(argv[1]); // Allow user to set number of elements
	}
	else
	{
		printf("ATTN: Usage: ./dot-product <NUM_ELEMS>\n");
		printf("Proceeding with default N = %lld\n", N);
	}

	//Initial variables
	long long start_cpu = 0, start_gpu = 0, stop_cpu = 0, stop_gpu1 = 0, stop_gpu2 = 0, stop_gpu3 = 0;
	char name_cpu[] = "CPU: Tcpu", name_gpu[] = "GPU Kernel: Tker", name_gpu2[] = "GPU Memcpy: Tmem";
	float *v1, *v2, *result_gpu, result_cpu = 0.0; // host copies
	float *device_v1, *device_v2, *device_result_gpu; // device copies
	int size = N * sizeof(float);

	//Allocate memory and generate random vectors;
	result_gpu = (float *) malloc(sizeof(float));
	*result_gpu = 0.0;
	v1 = get_random_vector(N);
	v2 = get_random_vector(N);

	//Allocate device memory
	cudaMalloc((void **)&device_v1, size);
	cudaMalloc((void **)&device_v2, size);
	cudaMalloc((void **)&device_result_gpu, sizeof(float));

	//Initialize the result to 0.0
	cudaMemset(device_result_gpu, 0.0, sizeof(float));

	//Compute CPU
	printf("\n---TIMING CPU---\n");
	start_cpu = start_timer();
	result_cpu = CPU_big_dot(v1, v2, N);
	stop_cpu = stop_timer(start_cpu, name_cpu);


	//This is to ensure that with some N < THREADS_PER_BLOCK we will
	//still have a non-zero number of blocks.
    	dim3 numBlocks((N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);
	
	printf("\n---TIMING GPU---\n");
	start_gpu = start_timer();
	//Copy the inputs to device
	cudaMemcpy(device_v1, v1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_v2, v2, size, cudaMemcpyHostToDevice);
    	stop_gpu1 = stop_timer(start_gpu, name_gpu2);

    	//Compute GPU
    	start_gpu = start_timer();
	GPU_big_dot<<<numBlocks,THREADS_PER_BLOCK>>>(device_v1, device_v2, device_result_gpu, N);
    	stop_gpu2 = stop_timer(start_gpu, name_gpu);

    	//Send data back
    	start_gpu = start_timer();
	cudaMemcpy(result_gpu,device_result_gpu,sizeof(float),cudaMemcpyDeviceToHost);
	stop_gpu3 = stop_timer(start_gpu, name_gpu2);
	printf("GPU: Tgpu = %.5f sec\n", (float) (stop_gpu1 + stop_gpu2 + stop_gpu3)/(1000.0*1000.0));
	

	//STATS REGION
	printf("\n---DATA RESULTS---\n");
	printf("CPU Result: %e\n", result_cpu);
	printf("GPU Result: %e\n", *result_gpu);
	printf("Relative Error: %lf%%\n", 1e2*abs(*result_gpu - result_cpu) / (result_cpu));

	printf("\n---SPEED RESULTS---\n");
	printf("Speedup (with data transfer): %lf\n", (float)(stop_cpu)/ ((float)(stop_gpu1 + stop_gpu2 + stop_gpu3)));
    	printf("Speedup (just kernel): %lf\n", (float) stop_cpu / (float) stop_gpu2);
	
	//Cleanup and exit
	cudaFree(device_v1);
	cudaFree(device_v2);
	cudaFree(device_result_gpu);
	free(v1);
	free(v2);
	free(result_gpu);
	return 0;
}
