#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024

//Kernel function, create a
__global__ void GPU_big_dot(float *A, float *B, float *C, long long N) {
	
	__shared__ float device_data[THREADS_PER_BLOCK];
	unsigned int idx, i;
	float mult;
	idx = threadIdx.x + blockIdx.x * blockDim.x; // Normal indexing

	device_data[threadIdx.x] = A[idx] * B[idx];

	__syncthreads();

	//Reduction done by the root thread in block
	if(threadIdx.x == 0)
	{
		mult = 0.0;
		for(i = 0; i < THREADS_PER_BLOCK; i++)
			mult += device_data[i];
		atomicAdd(C, mult);
	}
}	



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

	int i;
	float c = 0.0;
	for(i = 0; i < N; i++)
	{
		c += A[i] * B[i];
	}
	return c;
}



int main(int argc, char ** argv) {

	long long N = 100*512*512;
	if(argc == 2)
	{
		N = atoi(argv[1]); // Allow user to set number of elements
	}
	else
	{
		printf("ATTN: Usage: ./dot-product <NUM_ELEMS>\n");
		printf("Proceeding with default N = %lld\n\n", N);
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
	cudaMemset(device_result_gpu, 0.0, sizeof(float));

	//Compute CPU
	start_cpu = start_timer();
	result_cpu = CPU_big_dot(v1, v2, N);
	stop_cpu = stop_timer(start_cpu, name_cpu);


	//This is to ensure that with some N < THREADS_PER_BLOCK we will
	//still have a non-zero number of blocks.
    	dim3 numBlocks((N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);
	
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
	

	//STATS REGION
	printf("\nCPU Result: %e\n", result_cpu);
	printf("GPU Result: %e\n", *result_gpu);
	printf("\nSpeedup (with data transfer): %lf\n", (float)(stop_cpu)/ ((float)(stop_gpu1 + stop_gpu2 + stop_gpu3)));
    	printf("Speedup (just kernel): %lf\n", (float) stop_cpu / (float) stop_gpu2);
	printf("\nRelative Error: %lf%%\n", 1e2*abs(*result_gpu - result_cpu) / (result_cpu));
	//Cleanup and exit
	cudaFree(device_v1);
	cudaFree(device_v2);
	cudaFree(device_result_gpu);
	free(v1);
	free(v2);
	free(result_gpu);
	return 0;
}
