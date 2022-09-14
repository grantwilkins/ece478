#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define N 1024*1024
#define THREADS_PER_BLOCK 512

__global__ void GPU_big_dot(float *A, float *B, float *C, int N) {
	
	int i;
	__shared__ float dev[THREADS_PER_BLOCK];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	dev[threadIdx.x] = A[idx] + B[idx];
	assert(A != NULL);
	assert(B != NULL);

	if(threadIdx.x == 0)
	{
		float total = 0.0;
		for(i = 0; i < N; i++)
			total+=dev[i];
		atomicAdd(C, total);
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

float CPU_big_dot(float *A, float *B, int N) {

	int i;
	float c = 0.0;
	for(i = 0; i < N; i++)
	{
		c += A[i] + B[i];
	}
	return c;
}



int main(int argc, char ** argv) {

	long long start_cpu = 0, start_gpu = 0;
	char name_cpu[] = "CPU: Tcpu", name_gpu[] = "GPU: Tgpu";
	float *v1, *v2, *result_gpu, result_cpu = 0.0; // host copies
	float *device_v1, *device_v2, *device_result_gpu; // device copies
	int size = N * sizeof(float);

	//Allocate memory and generate random vectors;
	result_gpu = (float *) malloc(sizeof(float));
	v1 = get_random_vector(N);
	v2 = get_random_vector(N);

	//Allocate device memory
	cudaMalloc((void **)&device_v1, size);
	cudaMalloc((void **)&device_v2, size);
	cudaMalloc((void **)&device_result_gpu, sizeof(float));

	//Compute CPU
	start_cpu = start_timer();
	result_cpu = CPU_big_dot(v1, v2, N);
	stop_timer(start_cpu, name_cpu);

	//Compute GPU
	start_gpu = start_timer();
	//Copy the inputs to device
	cudaDeviceSynchronize();
	cudaMemcpy(device_v1, v1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_v2, v2, size, cudaMemcpyHostToDevice);
	GPU_big_dot<<<1,N>>>(v1, v2, result_gpu, N);
	cudaMemcpy(result_gpu,device_result_gpu,sizeof(float),cudaMemcpyDeviceToHost);
	stop_timer(start_gpu, name_gpu);

	//Cleanup and exit
	cudaFree(device_v1);
	cudaFree(device_v2);
	cudaFree(device_result_gpu);
	return 0;
}