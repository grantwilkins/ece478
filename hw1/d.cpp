#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

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
float *get_random_vector(int N) {
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
	long long start = 0, stop = 0;
	float c = 0.0;
	assert(A != NULL);
	assert(B != NULL);
	start = start_timer();
	for(i = 0; i < N; i++)
	{
		c += A[i] + B[i];
	}
	stop_timer(start, "CPU Dot Product");
	return c;
}

float *GPU_big_dot(float *A, float *B, int N) {

}

int main(int argc, char ** argv) {

	int N = 100;
	float * v1 = get_random_vector(N);
	float * v2 = get_random_vector(N);
	CPU_big_dot(v1, v2, N);
}