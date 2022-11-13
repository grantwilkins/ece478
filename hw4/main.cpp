#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <CL/cl.h>
#define N 100
#define BLOCK_SIZE 1

char* loadProgSource(const char* filename, const char* preamble, size_t *sz) {
  FILE* fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char* sourceString;

  // Open the OpenCL source code file
  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);

  // Get the length of the source code
  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);

  // Allocate a buffer for the source code string and read it in
  sourceString = (char *) calloc(szSource + szPreamble+1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
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
	printf("%s: %.5f ms\n", name, ((float) (end_time-start_time)) / (1000));
	return (end_time-start_time);
}

int main()
{

	/* INITIALIZE MATRICES */
	float *inputMatrix1;
	float *inputMatrix2;
	float *results;
	cl_uint width = N;

	int x,y;
	int data = 0;
	inputMatrix1 = (float *) malloc(sizeof(float) * width * width);
	inputMatrix2 = (float *) malloc(sizeof(float) * width * width);
	results = (float *) malloc(sizeof(float) * width * width);

	// Populate the input matrices
	for(y = 0; y < width; y++)
	{
		for(x = 0; x < width; x++)
		{
			inputMatrix1[y * width + x] = data;
			inputMatrix2[y * width + x] = data;
			results[y * width + x] = 0;
			data++;
		}
	}

	/* OPENCL VARIABLE INSTANTIATION */
	cl_platform_id platform_id;
	cl_uint num_of_platforms = 0;
	cl_uint num_of_devices = 0;
	cl_device_id device_id;
	cl_context_properties properties[3];
	cl_int err;
	cl_context context;
	cl_command_queue command_queue;
	char *kernelSource;
	size_t kernelSize;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputMatrix1_mem_obj, inputMatrix2_mem_obj, results_mem_obj;
	size_t global[2] = {N, N};
	size_t local[2] = {BLOCK_SIZE, BLOCK_SIZE};
	cl_event prof_event, event;
	cl_command_queue comm;
	cl_ulong start_time, end_time, run_time;
	size_t return_bytes;
	long long start = 0, end = 0;

	// Retrives a list of platforms available
	if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
		printf("Unable to get platform_id\n");
		return 1;
	}

	// Get a supported GPU device
	if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, 
		&num_of_devices) != CL_SUCCESS) {
		printf("Unable to get device_id\n");
		return 1;
	}

	// Context properties list (must be terminated with 0)
	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties) platform_id;
	properties[2] = 0;

	// Create a context with the GPU device
	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

	// Create a command queue using the context and device
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);

	// Load kernel file, prepend static info, and return total kernel size
	kernelSource = loadProgSource("matrix_mul.cl", "", &kernelSize);

	// Create a program from the kernel source code
	program = clCreateProgramWithSource(context, 1, (const char **) 
			&kernelSource, NULL, &err);

	// Compile the program
	if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
		printf("Error building program\n");
		return 1;
	}

	//BULLET POINT 3
	char buffer[4096];
	size_t length;
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 
		sizeof(buffer), buffer, &length);
	printf("%s\n", buffer);


	// Specify which kernel from the program to execute
	kernel = clCreateKernel(program, "matrix_mul", &err);

	// Create buffers for the input and output
	inputMatrix1_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N, NULL, NULL);
	inputMatrix2_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N, NULL, NULL);
	results_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N, NULL, NULL);

	// Load data into the input buffer
	clEnqueueWriteBuffer(command_queue, inputMatrix1_mem_obj, CL_TRUE, 0,
	               sizeof(float) * N * N, inputMatrix1, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, inputMatrix2_mem_obj, CL_TRUE, 0,
	               sizeof(float) * N * N, inputMatrix2, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, results_mem_obj, CL_TRUE, 0,
	               sizeof(float) * N * N, results, 0, NULL, NULL);


	// Set the argument list for the kernel command
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputMatrix1_mem_obj);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputMatrix2_mem_obj);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &results_mem_obj);
	start = start_timer();
	// Enqueue the kernel command for execution
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global,
	                 local, 0, NULL, &prof_event);


	err = clWaitForEvents(1, &prof_event);

	clFinish(command_queue);

	// Copy the results from out of the output buffer
	clEnqueueReadBuffer(command_queue, results_mem_obj, CL_TRUE, 0,
	              sizeof(float) * N * N, results, 0, NULL, NULL);

	char name[] = "Run Time using Timers";
	end = stop_timer(start, name);
	err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, 
		sizeof(cl_ulong), &start_time, &return_bytes);
	err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &end_time, &return_bytes);
	run_time = (double)(end_time - start_time);

	printf("MULTIPLICATION: ");
	for(y = 0; y < width; y++)
	{
		for(x = 0; x < width; x++)
		{
			printf("%lf ", results[y * width + x]);
		}
		printf("\n");
	}

	//printf("Run Time: %llu\n", run_time);

	// Cleanup (release OpenCL resources)
	clReleaseContext(context);
	clReleaseCommandQueue(command_queue);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(inputMatrix1_mem_obj);
	clReleaseMemObject(inputMatrix2_mem_obj);
	clReleaseMemObject(results_mem_obj);

	return 0;

}