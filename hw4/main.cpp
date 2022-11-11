#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <OpenCL/cl.h>
#define N 40
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
			results[y * width + x] = data;
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

	comm = clCreateCommandQueue(context, device_id, 
		CL_QUEUE_PROFILING_ENABLE, &err);
	err = clEnqueueNDRangeKernel(comm, kernel, 1, NULL, 
		global, NULL, 0, NULL, &prof_event);

	clFinish(comm);

	err = clWaitForEvents(1, &prof_event);

	err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, 
		sizeof(cl_ulong), &start_time, &return_bytes);

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
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &results_mem_obj);

	// Enqueue the kernel command for execution
	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global,
	                 local, 0, NULL, NULL);
	clFinish(command_queue);
	
	err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &end_time, &return_bytes);

	// Copy the results from out of the output buffer
	clEnqueueReadBuffer(command_queue, results_mem_obj, CL_TRUE, 0,
	              sizeof(float) * N * N, results, 0, NULL, NULL);

	run_time = (double)(end_time - start_time);

	printf("Run Time: %llu\n", run_time);

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