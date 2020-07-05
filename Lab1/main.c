#define CL_TARGET_OPENCL_VERSION 220

#define FILE_NAME "matrices_mul_local"
#define FUNCTION_NAME "mul"
#define N 2048
#define L 512
#define M 1024
#define TILESIZE 32
#define WPT 16

#include <stdio.h>
#include <CL/opencl.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

int i, j;

// read FILE_NAME and returns content in char*
char *read_program(size_t *lSize) {
	FILE *fp;
	char *buffer;

	fp = fopen(FILE_NAME, "rb");

	fseek(fp, 0L, SEEK_END);
	*lSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	buffer = malloc(*lSize);
	fread(buffer, 1, *lSize, fp);
	fclose(fp);
	printf("Successfully read the program\n");
	return buffer;
}

// Naive implementation for verification
int verify_result(const cl_float *first,
                  const cl_float *second,
                  const cl_float *result,
                  cl_uint n,
                  cl_uint l,
                  cl_uint m) {
	cl_uint *matrix = (cl_uint *)malloc(sizeof(cl_float) * n * m);

	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j) {
			matrix[i * m + j] = 0;
			for (int k = 0; k < l; ++k) {
				matrix[i * m + j] += first[i * l + k] * second[m * k + j];
			}
			if (matrix[i * m + j] != result[i * m + j]) {
				printf("Error at [%d][%d]. Expected value: %u. Found: %f\n", i, j, matrix[i * m + j], result[i * m + j]);
				return 0;
			}
		}
	}
	return 1;
}

// print information about clDevices
// return the most "powerful"
// "powerful" means with the biggest maxComputeUnits * maxClockFrequency
cl_device_id openClDevicesInfo() {
	char *value;
	size_t valueSize;
	cl_uint platformCount;
	cl_platform_id *platforms;
	cl_uint deviceCount;
	cl_device_id *devices;
	cl_uint maxComputeUnits;
	cl_uint maxClockFrequency;
	int max_device_rate = 0;
	cl_device_id result;

	cl_int num_platform = clGetPlatformIDs(0, NULL, &platformCount);
	if (num_platform != CL_SUCCESS) {
		printf("Error: Platform num code: %d", num_platform);
		return (cl_device_id)-1;
	}

	printf("Platforms num: %d\n", platformCount);

	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	for (i = 0; i < platformCount; i++) {
		printf("%d. PlatformId: %d\n", i + 1, (int)platforms[i]);

		cl_int num_device = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
		if (num_device != CL_SUCCESS) {
			printf("Error: Device num code: %d", num_device);
			return (cl_device_id)-1;
		}

		devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

		for (j = 0; j < deviceCount; j++) {
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
			value = (char *)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
			printf(" %d.%d. Device: %s\n", i + 1, j + 1, value);
			free(value);

			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
			value = (char *)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
			printf("  %d.%d.%d Hardware version: %s\n", i + 1, j + 1, 1, value);
			free(value);

			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
			value = (char *)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
			printf("  %d.%d.%d Software version: %s\n", i + 1, j + 1, 2, value);
			free(value);

			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
			value = (char *)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
			printf("  %d.%d.%d OpenCL C version: %s\n", i + 1, j + 1, 3, value);
			free(value);

			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
			printf("  %d.%d.%d Parallel compute units: %d\n", i + 1, j + 1, 4, maxComputeUnits);

			clGetDeviceInfo(devices[j],
			                CL_DEVICE_MAX_CLOCK_FREQUENCY,
			                sizeof(maxClockFrequency),
			                &maxClockFrequency,
			                NULL);
			printf("  %d.%d.%d Max clock frequency: %d\n", i + 1, j + 1, 5, maxClockFrequency);

			if (maxComputeUnits * maxClockFrequency > max_device_rate) {
				result = devices[j];
			}
		}
		free(devices);
	}
	free(platforms);

	return result;
}

// return kernel
cl_kernel createKernel(cl_context *cl_context_value, cl_command_queue *queue) {
	cl_device_id device = openClDevicesInfo();
	if ((int)device == -1) {
		return (cl_kernel)-1;
	}
	printf("\n");

	cl_int cl_context_error;
	*cl_context_value = clCreateContext(NULL, 1, &device, NULL, NULL, &cl_context_error);
	if (*cl_context_value == NULL || cl_context_error != CL_SUCCESS) {
		printf("Error: can't create context: %d", cl_context_error);
		return (cl_kernel)-1;
	}

	cl_int cl_queue_error;
	*queue =
			clCreateCommandQueue(*cl_context_value, device, CL_QUEUE_PROFILING_ENABLE, &cl_queue_error);
	if (*queue == NULL || cl_queue_error != CL_SUCCESS) {
		printf("Error: can't create queue: %d\n", cl_queue_error);
		return (cl_kernel)-1;
	}

	size_t lSize;
	char *buffer = read_program(&lSize);

	cl_int program_error;
	cl_program program = clCreateProgramWithSource(*cl_context_value, 1, &buffer, &lSize, &program_error);

	if (program_error != CL_SUCCESS) {
		printf("Error: can't create program: %d", program_error);
		return (cl_kernel)-1;
	}

	printf("Program successfully created\n");

	char *options = (char *)malloc(64 * sizeof(char));

	sprintf(options, "-D TILE_SIZE=%d -D WORK_PER_THREAD=%d", TILESIZE, WPT);

	cl_int err = clBuildProgram(program, 1, &device, options, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Build error");
		size_t length;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
		char *log_buffer = calloc(length, sizeof(char));
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log_buffer, &length);
		printf("%s", log_buffer);
	} else {
		printf("Build finished\n");
	}

	cl_int kernel_error;
	cl_kernel kernel = clCreateKernel(program, FUNCTION_NAME, &kernel_error);
	if (kernel_error != CL_SUCCESS) {
		printf("Error: can't create a kernel: %d", program_error);
		return (cl_kernel)-1;
	}

	printf("Kernel successfully created\n");
	return kernel;
}

// multiplication matrix, return -1 if something gone wrong
int compute(cl_kernel kernel,
            cl_command_queue queue,
            cl_context cl_context_value,
            cl_uint n,
            cl_uint l,
            cl_uint m,
            cl_float *first_matrix,
            cl_float *second_matrix,
            cl_float *result,
            cl_long *start,
            cl_long *end) {

	cl_int a_error;
	cl_int b_error;
	cl_int c_error;
	cl_mem mem_a = clCreateBuffer(cl_context_value, CL_MEM_READ_ONLY, sizeof(cl_float) * n * l, NULL, &a_error);
	cl_mem mem_b = clCreateBuffer(cl_context_value, CL_MEM_READ_ONLY, sizeof(cl_float) * l * m, NULL, &b_error);
	cl_mem mem_c = clCreateBuffer(cl_context_value, CL_MEM_WRITE_ONLY, sizeof(cl_float) * n * m, NULL, &c_error);
	if (a_error != CL_SUCCESS || b_error != CL_SUCCESS || c_error != CL_SUCCESS) {
		printf("Error: can'n create buffers: %d, %d, %d", a_error, b_error, c_error);
		return -1;
	}
	printf("Buffers created\n");

	a_error = clEnqueueWriteBuffer(queue, mem_a, CL_FALSE, 0, sizeof(cl_float) * n * l, first_matrix, 0, NULL, NULL);
	b_error = clEnqueueWriteBuffer(queue, mem_b, CL_FALSE, 0, sizeof(cl_float) * m * l, second_matrix, 0, NULL, NULL);
	if (a_error != CL_SUCCESS || b_error != CL_SUCCESS) {
		printf("Error: can'n write in the buffers: %d, %d", a_error, b_error);
		return -1;
	}

	cl_int mem_a_error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_a);
	cl_int mem_b_error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_b);
	cl_int mem_c_error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_c);
	cl_int fst_arg_error = clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
	cl_int snd_arg_error = clSetKernelArg(kernel, 4, sizeof(cl_uint), &l);
	cl_int third_arg_error = clSetKernelArg(kernel, 5, sizeof(cl_uint), &m);

	if (mem_a_error != CL_SUCCESS || mem_b_error != CL_SUCCESS
			|| mem_c_error != CL_SUCCESS || fst_arg_error != CL_SUCCESS || snd_arg_error != CL_SUCCESS
			|| third_arg_error != CL_SUCCESS) {
		printf("Error: can'n set kernel args: %d, %d. %d, %d, %d",
		       mem_a_error,
		       mem_b_error,
		       mem_c_error,
		       fst_arg_error,
		       snd_arg_error);
		return -1;
	}

	size_t *global = (size_t *)malloc(2 * sizeof(size_t));
	size_t *local = (size_t *)malloc(2 * sizeof(size_t));
	global[0] = n / WPT;
	global[1] = m;
	local[0] = TILESIZE / WPT;
	local[1] = TILESIZE;

	cl_event event;

	cl_int range_error = clEnqueueNDRangeKernel(queue, kernel, 2, 0, global, local, 0, NULL, &event);
	if (range_error != CL_SUCCESS) {
		printf("Error: can't enqueue range: %d", range_error);
		return -1;
	}

	c_error = clEnqueueReadBuffer(queue, mem_c, CL_TRUE, 0, sizeof(float) * n * m, result, 0, NULL, NULL);
	if (c_error != CL_SUCCESS) {
		printf("Error: Can't enqueue read buffer: %d", c_error);
		return -1;
	}

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), end, 0);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), start, 0);

	clReleaseMemObject(mem_a);
	clReleaseMemObject(mem_b);
	clReleaseMemObject(mem_c);
}

int main() {
	cl_uint n = N;
	cl_uint l = L;
	cl_uint m = M;

	time_t t;
	srand((unsigned)time(&t));

	cl_context cl_context_value;
	cl_command_queue queue;
	cl_kernel kernel = createKernel(&cl_context_value, &queue);

	if (kernel == -1)
		return -1;

	cl_float *first_matrix = (cl_float *)malloc(n * l * sizeof(cl_float));
	cl_float *second_matrix = (cl_float *)malloc(l * m * sizeof(cl_float));
	cl_float *result = (cl_float *)malloc(n * m * sizeof(cl_float));

	for (i = 0; i < n; ++i) {
		for (j = 0; j < l; ++j) {
			first_matrix[i * l + j] = rand() % 100 + 1;
		}
	}

	for (i = 0; i < l; ++i) {
		for (j = 0; j < m; ++j) {
			second_matrix[i * m + j] = rand() % 100 + 1;
		}
	}

	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j) {
			result[i * m + j] = 0;
		}
	}
	cl_long end;
	cl_long start;

	int computation_error =
			compute(kernel, queue, cl_context_value, n, l, m, first_matrix, second_matrix, result, &start, &end);
	if (computation_error == -1) {
		return -1;
	}

	printf("\n");

	cl_ulong time = (end - start);
	printf("Global kernel time: %f(ms)", time * 1.0e-6f);
	cl_ulong operations = 2 * n * m * l;
	cl_ulong gflops = operations / time;
	printf("\nGFlops: %llu\n", gflops);
	printf("Verification...\n");
	int verification = verify_result(first_matrix, second_matrix, result, n, l, m);

	if (verification) {
		printf("Result: SUCCESS\n");
	} else {
		printf("Something gone wrong");
		return -1;
	}

	free(first_matrix);
	free(second_matrix);
	free(result);

	clReleaseContext(cl_context_value);
	clReleaseCommandQueue(queue);

	clReleaseKernel(kernel);

	return 0;
}