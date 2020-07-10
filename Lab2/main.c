#define PART_SIZE 1024 // It depends on your openCL device. More -- better
#define N 1024*1024
#define FILE_NAME "prefixSum.cl"
#define FUNCTION_NAME "prefixSum"

#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <time.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <math.h>

int i, j;

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

int verifyResult(const float *array, const float *result, int n) {
	float tmp = 0.0f;
	for (i = 0; i < n; ++i) {
		tmp += array[i];
		if (tmp != result[i]) {
			return -1;
		}
	}
	return 0;
}

cl_kernel createKernel(cl_context *cl_context_value, cl_command_queue *queue, cl_device_id device) {
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

	sprintf(options, "-D PART_SIZE=%d", PART_SIZE);

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

bool printInformation() {
	char *value;
	size_t valueSize;
	cl_uint platformCount;
	cl_platform_id *platforms;
	cl_uint deviceCount;
	cl_device_id *devices;
	cl_uint maxComputeUnits;
	cl_uint maxClockFrequency;

	cl_int num_platform = clGetPlatformIDs(0, NULL, &platformCount);
	if (num_platform != CL_SUCCESS) {
		printf("Error: Platform num code: %d", num_platform);
		return false;
	}

	printf("Platforms num: %d\n", platformCount);

	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	for (i = 0; i < platformCount; i++) {
		printf("%d. PlatformId: %d\n", i + 1, (int)platforms[i]);

		cl_int num_device = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
		if (num_device != CL_SUCCESS) {
			printf("Error: Device num code: %d", num_device);
			return false;
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
		}
		free(devices);
	}
	free(platforms);
	return true;
}

cl_device_id getSpecificDevice(cl_platform_id *platform_ids, cl_uint platform_count,
                               cl_device_type cl_device, cl_bool discrete, cl_bool *success) {
	cl_int result;
	cl_device_id resultDevice;

	for (i = 0; i < platform_count && !(*success); ++i) {
		cl_platform_id current = platform_ids[i];

		cl_uint device_count = 0;
		result = clGetDeviceIDs(current, cl_device, 0, NULL, &device_count);
		if (result != CL_SUCCESS || device_count == 0) {
			continue;
		}

		cl_device_id *devices = (cl_device_id *)malloc(device_count * sizeof(cl_device_id));
		result = clGetDeviceIDs(current, cl_device, device_count, devices, NULL);

		if (result == CL_SUCCESS) {
			for (j = 0; j < device_count && !(*success); ++j) {
				cl_device_id current_device = devices[i];

				if (discrete) {
					cl_bool memoryInfo;
					clGetDeviceInfo(current_device, CL_DEVICE_HOST_UNIFIED_MEMORY,
					                sizeof(cl_bool), &memoryInfo, NULL);

					if (memoryInfo == CL_FALSE) {
						printf("Found suitable device\n");
						*success = true;
						resultDevice = current_device;
					} else {
						continue;
					}
				} else {
					*success = true;
					resultDevice = current_device;
				}
			}
		}
		free(devices);
	}
	if (*success) {
		size_t nameLen = 0;
		clGetDeviceInfo(resultDevice, CL_DEVICE_NAME, 0, NULL, &nameLen);
		char *deviceName = (char *)malloc(nameLen * sizeof(char));
		clGetDeviceInfo(resultDevice, CL_DEVICE_NAME, nameLen, deviceName, NULL);
		printf("Found suitable device. Device name: %s\n", deviceName);
		free(deviceName);
	}
	return resultDevice;
}

cl_device_id getDevice() {
	cl_uint platforms_count = 0;
	cl_uint result = clGetPlatformIDs(0, NULL, &platforms_count);
	cl_device_id device_result;
	if (result != CL_SUCCESS) {
		printf("An error occurred while getting devices count");
		return (cl_device_id)-1;
	}

	cl_platform_id *platforms = (cl_platform_id *)malloc(platforms_count * sizeof(cl_platform_id));
	result = clGetPlatformIDs(platforms_count, platforms, NULL);
	if (result != CL_SUCCESS) {
		printf("An error occurred while getting device ids");
		return (cl_device_id)-1;
	}

	cl_bool device_found = CL_FALSE;
	// Try to find a discrete GPU
	device_result = getSpecificDevice(platforms, platforms_count, CL_DEVICE_TYPE_GPU, CL_TRUE, &device_found);
	if (!device_found) {
		// Try to find an integrated GPU
		device_result = getSpecificDevice(platforms, platforms_count, CL_DEVICE_TYPE_GPU, CL_FALSE, &device_found);
	}
	if (!device_found) {
		// Try to find a CPU
		device_result = getSpecificDevice(platforms, platforms_count, CL_DEVICE_TYPE_CPU, CL_FALSE, device_found);
	}
	if (!device_found) {
		printf("No suitable devices found");
		return (cl_device_id)-1;
	}

	return device_result;
}

int compute(cl_kernel kernel,
            cl_command_queue queue,
            cl_context cl_context_value,
            int n,
            cl_float *array,
            cl_float *result,
            cl_long *start,
            cl_long *end) {
	cl_int a_error;
	cl_int b_error;
	cl_mem mem_a = clCreateBuffer(cl_context_value, CL_MEM_READ_ONLY, sizeof(cl_float) * n, NULL, &a_error);
	cl_mem mem_b = clCreateBuffer(cl_context_value, CL_MEM_WRITE_ONLY, sizeof(cl_float) * n, NULL, &b_error);

	if (a_error != CL_SUCCESS || b_error != CL_SUCCESS) {
		printf("Error: can'n create buffers: %d, %d", a_error, b_error);
		return -1;
	}

	printf("Buffers created\n");

	a_error = clEnqueueWriteBuffer(queue, mem_a, CL_FALSE, 0, sizeof(cl_float) * n, array, 0, NULL, NULL);
	if (a_error != CL_SUCCESS) {
		printf("Error: can'n write in the buffers: %d", a_error);
		return -1;
	}

	cl_int mem_a_error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_a);
	cl_int mem_b_error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_b);
	cl_int fst_arg_error = clSetKernelArg(kernel, 2, sizeof(cl_uint), &n);

	if (mem_a_error != CL_SUCCESS || mem_b_error != CL_SUCCESS
			|| fst_arg_error != CL_SUCCESS) {
		printf("Error: can'n set kernel args: %d, %d, %d",
		       mem_a_error,
		       mem_b_error,
		       fst_arg_error);
		return -1;
	}

	size_t *global = (size_t *)malloc(1 * sizeof(size_t));
	size_t *local = (size_t *)malloc(1 * sizeof(size_t));
	global[0] = n;
	local[0] = PART_SIZE;

	cl_event event;

	cl_int range_error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, global, local, 0, NULL, &event);
	if (range_error != CL_SUCCESS) {
		printf("Error: can't enqueue range: %d", range_error);
		return -1;
	}

	b_error = clEnqueueReadBuffer(queue, mem_b, CL_TRUE, 0, sizeof(float) * n, result, 0, NULL, NULL);
	if (b_error != CL_SUCCESS) {
		printf("Error: Can't enqueue read buffer: %d", b_error);
		return -1;
	}

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), end, 0);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), start, 0);

	clReleaseMemObject(mem_a);
	clReleaseMemObject(mem_b);
}

int main() {
	time_t t;
	srand((unsigned)time(&t));

	// Print information about devices
	if (!printInformation()) {
		return -1;
	}

	printf("\n");

	// Get device
	cl_device_id device = getDevice();
	if (device == (cl_device_id)-1) {
		return -1;
	}

	cl_context cl_context_value;
	cl_command_queue queue;
	cl_kernel kernel = createKernel(&cl_context_value, &queue, device);

	if (kernel == -1)
		return -1;

	size_t n = N;
	if (n < PART_SIZE) {
		n = PART_SIZE;
	}

	if ((n & (n - 1)) != 0) {
		n = pow(2, ceil(log(n) / log(2)));
	}

	cl_float *array = (cl_float *)malloc(n * sizeof(cl_float));
	cl_float *result = (cl_float *)malloc(n * sizeof(cl_float));

	for (i = 0; i < N; ++i) {
		array[i] = rand() % 100 + 1;
	}
	for (i = N; i < n; ++i) {
		array[i] = 0;
	}

	for (i = 0; i < n; ++i) {
		result[i] = 0;
	}

	cl_long end;
	cl_long start;

	int computation_error = compute(kernel, queue, cl_context_value, n, array, result, &start, &end);
	if (computation_error == -1) {
		return -1;
	}

	printf("\n");
	cl_ulong time = (end - start);
	printf("Global kernel time: %f(ms)\n", time * 1.0e-6f);

	int verification = verifyResult(array, result, n);
	if (verification) {
		printf("Result: SUCCESS\n");
	} else {
		printf("Something gone wrong");
		return -1;
	}

	free(array);
	free(result);

	clReleaseContext(cl_context_value);
	clReleaseCommandQueue(queue);
	clReleaseKernel(kernel);

	return 0;
}
