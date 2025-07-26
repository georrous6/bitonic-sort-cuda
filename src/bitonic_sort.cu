#include "bitonic_sort.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

// Swap elements if needed, based on direction
__host__ __device__ __forceinline__
static void compare_and_swap(int *arr, int i, int j, int ascending) {
    if ((ascending && arr[i] > arr[j]) || (!ascending && arr[i] < arr[j])) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}


__host__
static void bitonic_sort_serial(int *arr, int n, int ascending) {
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = 0; i < n; i++) {
                int j = i ^ stride;
                if (j > i) {
                    int is_ascending = ((i & size) == 0) ? ascending : !ascending;
                    compare_and_swap(arr, i, j, is_ascending);
                }
            }
        }
    }
}


__global__ 
void kernel_v0(int *data, int n, int ascending, int size, int step) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        int j = i ^ step;
        if (j > i) {
            int is_ascending = ((i & size) == 0) ? ascending : !ascending;
            compare_and_swap(data, i, j, is_ascending);
        }
    }
}


__host__
static int host_to_device_data(int *host_data, int n, int **device_data) {
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void **)device_data, n * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    err = cudaMemcpy(*device_data, host_data, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data to device: %s\n", cudaGetErrorString(err));
        cudaFree(*device_data);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


__host__
static int device_to_host_data(int *host_data, int n, int *device_data) {
    cudaError_t err;

    // Copy data from device to host
    err = cudaMemcpy(host_data, device_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data back to host: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


__host__
static int post_launch_barrier_and_check(void) {
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


__host__
static int bitonic_sort_v0(int *host_data, int n, int ascending) {

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    int *device_data = NULL;

    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    for (int size = 2; size <= n; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {

            // Launch the kernel
            kernel_v0<<<numBlocks, threadsPerBlock>>>(device_data, n, ascending, size, step);

            if (post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
    }

    // Copy the sorted data back to host
    if (device_to_host_data(host_data, n, device_data) != EXIT_SUCCESS) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    cudaFree(device_data);
    return EXIT_SUCCESS;
}


__host__
static int bitonic_sort_v1(int *host_data, int n, int ascending) {
    // Placeholder for future kernel version 1 implementation
    fprintf(stderr, "Kernel version 1 is not implemented yet.\n");
    return EXIT_FAILURE;
}


__host__
static int bitonic_sort_v2(int *host_data, int n, int ascending) {
    // Placeholder for future kernel version 2 implementation
    fprintf(stderr, "Kernel version 2 is not implemented yet.\n");
    return EXIT_FAILURE;
}


__host__
int bitonic_sort(int *data, int n, int ascending, kernel_version_t kernel_version) {

    if ((n & (n - 1)) != 0) {
        fprintf(stderr, "Error: Input size n=%d is not a power of 2.\n", n);
        return EXIT_FAILURE;
    }


    int status = EXIT_SUCCESS;
    // Kernel launch
    switch (kernel_version) {
        case KERNEL_V0:
            status = bitonic_sort_v0(data, n, ascending);
            break;
        case KERNEL_V1:
            status = bitonic_sort_v1(data, n, ascending);
            break;
        case KERNEL_V2:
            status = bitonic_sort_v2(data, n, ascending);
            break;
        case KERNEL_NONE:
            bitonic_sort_serial(data, n, ascending);
            break;
        default:
            fprintf(stderr, "Unsupported kernel version: %d\n", kernel_version);
            status = EXIT_FAILURE;
    }

    if (status) {
        fprintf(stderr, "Fallback to serial bitonic sort.\n");
        bitonic_sort_serial(data, n, ascending);
    }

    return EXIT_SUCCESS;
}
