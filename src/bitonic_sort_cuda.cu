#include "bitonic_sort_cuda.cuh"
#include "util.cuh"
#include "kernel.cuh"
#include <stdio.h>
#include <cuda_runtime.h>


__host__
int wakeup_cuda(void) {
    wakeup_kernel<<<1, 1>>>();
    return post_launch_barrier_and_check();
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
int bitonic_sort_cuda(int *data, int n, int ascending, kernel_version_t kernel_version) {

    if ((n & (n - 1)) != 0) {
        fprintf(stderr, "Error: Input size n=%d is not a power of 2.\n", n);
        return EXIT_FAILURE;
    }


    int status = EXIT_SUCCESS;
    // Kernel launch
    switch (kernel_version) {
        case KERNEL_NONE:
            bitonic_sort_serial(data, n, ascending);
            break;
        case KERNEL_V0:
            status = bitonic_sort_v0(data, n, ascending);
            break;
        case KERNEL_V1:
            status = bitonic_sort_v1(data, n, ascending);
            break;
        case KERNEL_V2:
            status = bitonic_sort_v2(data, n, ascending);
            break;
        default:
            fprintf(stderr, "Unsupported kernel version: %d\n", kernel_version);
            status = EXIT_FAILURE;
    }

    if (status) {
        fprintf(stderr, "Fallback to serial bitonic sort.\n");
        bitonic_sort_serial(data, n, ascending);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
