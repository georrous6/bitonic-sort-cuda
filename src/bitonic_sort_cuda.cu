#include "bitonic_sort_cuda.cuh"
#include "util.cuh"
#include "config.cuh"
#include "bitonic_sort_v0.cuh"
#include "bitonic_sort_v1.cuh"
#include "bitonic_sort_v2.cuh"
#include "bitonic_sort_v3.cuh"
#include "bitonic_sort_v4.cuh"
#include "bitonic_sort_v5.cuh"
#include <stdio.h>
#include <stdlib.h>


__global__
static void kernel_wakeup(void) {}


__host__
int wakeup_cuda(void) {
    kernel_wakeup<<<1, 1>>>();
    return util::post_launch_barrier_and_check();
}


__host__
int print_cuda_device_info(void) {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Found %d CUDA device(s)\n", device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("\nDevice %d: %s\n", dev, prop.name);
        printf("  Compute Capability:          %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory:         %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("  Shared Memory per Block:     %lu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Registers per Block:         %d\n", prop.regsPerBlock);
        printf("  Warp Size:                   %d\n", prop.warpSize);
        printf("  Max Threads per Block:       %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads Dim:             [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Size:               [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Clock Rate:                  %.2f MHz\n", prop.clockRate / 1000.0);
        printf("  Multi-Processor Count:       %d\n", prop.multiProcessorCount);
    }

    return EXIT_SUCCESS;
}


// // Just a serial implementation of bitonic sort for starting point
// __host__
// static void bitonic_sort_serial(int *data, int n, int ascending) {
//     for (int size = 2; size <= n; size <<= 1) {
//         for (int step = size >> 1; step > 0; step >>= 1) {
//             for (int i = 0; i < n; i++) {
//                 int j = i ^ step;
//                 if (j > i) {
//                     int is_ascending = ((i & size) == 0) ? ascending : !ascending;
//                     compare_and_swap(data, i, j, is_ascending);
//                 }
//             }
//         }
//     }
// }


static int compare_asc(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}


static int compare_desc(const void *a, const void *b) {
    return (*(int *)b - *(int *)a);
}


static void sort_serial(int *data, int n, int descending) {
    if (descending) {
        qsort(data, n, sizeof(int), compare_desc);
    } else {
        qsort(data, n, sizeof(int), compare_asc);
    }
}


__host__
int bitonic_sort_cuda(int *data, int n, int descending, bitonic_version_t kernel_version) {

    if ((n & (n - 1)) != 0 || n < 2) {
        fprintf(stderr, "Error: Input size n=%d is not a power of 2 or less than 2.\n", n);
        return EXIT_FAILURE;
    }


    int status = EXIT_SUCCESS;
    // Kernel launch
    switch (kernel_version) {
        case VERSION_SERIAL:
            sort_serial(data, n, descending);
            break;
        case VERSION_V0:
            status = v0::bitonic_sort(data, n, descending);
            break;
        case VERSION_V1:
            status = v1::bitonic_sort(data, n, descending);
            break;
        case VERSION_V2:
            status = v2::bitonic_sort(data, n, descending);
            break;
        case VERSION_V3:
            status = v3::bitonic_sort(data, n, descending);
            break;
        case VERSION_V4:
            status = v4::bitonic_sort(data, n, descending);
            break;
        case VERSION_V5:
            status = v5::bitonic_sort(data, n, descending);
            break;
        default:
            fprintf(stderr, "Unsupported kernel version: %d\n", kernel_version);
            status = EXIT_FAILURE;
    }

    if (status) {
        fprintf(stderr, "Falling back to serial sort\n");
        sort_serial(data, n, descending);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
