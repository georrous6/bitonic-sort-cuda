#include "bitonic_sort_cuda.cuh"
#include "util.cuh"
#include "kernel.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "config.cuh"


__host__
int wakeup_cuda(void) {
    wakeup_kernel<<<1, 1>>>();
    return post_launch_barrier_and_check();
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


static void sort_serial(int *data, int n, int ascending) {
    if (ascending) {
        qsort(data, n, sizeof(int), compare_asc);
    } else {
        qsort(data, n, sizeof(int), compare_desc);
    }
}


__host__
static int bitonic_sort_v0(int *host_data, int n, int ascending) {

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int *device_data = NULL;

    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    for (int size = 2; size <= n; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {

            // Launch the kernel
            kernel_v0_compare_and_swap<<<numBlocks, BLOCK_SIZE>>>(device_data, n, ascending, size, step);

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
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int chunk_size   = n / numBlocks;
    int max_step  = chunk_size >> 1;  // half block size

    printf("size: %d, number of blocks: %d, chunk size: %d, threads per block: %d\n", n, numBlocks, chunk_size, BLOCK_SIZE);
    fflush(stdout);

    int *device_data = NULL;
    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_v1_intra_block_sort<<<numBlocks, BLOCK_SIZE>>>(device_data, n, chunk_size, ascending);
    if (post_launch_barrier_and_check()) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    // merge across blocks
    for (int size = chunk_size << 1; size <= n; size <<= 1) {
        
        for (int step = size >> 1; step > max_step; step >>= 1) {

            // Inter block merge
            kernel_v0_compare_and_swap<<<numBlocks, BLOCK_SIZE>>>(device_data, n, ascending, size, step);
            if (post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
        // intra-block refinement
        kernel_v1_intra_block_refine<<<numBlocks, BLOCK_SIZE>>>(device_data, n, chunk_size, ascending, size);
        if (post_launch_barrier_and_check()) {
            cudaFree(device_data);
            return EXIT_FAILURE;
        }
    }

    if (device_to_host_data(host_data, n, device_data) != EXIT_SUCCESS) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    cudaFree(device_data);
    return EXIT_SUCCESS;
}


__host__
static int bitonic_sort_v2(int *host_data, int n, int ascending) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int chunk_size   = n / numBlocks;
    int max_step  = chunk_size >> 1;  // half block size
    size_t shared_mem_block_bytes = chunk_size * sizeof(int);

    printf("size: %d, number of blocks: %d, chunk size: %d, threads per block: %d\n", n, numBlocks, chunk_size, BLOCK_SIZE);
    fflush(stdout);

    int *device_data = NULL;
    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_v2_intra_block_sort<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, chunk_size, ascending);
    if (post_launch_barrier_and_check()) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    // merge across blocks
    for (int size = chunk_size << 1; size <= n; size <<= 1) {
        
        for (int step = size >> 1; step > max_step; step >>= 1) {

            // Inter block merge
            kernel_v0_compare_and_swap<<<numBlocks, BLOCK_SIZE>>>(device_data, n, ascending, size, step);
            if (post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
        // intra-block refinement
        kernel_v2_intra_block_refine<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, chunk_size, ascending, size);
        if (post_launch_barrier_and_check()) {
            cudaFree(device_data);
            return EXIT_FAILURE;
        }
    }

    if (device_to_host_data(host_data, n, device_data) != EXIT_SUCCESS) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    cudaFree(device_data);
    return EXIT_SUCCESS;
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
            sort_serial(data, n, ascending);
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
        sort_serial(data, n, ascending);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
