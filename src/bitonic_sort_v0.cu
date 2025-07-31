#include "bitonic_sort_v0.cuh"
#include "util.cuh"
#include "config.cuh"


__global__ 
void kernel_compare_and_swap_v0(int *data, int n, int size, int step) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int j = i ^ step;
    if (j > i) {
        int is_asc = (i & size) == 0;
        compare_and_swap(data, i, j, is_asc);
    }
}


__host__
int bitonic_sort_v0(int *host_data, int n) {

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int *device_data = NULL;

    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    for (int size = 2; size <= n; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {

            // Launch the kernel
            kernel_compare_and_swap_v0<<<numBlocks, BLOCK_SIZE>>>(device_data, n, size, step);

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
