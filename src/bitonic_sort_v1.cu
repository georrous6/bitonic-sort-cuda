#include "bitonic_sort_v1.cuh"
#include "util.cuh"
#include "config.cuh"
#include "bitonic_sort_v0.cuh"


__global__ 
static void kernel_intra_block_sort_v1(int *data, int n, int ascending) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int max_size = blockDim.x > n ? n : blockDim.x;

    for (int size = 2; size <= max_size; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            int j  = i ^ step;
            if (j > i) {
                int is_asc = ((i & size) == 0) ? ascending : !ascending;
                compare_and_swap(data, i, j, is_asc);
            }
            __syncthreads();
        }
    }
}


__global__ 
static void kernel_intra_block_refine_v1(int *data, int n, int ascending, int size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int max_size = blockDim.x > n ? n : blockDim.x;

    for (int step = max_size >> 1; step > 0; step >>= 1) {
            int j  = i ^ step;
            if (j > i) {
                int is_asc = ((i & size) == 0) ? ascending : !ascending;
                compare_and_swap(data, i, j, is_asc);
            }
        __syncthreads();
    }
}


__host__
int bitonic_sort_v1(int *host_data, int n, int ascending) {

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int max_size = BLOCK_SIZE > n ? n : BLOCK_SIZE;
    int max_step = max_size >> 1;  // half block size

    int *device_data = NULL;
    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_intra_block_sort_v1<<<numBlocks, BLOCK_SIZE>>>(device_data, n, ascending);
    if (post_launch_barrier_and_check()) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    // merge across blocks
    for (int size = max_size << 1; size <= n; size <<= 1) {

        for (int step = size >> 1; step > max_step; step >>= 1) {

            // Inter block merge
            kernel_compare_and_swap_v0<<<numBlocks, BLOCK_SIZE>>>(device_data, n, ascending, size, step);
            if (post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
        // intra-block refinement
        kernel_intra_block_refine_v1<<<numBlocks, BLOCK_SIZE>>>(device_data, n, ascending, size);
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

