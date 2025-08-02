#include "bitonic_sort_v1.cuh"
#include "util.cuh"
#include "config.cuh"
#include "bitonic_sort_v0.cuh"

namespace {

__global__ 
void kernel_intra_block_sort(int *data, int n, int chunk_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    for (int size = 2; size <= chunk_size; size <<= 1) {
        int is_asc = (i & size) == 0;
        for (int step = size >> 1; step > 0; step >>= 1) {
            int j  = i ^ step;
            if (j > i) {
                util::compare_and_swap(data, i, j, is_asc);
            }
            __syncthreads();
        }
    }
}


__global__ 
void kernel_intra_block_refine(int *data, int n, int size, int chunk_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int is_asc = (i & size) == 0;
    for (int step = chunk_size >> 1; step > 0; step >>= 1) {
            int j  = i ^ step;
            if (j > i) {
                util::compare_and_swap(data, i, j, is_asc);
            }
        __syncthreads();
    }
}

}

namespace v1 {

__host__
int bitonic_sort(int *host_data, int n, int descending) {

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int chunk_size = BLOCK_SIZE > n ? n : BLOCK_SIZE;
    int max_step = chunk_size >> 1;  // half chunk size

    int *device_data = NULL;
    if (util::host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_intra_block_sort<<<numBlocks, BLOCK_SIZE>>>(device_data, n, chunk_size);
    if (util::post_launch_barrier_and_check()) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    // merge across blocks
    for (int size = chunk_size << 1; size <= n; size <<= 1) {

        for (int step = size >> 1; step > max_step; step >>= 1) {

            // Inter block merge
            v0::kernel_compare_and_swap<<<numBlocks, BLOCK_SIZE>>>(device_data, n, size, step);
            if (util::post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
        // intra-block refinement
        kernel_intra_block_refine<<<numBlocks, BLOCK_SIZE>>>(device_data, n, size, chunk_size);
        if (util::post_launch_barrier_and_check()) {
            cudaFree(device_data);
            return EXIT_FAILURE;
        }
    }

    // If descending order is requested, reverse the data
    if (descending) {
        util::kernel_reverse<<<numBlocks, BLOCK_SIZE>>>(device_data, n);
        if (util::post_launch_barrier_and_check()) {
            cudaFree(device_data);
            return EXIT_FAILURE;
        }
    }

    if (util::device_to_host_data(host_data, n, device_data) != EXIT_SUCCESS) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    cudaFree(device_data);
    return EXIT_SUCCESS;
}

}

