#include "bitonic_sort_v2.cuh"
#include "util.cuh"
#include "config.cuh"
#include "bitonic_sort_v0.cuh"


namespace {

__global__ 
void kernel_intra_block_sort(int *data, int n, int chunk_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    extern __shared__ int s_data[];
    int offset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    s_data[tid] = data[offset + tid];

    __syncthreads();

    int global_id = offset + tid;

    for (int size = 2; size <= chunk_size; size <<= 1) {
        int is_asc = (global_id & size) == 0;
        for (int step = size >> 1; step > 0; step >>= 1) {
            int j = tid ^ step;
            if (j > tid) {
                util::compare_and_swap(s_data, tid, j, is_asc);
            }
            __syncthreads();
        }
    }

    // Copy data back to global memory
    data[offset + tid] = s_data[tid];
}


__global__
void kernel_intra_block_refine(int *data, int n, int size, int chunk_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    extern __shared__ int s_data[];
    int offset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    s_data[tid] = data[offset + tid];

    __syncthreads();

    int global_id = offset + tid;
    int is_asc = (global_id & size) == 0;

    for (int step = chunk_size >> 1; step > 0; step >>= 1) {
        int j  = tid ^ step;
        if (j > tid) {
            util::compare_and_swap(s_data, tid, j, is_asc);
        }
        __syncthreads();
    }

    // Copy data back to global memory
    data[offset + tid] = s_data[tid];
}

}


namespace v2 {

__host__
int bitonic_sort(int *host_data, int n, int descending) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int chunk_size = BLOCK_SIZE > n ? n : BLOCK_SIZE;
    int max_step  = chunk_size >> 1;  // half block size
    size_t shared_mem_block_bytes = chunk_size * sizeof(int);

    int *device_data = NULL;
    if (util::host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_intra_block_sort<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, chunk_size);
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
        kernel_intra_block_refine<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, size, chunk_size);
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
