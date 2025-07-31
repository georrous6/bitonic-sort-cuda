#include "bitonic_sort_v2.cuh"
#include "util.cuh"
#include "config.cuh"
#include "bitonic_sort_v0.cuh"


__global__ 
static void kernel_intra_block_sort_v2(int *data, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    extern __shared__ int s_data[];
    int offset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int max_size = blockDim.x > n ? n : blockDim.x;

    // Load data into shared memory
    s_data[tid] = data[offset + tid];

    __syncthreads();

    for (int size = 2; size <= max_size; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            int j = tid ^ step;
            if (j > tid) {
                int global_id = offset + tid;
                int is_asc = (global_id & size) == 0;
                compare_and_swap(s_data, tid, j, is_asc);
            }
            __syncthreads();
        }
    }

    // Copy data back to global memory
    data[offset + tid] = s_data[tid];
}


__global__
static void kernel_intra_block_refine_v2(int *data, int n, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    extern __shared__ int s_data[];
    int offset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int max_size = blockDim.x > n ? n : blockDim.x;

    // Load data into shared memory
    s_data[tid] = data[offset + tid];

    __syncthreads();

    for (int step = max_size >> 1; step > 0; step >>= 1) {
        int j  = tid ^ step;
        if (j > tid) {
            int global_id = offset + tid;
            int is_asc = (global_id & size) == 0;
            compare_and_swap(s_data, tid, j, is_asc);
        }
        __syncthreads();
    }

    // Copy data back to global memory
    data[offset + tid] = s_data[tid];
}


__host__
int bitonic_sort_v2(int *host_data, int n) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int max_size = BLOCK_SIZE > n ? n : BLOCK_SIZE;
    int max_step  = max_size >> 1;  // half block size
    size_t shared_mem_block_bytes = max_size * sizeof(int);

    int *device_data = NULL;
    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_intra_block_sort_v2<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n);
    if (post_launch_barrier_and_check()) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    // merge across blocks
    for (int size = max_size << 1; size <= n; size <<= 1) {
        
        for (int step = size >> 1; step > max_step; step >>= 1) {

            // Inter block merge
            kernel_compare_and_swap_v0<<<numBlocks, BLOCK_SIZE>>>(device_data, n, size, step);
            if (post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
        // intra-block refinement
        kernel_intra_block_refine_v2<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, size);
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
