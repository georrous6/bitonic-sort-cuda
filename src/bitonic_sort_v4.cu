#include "bitonic_sort_v4.cuh"
#include "util.cuh"
#include "config.cuh"
#include "bitonic_sort_v0.cuh"


__global__ 
static void kernel_intra_block_sort_v4(int *data, int n, int max_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    extern __shared__ int s_data[];
    int offset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    s_data[tid] = data[offset + tid];

    __syncthreads();

    int global_id = offset + tid;

    for (int size = 2; size <= max_size; size <<= 1) {
        int is_asc = (global_id & size) == 0;
        int step = size >> 1;
        for (; step >= WARP_SIZE; step >>= 1) {
            int j = tid ^ step;
            if (j > tid) {
                compare_and_swap(s_data, tid, j, is_asc);
            }
            __syncthreads();
        }

        // Wrap level sorting using warp shuffle
        int my_val = s_data[tid];
        for (; step > 0; step >>= 1) {
            int j = tid ^ step;
            int partner_val = __shfl_xor_sync(0xFFFFFFFF, my_val, step);
            int is_asc_order = (tid < j) == (my_val <= partner_val);
            int should_swap = is_asc != is_asc_order;
            my_val = should_swap ? partner_val : my_val;
        }

        s_data[tid] = my_val;
        __syncthreads();
    }

    // Copy data back to global memory
    data[offset + tid] = s_data[tid];
}


__global__
static void kernel_intra_block_refine_v4(int *data, int n, int size, int max_size) {

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
    int step = max_size >> 1;
    for (; step >= WARP_SIZE; step >>= 1) {
        int j  = tid ^ step;
        if (j > tid) {
            compare_and_swap(s_data, tid, j, is_asc);
        }
        __syncthreads();
    }

    // Wrap level sorting using warp shuffle
    int my_val = s_data[tid];
    for (; step > 0; step >>= 1) {
        int j = tid ^ step;
        int partner_val = __shfl_xor_sync(0xFFFFFFFF, my_val, step);
        int is_asc_order = (tid < j) == (my_val <= partner_val);
        int should_swap = is_asc != is_asc_order;
        my_val = should_swap ? partner_val : my_val;
    }

    // Copy data back to global memory
    data[offset + tid] = my_val;
}


__host__
int bitonic_sort_v4(int *host_data, int n, int descending) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int max_size = BLOCK_SIZE > n ? n : BLOCK_SIZE;
    int max_step  = max_size >> 1;  // half block size
    size_t shared_mem_block_bytes = max_size * sizeof(int);

    int *device_data = NULL;
    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_intra_block_sort_v4<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, max_size);
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
        kernel_intra_block_refine_v4<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, size, max_size);
        if (post_launch_barrier_and_check()) {
            cudaFree(device_data);
            return EXIT_FAILURE;
        }
    }

    // If descending order is requested, reverse the data
    if (descending) {
        kernel_reverse<<<numBlocks, BLOCK_SIZE>>>(device_data, n);
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
