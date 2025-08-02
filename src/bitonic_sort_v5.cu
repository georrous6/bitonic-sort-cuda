#include "bitonic_sort_v5.cuh"
#include "bitonic_sort_v4.cuh"
#include "util.cuh"
#include "config.cuh"


namespace {

__global__ 
void kernel_intra_block_sort(int *data, int chunk_size) {
    
    extern __shared__ int s_data[];
    int offset = 2 * blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    int local_tid_2 = 2 * local_tid;
    constexpr int HALF_WARP_SIZE = WARP_SIZE >> 1;
    int is_in_lower_half = (local_tid & HALF_WARP_SIZE) == 0;
    int is_in_upper_half = !is_in_lower_half;

    // Load data into shared memory
    s_data[local_tid_2 + is_in_upper_half] = data[offset + local_tid_2 + is_in_upper_half];
    s_data[local_tid_2 + is_in_lower_half] = data[offset + local_tid_2 + is_in_lower_half];

    __syncthreads();

    for (int size = 2; size <= chunk_size; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            int log2step = __ffs(step) - 1;
            int i = util::get_lower_partner(local_tid, step, log2step);
            int j = i + step;
            int global_id = offset + i;
            int is_asc = ((global_id & size) == 0);
            util::compare_and_swap(s_data, i, j, is_asc);
            __syncthreads();
        }
    }

    // Copy data back to global memory
    data[offset + local_tid_2 + is_in_upper_half] = s_data[local_tid_2 + is_in_upper_half];
    data[offset + local_tid_2 + is_in_lower_half] = s_data[local_tid_2 + is_in_lower_half];
}


__global__
void kernel_intra_block_refine(int *data, int size, int chunk_size) {

    extern __shared__ int s_data[];
    int offset = 2 * blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    int local_tid_2 = 2 * local_tid;
    constexpr int HALF_WARP_SIZE = WARP_SIZE >> 1;
    int is_in_lower_half = (local_tid & HALF_WARP_SIZE) == 0;
    int is_in_upper_half = !is_in_lower_half;

    // Load data into shared memory
    s_data[local_tid_2 + is_in_upper_half] = data[offset + local_tid_2 + is_in_upper_half];
    s_data[local_tid_2 + is_in_lower_half] = data[offset + local_tid_2 + is_in_lower_half];

    __syncthreads();

    for (int step = chunk_size >> 1; step > 0; step >>= 1) {
        int log2step = __ffs(step) - 1;
        int i = util::get_lower_partner(local_tid, step, log2step);
        int j  = i + step;
        int global_id = offset + i;
        int is_asc = ((global_id & size) == 0);
        util::compare_and_swap(s_data, i, j, is_asc);
        __syncthreads();
    }

    // Copy data back to global memory
    data[offset + local_tid_2 + is_in_upper_half] = s_data[local_tid_2 + is_in_upper_half];
    data[offset + local_tid_2 + is_in_lower_half] = s_data[local_tid_2 + is_in_lower_half];
}

}


namespace v5 {

__host__
int bitonic_sort(int *host_data, int n, int descending) {
    int n_half = n >> 1;
    int threadsPerBlock = BLOCK_SIZE > n_half ? n_half : BLOCK_SIZE;
    int numBlocks = (n_half + threadsPerBlock - 1) / threadsPerBlock;
    int chunk_size = threadsPerBlock << 1;
    int max_step  = chunk_size >> 1;
    size_t shared_mem_block_bytes = chunk_size * sizeof(int);

    int *device_data = NULL;
    if (util::host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_intra_block_sort<<<numBlocks, threadsPerBlock, shared_mem_block_bytes>>>(device_data, chunk_size);
    if (util::post_launch_barrier_and_check()) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    // merge across blocks
    for (int size = chunk_size << 1; size <= n; size <<= 1) {
        
        for (int step = size >> 1; step > max_step; step >>= 1) {

            // Inter block merge
            int log2step = __builtin_ctz(step);
            v4::kernel_compare_and_swap<<<numBlocks, threadsPerBlock>>>(device_data, size, step, log2step);
            if (util::post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
        // intra-block refinement
        kernel_intra_block_refine<<<numBlocks, threadsPerBlock, shared_mem_block_bytes>>>(device_data, size, chunk_size);
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