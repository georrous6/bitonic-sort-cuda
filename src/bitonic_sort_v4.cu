#include "bitonic_sort_v4.cuh"
#include "util.cuh"
#include "config.cuh"


__device__ __forceinline__
static int get_lower_partner(int tid, int step, int log2step) {
    int blockPair = tid >> log2step;
    int offset    = tid & (step - 1);
    int base = blockPair << (log2step + 1);
    return base + offset;
}


__global__
static void kernel_compare_and_swap_v4(int *data, int size, int step, int log2step)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = get_lower_partner(tid, step, log2step);
    int j = i + step;

    int is_asc = (i & size) == 0;
    compare_and_swap(data, i, j, is_asc);
}



__global__ 
static void kernel_intra_block_sort_v4(int *data, int chunk_size) {
    
    extern __shared__ int s_data[];
    int offset = 2 * blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    int local_tid_2 = 2 * local_tid;

    // Load data into shared memory
    s_data[local_tid_2] = data[offset + local_tid_2];
    s_data[local_tid_2 + 1] = data[offset + local_tid_2 + 1];

    __syncthreads();

    for (int size = 2; size <= chunk_size; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            int log2step = __ffs(step) - 1;
            int i = get_lower_partner(local_tid, step, log2step);
            int j = i + step;
            int global_id = offset + i;
            int is_asc = ((global_id & size) == 0);
            compare_and_swap(s_data, i, j, is_asc);
            __syncthreads();
        }
    }

    // Copy data back to global memory
    data[offset + local_tid_2] = s_data[local_tid_2];
    data[offset + local_tid_2 + 1] = s_data[local_tid_2 + 1];
}


__global__
static void kernel_intra_block_refine_v4(int *data, int size, int chunk_size) {

    extern __shared__ int s_data[];
    int offset = 2 * blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    int local_tid_2 = 2 * local_tid;

    // Load data into shared memory
    s_data[local_tid_2] = data[offset + local_tid_2];
    s_data[local_tid_2 + 1] = data[offset + local_tid_2 + 1];

    __syncthreads();

    for (int step = chunk_size >> 1; step > 0; step >>= 1) {
        int log2step = __ffs(step) - 1;
        int i = get_lower_partner(local_tid, step, log2step);
        int j  = i + step;
        int global_id = offset + i;
        int is_asc = ((global_id & size) == 0);
        compare_and_swap(s_data, i, j, is_asc);
        __syncthreads();
    }

    // Copy data back to global memory
    data[offset + local_tid_2] = s_data[local_tid_2];
    data[offset + local_tid_2 + 1] = s_data[local_tid_2 + 1];
}


__host__
int bitonic_sort_v4(int *host_data, int n, int descending) {
    int n_half = n >> 1;
    int threadsPerBlock = BLOCK_SIZE > n_half ? n_half : BLOCK_SIZE;
    int numBlocks = (n_half + threadsPerBlock - 1) / threadsPerBlock;
    int chunk_size = threadsPerBlock << 1;
    int max_step  = chunk_size >> 1;
    size_t shared_mem_block_bytes = chunk_size * sizeof(int);

    int *device_data = NULL;
    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_intra_block_sort_v4<<<numBlocks, threadsPerBlock, shared_mem_block_bytes>>>(device_data, chunk_size);
    if (post_launch_barrier_and_check()) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    // merge across blocks
    for (int size = chunk_size << 1; size <= n; size <<= 1) {
        
        for (int step = size >> 1; step > max_step; step >>= 1) {

            // Inter block merge
            int log2step = __builtin_ctz(step);
            kernel_compare_and_swap_v4<<<numBlocks, threadsPerBlock>>>(device_data, size, step, log2step);
            if (post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
        // intra-block refinement
        kernel_intra_block_refine_v4<<<numBlocks, threadsPerBlock, shared_mem_block_bytes>>>(device_data, size, chunk_size);
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