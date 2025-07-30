#include "bitonic_sort_v3.cuh"
#include "util.cuh"
#include "config.cuh"
#include "bitonic_sort_v2.cuh"


__global__
static void kernel_compare_and_swap_v3(int *data,
                                int ascending,
                                int size,
                                int step,
                                int log2step)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int blockPair = tid >> log2step;
    int offset    = tid & (step - 1);
    int base = blockPair << step;
    int i    = base + offset;
    int j    = i ^ step;  // guaranteed j>i

    int is_asc = ((i & size) == 0) ? ascending : !ascending;
    compare_and_swap(data, i, j, is_asc);
}


__host__
int bitonic_sort_v3(int *host_data, int n, int ascending) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlocksHalf = ((n >> 1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int max_size = BLOCK_SIZE > n ? n : BLOCK_SIZE;
    int max_step  = max_size >> 1;  // half block size
    size_t shared_mem_block_bytes = max_size * sizeof(int);

    int *device_data = NULL;
    if (host_to_device_data(host_data, n, &device_data) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    // Intra block sorting
    kernel_intra_block_sort_v2<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, ascending);
    if (post_launch_barrier_and_check()) {
        cudaFree(device_data);
        return EXIT_FAILURE;
    }

    // merge across blocks
    for (int size = max_size << 1; size <= n; size <<= 1) {
        
        for (int step = size >> 1; step > max_step; step >>= 1) {

            // Inter block merge
            int log2step = __builtin_ctz(step);
            kernel_compare_and_swap_v3<<<numBlocksHalf, BLOCK_SIZE>>>(device_data, ascending, size, step, log2step);
            if (post_launch_barrier_and_check()) {
                cudaFree(device_data);
                return EXIT_FAILURE;
            }
        }
        // intra-block refinement
        kernel_intra_block_refine_v2<<<numBlocks, BLOCK_SIZE, shared_mem_block_bytes>>>(device_data, n, ascending, size);
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
