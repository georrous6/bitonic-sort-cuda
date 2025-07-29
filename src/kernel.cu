#include "kernel.cuh"

__global__
void wakeup_kernel(void) {}


__global__ 
void kernel_v0(int *data, int n, int ascending, int size, int step) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        int j = i ^ step;
        if (j > i) {
            int is_ascending = ((i & size) == 0) ? ascending : !ascending;
            compare_and_swap(data, i, j, is_ascending);
        }
    }
}


__global__ void kernel_v1_alternating_sort(int *data, int chunk_size, int ascending) {
    int local_tid    = threadIdx.x;
    int block_offset = blockIdx.x * chunk_size;
    int stride       = blockDim.x;

    // flip direction for odd-numbered blocks
    int dir = (blockIdx.x & 1) ? !ascending : ascending;

    for (int size = 2; size <= chunk_size; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            for (int i = local_tid; i < chunk_size; i += stride) {
                int gi = block_offset + i;
                int j  = gi ^ step;
                if (j > gi && j < block_offset + chunk_size) {
                    int is_asc = ((gi & size) == 0) ? dir : !dir;
                    compare_and_swap(data, gi, j, is_asc);
                }
            }
            __syncthreads();
        }
    }
}


__global__ 
void kernel_v1_intra_block_sort(int *data, int chunk_size, int ascending, int size, int step_start) {
    int local_tid    = threadIdx.x;
    int block_offset = blockIdx.x * chunk_size;
    int stride       = blockDim.x;

    for (int step = step_start; step > 0; step >>= 1) {
        for (int i = local_tid; i < chunk_size; i += stride) {
            int gi = block_offset + i;
            int j  = gi ^ step;
            if (j > gi && j < block_offset + chunk_size) {
                int is_asc = ((gi & size) == 0) ? ascending : !ascending;
                compare_and_swap(data, gi, j, is_asc);
            }
        }
        __syncthreads();
    }
}
