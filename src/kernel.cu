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


__global__ void kernel_v1_alternating_sort(int *data, int n, int chunk_size, int ascending) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int size = 2; size <= chunk_size; size <<= 1) {
        for (int step = size >> 1; step > 0; step >>= 1) {
            for (int i = idx; i < n; i += stride) {
                int j  = i ^ step;
                if (j > i) {
                    int is_asc = ((i & size) == 0) ? ascending : !ascending;
                    compare_and_swap(data, i, j, is_asc);
                }
            }
            __syncthreads();
        }
    }
}


__global__ 
void kernel_v1_intra_block_sort(int *data, int n, int chunk_size, int ascending, int size, int step_start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int step = step_start; step > 0; step >>= 1) {
        for (int i = idx; i < n; i += stride) {
            int j  = i ^ step;
            if (j > i) {
                int is_asc = ((i & size) == 0) ? ascending : !ascending;
                compare_and_swap(data, i, j, is_asc);
            }
        }
        __syncthreads();
    }
}
