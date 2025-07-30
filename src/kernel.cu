#include "kernel.cuh"

__global__
void wakeup_kernel(void) {}


__global__ 
void kernel_v0_compare_and_swap(int *data, int n, int ascending, int size, int step) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int j = i ^ step;
    if (j > i) {
        int is_asc = ((i & size) == 0) ? ascending : !ascending;
        compare_and_swap(data, i, j, is_asc);
    }
}


__global__ void kernel_v1_intra_block_sort(int *data, int n, int ascending) {
    
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
void kernel_v1_intra_block_refine(int *data, int n, int ascending, int size) {

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


__global__ 
void kernel_v2_intra_block_sort(int *data, int n, int ascending) {

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
                int is_asc = ((global_id & size) == 0) ? ascending : !ascending;
                compare_and_swap(s_data, tid, j, is_asc);
            }
            __syncthreads();
        }
    }

    // Copy data back to global memory
    data[offset + tid] = s_data[tid];
}


__global__
void kernel_v2_intra_block_refine(int *data, int n, int ascending, int size) {

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
            int is_asc = ((global_id & size) == 0) ? ascending : !ascending;
            compare_and_swap(s_data, tid, j, is_asc);
        }
        __syncthreads();
    }

    // Copy data back to global memory
    data[offset + tid] = s_data[tid];
}
